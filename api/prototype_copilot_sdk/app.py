"""
Azure Support Assistant — rewritten for github-copilot-sdk + stdio MCP.

Replaces Azure Durable Functions with a FastAPI server that supports:
  - Direct HTTP responses (no polling)
  - SSE streaming for real-time responses
  - Both agents can run in parallel via asyncio.gather
  - Two trigger modes: web UI (user token) and webhook (managed identity)

Architecture changes from agent-framework version:
  - AgentFunctionApp (Azure Functions) → FastAPI (Container App)
  - Durable Functions orchestration → asyncio.gather for parallel agents
  - REST + polling → SSE streaming (optional, falls back to blocking)
  - MCP Container App (HTTP + OBO) → azure-mcp subprocess (stdio)

Dual auth for ARM access:
  - Web UI:  user's ARM token (per-user RBAC) — scope: management.azure.com
  - Webhook: managed identity (service RBAC) — no token needed

Infrastructure eliminated:
  - MCP Container App + Container Apps Environment
  - MCP Server/Client App Registrations + Service Principals
  - Federated Identity Credential for OBO
  - MCP-specific User-Assigned Managed Identity
  - Custom MCP audience scopes (api://.../Mcp.Tools.ReadWrite)

Frontend auth simplification:
  - MSAL scope: https://management.azure.com/user_impersonation (ARM direct)
  - No more REACT_APP_MCP_SERVER_CLIENT_ID
  - No more custom MCP audience

Requires:
  - github-copilot-sdk (pip install github-copilot-sdk)
  - fastapi + uvicorn
  - Copilot CLI installed (or sidecar)
  - azure-mcp binary in container image
"""
import os
import json
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from azure_agent import run_azure_query, run_azure_query_streaming
from ticket_agent import run_ticket_query

logger = logging.getLogger(__name__)

app = FastAPI(title="Azure Support Assistant (Copilot SDK)")

# CORS — allow the React frontend (local dev + production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================== Request/Response Models ==================

class QueryRequest(BaseModel):
    """Request model for web UI queries (user provides ARM token)."""
    query: str
    userAccessToken: Optional[str] = None  # ARM-scoped token from logged-in user


class WebhookRequest(BaseModel):
    """Request model for webhook-triggered queries (no user, uses MI)."""
    query: str
    ticket_id: Optional[str] = None  # For correlating responses back to tickets


class QueryResponse(BaseModel):
    result: str | None = None
    error: str | None = None


class CombinedResponse(BaseModel):
    azure: QueryResponse
    tickets: QueryResponse


# ================== Web UI Endpoints (user token → per-user RBAC) ==================

@app.post("/api/query/azure", response_model=QueryResponse)
async def azure_query(req: QueryRequest):
    """Run an Azure resource query. User's ARM token used if provided."""
    if not req.query:
        raise HTTPException(status_code=400, detail="Missing 'query'")

    try:
        result = await run_azure_query(req.query, req.userAccessToken)
        return QueryResponse(result=result)
    except Exception as ex:
        logger.error(f"Azure query error: {ex}", exc_info=True)
        return QueryResponse(error=str(ex))


@app.post("/api/query/azure/stream")
async def azure_query_stream(req: QueryRequest):
    """Run an Azure resource query with SSE streaming."""
    if not req.query:
        raise HTTPException(status_code=400, detail="Missing 'query'")

    async def event_generator():
        async for chunk in run_azure_query_streaming(req.query, req.userAccessToken):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/query/tickets", response_model=QueryResponse)
async def ticket_query(req: QueryRequest):
    """Run a ticket search query."""
    if not req.query:
        raise HTTPException(status_code=400, detail="Missing 'query'")

    try:
        result = await run_ticket_query(req.query)
        return QueryResponse(result=result)
    except Exception as ex:
        logger.error(f"Ticket query error: {ex}", exc_info=True)
        return QueryResponse(error=str(ex))


@app.post("/api/query/both", response_model=CombinedResponse)
async def combined_query(req: QueryRequest):
    """Run both agents in parallel and combine results."""
    if not req.query:
        raise HTTPException(status_code=400, detail="Missing 'query'")

    azure_task = run_azure_query(req.query, req.userAccessToken)
    ticket_task = run_ticket_query(req.query)

    results = await asyncio.gather(azure_task, ticket_task, return_exceptions=True)

    azure_result = results[0]
    ticket_result = results[1]

    return CombinedResponse(
        azure=QueryResponse(
            result=azure_result if isinstance(azure_result, str) else None,
            error=str(azure_result) if isinstance(azure_result, Exception) else None,
        ),
        tickets=QueryResponse(
            result=ticket_result if isinstance(ticket_result, str) else None,
            error=str(ticket_result) if isinstance(ticket_result, Exception) else None,
        ),
    )


# ================== Webhook Endpoint (MI auth → service RBAC) ==================

@app.post("/api/webhook/ticket-created", response_model=CombinedResponse)
async def ticket_created_webhook(req: WebhookRequest):
    """Handle ticket creation webhook.

    No user token — azure-mcp uses the container's managed identity.
    The MI should have Reader on the subscriptions the support team covers.

    Typical flow:
      1. Ticket system fires webhook on new ticket
      2. This endpoint runs both agents:
         - Azure agent investigates the resources mentioned in the ticket
         - Ticket agent fetches full ticket context
      3. Response can be posted back to the ticket as an internal note
    """
    if not req.query:
        raise HTTPException(status_code=400, detail="Missing 'query'")

    logger.info(f"Webhook: ticket created — {req.ticket_id or 'no ticket ID'}")

    # No userAccessToken → azure-mcp falls back to managed identity
    azure_task = run_azure_query(req.query, user_access_token=None)
    ticket_task = run_ticket_query(req.query)

    results = await asyncio.gather(azure_task, ticket_task, return_exceptions=True)

    azure_result = results[0]
    ticket_result = results[1]

    return CombinedResponse(
        azure=QueryResponse(
            result=azure_result if isinstance(azure_result, str) else None,
            error=str(azure_result) if isinstance(azure_result, Exception) else None,
        ),
        tickets=QueryResponse(
            result=ticket_result if isinstance(ticket_result, str) else None,
            error=str(ticket_result) if isinstance(ticket_result, Exception) else None,
        ),
    )


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "framework": "github-copilot-sdk"}
