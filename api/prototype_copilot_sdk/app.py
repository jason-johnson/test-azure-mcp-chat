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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
        logger.info(f"Azure query received: {req.query[:50]}... | Token provided: {bool(req.userAccessToken)}")
        result = await run_azure_query(req.query, req.userAccessToken)
        logger.info(f"Azure query result length: {len(result) if result else 0}")
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
@app.get("/api/debug/env")
async def debug_env():
    """Show environment configuration."""
    return {
        "azure_client_id": os.getenv("AZURE_CLIENT_ID", "not set"),
        "copilot_model": os.getenv("COPILOT_MODEL", "not set"),
        "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "not set"),
        "azure_openai_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "not set"),
        "azure_mcp_command": os.getenv("AZURE_MCP_COMMAND", "npx"),
        "azure_mcp_args": os.getenv("AZURE_MCP_ARGS", "-y @azure/mcp server start --mode all --read-only"),
        "node_available": os.path.exists("/usr/local/bin/node"),
        "npx_available": os.path.exists("/usr/local/bin/npx"),
    }


@app.get("/api/debug/test-mcp")
async def test_mcp():
    """Test MCP subprocess startup."""
    import subprocess
    try:
        # Test if npx can run
        result = subprocess.run(
            ["npx", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        npx_version = result.stdout.strip() if result.returncode == 0 else f"failed: {result.stderr}"
        
        # Test if we can spawn the MCP server
        mcp_test = subprocess.run(
            ["npx", "-y", "@azure/mcp", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )
        mcp_version = mcp_test.stdout.strip() if mcp_test.returncode == 0 else f"failed: {mcp_test.stderr}"
        
        return {
            "npx_version": npx_version,
            "mcp_version": mcp_version,
            "mcp_command_works": mcp_test.returncode == 0
        }
    except Exception as ex:
        return {"error": str(ex), "type": type(ex).__name__}


@app.get("/api/debug/test-session")
async def test_session():
    """Test Copilot SDK session creation with MCP."""
    try:
        from azure_agent import _get_mcp_server_config, _get_copilot_config
        from copilot import CopilotClient
        
        result = {}
        
        # Try to create a session with MCP
        async with CopilotClient(_get_copilot_config()) as client:
            result["client_created"] = True
            
            mcp_config = _get_mcp_server_config(None)  # Test with MI
            result["mcp_config"] = mcp_config
            
            try:
                async with await client.create_session(
                    model=os.getenv("COPILOT_MODEL", "gpt-4o"),
                    mcp_servers={"azure": mcp_config}
                ) as session:
                    result["session_created"] = True
                    
                    # Give MCP time to start
                    await asyncio.sleep(3)
                    
                    # Try to send a simple query
                    response_parts = []
                    done = asyncio.Event()
                    
                    def on_event(event):
                        from copilot.generated.session_events import (
                            AssistantMessageData,
                            SessionIdleData,
                            SessionErrorData,
                        )
                        match event.data:
                            case AssistantMessageData() as data:
                                response_parts.append(data.content)
                            case SessionIdleData():
                                done.set()
                            case SessionErrorData() as data:
                                response_parts.append(f"ERROR: {data.message}")
                                done.set()
                    
                    session.on(on_event)
                    await session.send("What is 2+2?")
                    await asyncio.wait_for(done.wait(), timeout=30)
                    
                    result["test_response"] = "".join(response_parts)
            except Exception as session_ex:
                result["session_error"] = str(session_ex)
                result["session_error_type"] = type(session_ex).__name__
                
        return result
    except Exception as ex:
        logger.error(f"Session test failed: {ex}", exc_info=True)
        return {"error": str(ex), "type": type(ex).__name__}


# ================== Static Frontend (served last, catch-all for SPA) ==================

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir + "/static"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve the React SPA for any non-API route."""
        file_path = os.path.join(_static_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_static_dir, "index.html"))
