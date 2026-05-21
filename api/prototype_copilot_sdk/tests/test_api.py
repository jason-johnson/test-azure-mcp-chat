"""
Integration tests for the FastAPI endpoints.

These tests exercise the full HTTP stack: routing, validation, serialization,
error handling, and (when LLM credentials are available) end-to-end responses.

Run without LLM (validation + error paths only):
    pytest tests/test_api.py -v -k "not llm"

Run with LLM (full end-to-end):
    GITHUB_TOKEN=ghp_... pytest tests/test_api.py -v
"""
import json
import os
import sys

import pytest
import httpx

from tests.conftest import requires_llm, requires_azure_mcp

# Add parent dir so `app` module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app  # noqa: E402


@pytest.fixture
def client():
    """HTTPX async client bound to the FastAPI app."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


# ================== Health check ==================

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["framework"] == "github-copilot-sdk"


# ================== Request validation ==================

@pytest.mark.asyncio
async def test_azure_query_empty_query(client):
    resp = await client.post("/api/query/azure", json={"query": ""})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_ticket_query_empty_query(client):
    resp = await client.post("/api/query/tickets", json={"query": ""})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_combined_query_empty_query(client):
    resp = await client.post("/api/query/both", json={"query": ""})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_webhook_empty_query(client):
    resp = await client.post("/api/webhook/ticket-created", json={"query": ""})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_azure_query_missing_body(client):
    resp = await client.post("/api/query/azure", content=b"not json", headers={"content-type": "application/json"})
    assert resp.status_code == 422  # Pydantic validation error


# ================== Response shape validation ==================

@pytest.mark.asyncio
async def test_query_response_shape(client):
    """Verify QueryResponse has result/error fields even on error."""
    # This will fail at the agent level (no LLM creds), but should still
    # return a 200 with an error field rather than a 500.
    resp = await client.post("/api/query/tickets", json={"query": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data or "error" in data


@pytest.mark.asyncio
async def test_combined_response_shape(client):
    """Verify CombinedResponse has azure/tickets sub-objects."""
    resp = await client.post("/api/query/both", json={"query": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "azure" in data
    assert "tickets" in data
    assert "result" in data["azure"] or "error" in data["azure"]
    assert "result" in data["tickets"] or "error" in data["tickets"]


# ================== SSE streaming shape ==================

@pytest.mark.asyncio
async def test_stream_endpoint_returns_sse(client):
    """Verify the stream endpoint returns text/event-stream content type."""
    resp = await client.post(
        "/api/query/azure/stream",
        json={"query": "test"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


# ================== End-to-end with LLM ==================

@requires_llm
@pytest.mark.asyncio
async def test_ticket_query_e2e(client):
    """Full round-trip: POST → agent → stub tools → response."""
    resp = await client.post(
        "/api/query/tickets",
        json={"query": "Show me all open high priority tickets"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("result"), f"Expected result, got: {data}"
    # Stub data has TKT-1042 and TKT-1035 as open+high
    result = data["result"]
    assert "1042" in result or "1035" in result, f"Expected ticket IDs in: {result[:200]}"


@requires_llm
@requires_azure_mcp
@pytest.mark.asyncio
async def test_azure_query_e2e(client):
    """Full round-trip with azure-mcp for Azure resource queries."""
    resp = await client.post(
        "/api/query/azure",
        json={"query": "List my resource groups"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("result") or data.get("error"), f"Empty response: {data}"


@requires_llm
@requires_azure_mcp
@pytest.mark.asyncio
async def test_combined_query_e2e(client):
    """Both agents run in parallel and return combined results."""
    resp = await client.post(
        "/api/query/both",
        json={"query": "What Azure resources are affected by open tickets?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Both sub-objects should have either result or error
    assert data["azure"].get("result") or data["azure"].get("error")
    assert data["tickets"].get("result") or data["tickets"].get("error")


@requires_llm
@requires_azure_mcp
@pytest.mark.asyncio
async def test_webhook_e2e(client):
    """Webhook endpoint uses MI auth (no user token)."""
    resp = await client.post(
        "/api/webhook/ticket-created",
        json={
            "query": "VM not responding in East US, ticket TKT-1042",
            "ticket_id": "TKT-1042",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["azure"].get("result") or data["azure"].get("error")
    assert data["tickets"].get("result") or data["tickets"].get("error")


@requires_llm
@pytest.mark.asyncio
async def test_stream_e2e(client):
    """SSE streaming returns data chunks and [DONE] sentinel."""
    resp = await client.post(
        "/api/query/azure/stream",
        json={"query": "How many subscriptions do I have?"},
    )
    assert resp.status_code == 200
    body = resp.text
    # SSE format: lines starting with "data: "
    lines = [l for l in body.strip().split("\n") if l.startswith("data: ")]
    assert lines, "No SSE data lines received"
    assert any("[DONE]" in l for l in lines), "Missing [DONE] sentinel"
