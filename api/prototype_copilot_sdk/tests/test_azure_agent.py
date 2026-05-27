"""
Integration tests for the Azure resource agent.

These tests exercise the full CopilotClient → CLI → LLM → azure-mcp (stdio) pipeline.
Requires:
  - LLM credentials (GITHUB_TOKEN or AZURE_OPENAI_ENDPOINT)
  - azure-mcp binary (AZURE_MCP_PATH, defaults to /usr/local/bin/azure-mcp)
  - Azure credentials (az login or managed identity)

Run:
    GITHUB_TOKEN=ghp_... AZURE_MCP_PATH=/path/to/azure-mcp \
      pytest tests/test_azure_agent.py -v
"""
import asyncio
import os

import pytest

from tests.conftest import requires_llm, requires_azure_mcp


@requires_llm
@requires_azure_mcp
@pytest.mark.asyncio
async def test_azure_agent_list_resources(copilot_config, provider_config):
    """Ask the agent to list resource groups — exercises MCP tools end-to-end."""
    from copilot import CopilotClient
    from copilot.session import PermissionHandler
    from copilot.generated.session_events import (
        AssistantMessageData,
        SessionIdleData,
    )
    from azure_agent import (
        AZURE_AGENT_INSTRUCTIONS,
        _get_copilot_config,
        _get_mcp_server_config,
    )

    parts: list[str] = []
    done = asyncio.Event()

    def on_event(event):
        match event.data:
            case AssistantMessageData() as data:
                parts.append(data.content)
            case SessionIdleData():
                done.set()

    # Use the agent's own config builder (includes MCP env vars)
    config = _get_copilot_config()

    session_kwargs = {
        "model": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
        "system_message": {"content": AZURE_AGENT_INSTRUCTIONS},
        "on_permission_request": PermissionHandler.approve_all,
        "mcp_servers": {
            "azure": _get_mcp_server_config(user_access_token=None),
        },
    }
    if provider_config:
        session_kwargs["provider"] = provider_config

    async with CopilotClient(config) as client:
        async with await client.create_session(**session_kwargs) as session:
            session.on(on_event)
            await session.send("List my Azure resource groups")
            await asyncio.wait_for(done.wait(), timeout=60)

    response = "".join(parts)
    assert response, "Agent returned an empty response"
    assert "error querying" not in response.lower(), f"Agent returned error: {response[:300]}"


@requires_llm
@requires_azure_mcp
@pytest.mark.asyncio
async def test_run_azure_query_function():
    """Test run_azure_query() helper directly."""
    from azure_agent import run_azure_query

    result = await run_azure_query(
        "How many subscriptions do I have access to?",
        user_access_token=None,  # Use MI / DefaultAzureCredential
    )

    assert result, "run_azure_query returned empty"
    assert "error querying" not in result.lower(), f"Got error: {result[:300]}"


@requires_llm
@requires_azure_mcp
@pytest.mark.asyncio
async def test_azure_agent_streaming():
    """Test streaming variant collects chunks."""
    from azure_agent import run_azure_query_streaming

    chunks: list[str] = []
    async for chunk in run_azure_query_streaming(
        "What Azure subscriptions do I have?",
        user_access_token=None,
    ):
        chunks.append(chunk)

    assert chunks, "Streaming returned no chunks"
    full = "".join(chunks)
    assert "error" not in full.lower()[:50], f"Streaming error: {full[:200]}"
