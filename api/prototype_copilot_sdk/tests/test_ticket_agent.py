"""
Integration tests for the ticket agent.

These tests exercise the full CopilotClient → CLI → LLM → stub tools pipeline.
The ticket agent uses @define_tool stubs (no MCP server needed), so only LLM
credentials are required.

Run:
    GITHUB_TOKEN=ghp_... pytest tests/test_ticket_agent.py -v
    # or with Azure OpenAI:
    AZURE_OPENAI_ENDPOINT=https://... pytest tests/test_ticket_agent.py -v
"""
import asyncio

import pytest

from tests.conftest import requires_llm


@requires_llm
@pytest.mark.asyncio
async def test_ticket_search_returns_results(copilot_config, provider_config):
    """Search for 'Contoso' tickets — should find TKT-1042 and TKT-1035."""
    from copilot import CopilotClient
    from copilot.session import PermissionHandler
    from copilot.generated.session_events import (
        AssistantMessageData,
        SessionIdleData,
    )
    from ticket_agent import (
        TICKET_AGENT_INSTRUCTIONS,
        search_tickets,
        get_ticket_details,
    )

    parts: list[str] = []
    done = asyncio.Event()

    def on_event(event):
        match event.data:
            case AssistantMessageData() as data:
                parts.append(data.content)
            case SessionIdleData():
                done.set()

    session_kwargs = {
        "model": "gpt-4o-mini",
        "system_message": {"content": TICKET_AGENT_INSTRUCTIONS},
        "tools": [search_tickets, get_ticket_details],
        "on_permission_request": PermissionHandler.approve_all,
    }
    if provider_config:
        session_kwargs["provider"] = provider_config

    async with CopilotClient(copilot_config) as client:
        async with await client.create_session(**session_kwargs) as session:
            session.on(on_event)
            await session.send("Search for all Contoso tickets")
            await asyncio.wait_for(done.wait(), timeout=30)

    response = "".join(parts)
    assert response, "Agent returned an empty response"
    # The stub data has two Contoso tickets
    assert "TKT-1042" in response or "1042" in response, (
        f"Expected TKT-1042 in response, got: {response[:200]}"
    )


@requires_llm
@pytest.mark.asyncio
async def test_ticket_details_returns_thread(copilot_config, provider_config):
    """Request details for TKT-1042 — should include the conversation thread."""
    from copilot import CopilotClient
    from copilot.session import PermissionHandler
    from copilot.generated.session_events import (
        AssistantMessageData,
        SessionIdleData,
    )
    from ticket_agent import (
        TICKET_AGENT_INSTRUCTIONS,
        search_tickets,
        get_ticket_details,
    )

    parts: list[str] = []
    done = asyncio.Event()

    def on_event(event):
        match event.data:
            case AssistantMessageData() as data:
                parts.append(data.content)
            case SessionIdleData():
                done.set()

    session_kwargs = {
        "model": "gpt-4o-mini",
        "system_message": {"content": TICKET_AGENT_INSTRUCTIONS},
        "tools": [search_tickets, get_ticket_details],
        "on_permission_request": PermissionHandler.approve_all,
    }
    if provider_config:
        session_kwargs["provider"] = provider_config

    async with CopilotClient(copilot_config) as client:
        async with await client.create_session(**session_kwargs) as session:
            session.on(on_event)
            await session.send("Get the full details and thread for ticket TKT-1042")
            await asyncio.wait_for(done.wait(), timeout=30)

    response = "".join(parts)
    assert response, "Agent returned an empty response"
    # TKT-1042 thread mentions NSG rules
    assert "NSG" in response or "nsg" in response.lower() or "network security" in response.lower(), (
        f"Expected NSG reference in TKT-1042 details, got: {response[:300]}"
    )


@requires_llm
@pytest.mark.asyncio
async def test_ticket_search_with_status_filter(copilot_config, provider_config):
    """Search for resolved tickets — should find TKT-1029 (SSL cert)."""
    from copilot import CopilotClient
    from copilot.session import PermissionHandler
    from copilot.generated.session_events import (
        AssistantMessageData,
        SessionIdleData,
    )
    from ticket_agent import (
        TICKET_AGENT_INSTRUCTIONS,
        search_tickets,
        get_ticket_details,
    )

    parts: list[str] = []
    done = asyncio.Event()

    def on_event(event):
        match event.data:
            case AssistantMessageData() as data:
                parts.append(data.content)
            case SessionIdleData():
                done.set()

    session_kwargs = {
        "model": "gpt-4o-mini",
        "system_message": {"content": TICKET_AGENT_INSTRUCTIONS},
        "tools": [search_tickets, get_ticket_details],
        "on_permission_request": PermissionHandler.approve_all,
    }
    if provider_config:
        session_kwargs["provider"] = provider_config

    async with CopilotClient(copilot_config) as client:
        async with await client.create_session(**session_kwargs) as session:
            session.on(on_event)
            await session.send("Show me all resolved tickets")
            await asyncio.wait_for(done.wait(), timeout=30)

    response = "".join(parts)
    assert response, "Agent returned an empty response"
    assert "TKT-1029" in response or "1029" in response or "SSL" in response.upper(), (
        f"Expected TKT-1029 / SSL in response, got: {response[:200]}"
    )


@requires_llm
@pytest.mark.asyncio
async def test_run_ticket_query_function(provider_config):
    """Test the run_ticket_query() helper directly (full agent loop)."""
    import os

    # If using Azure OpenAI, the env vars need to be set for _get_azure_provider_config
    # If using GitHub Copilot, run_ticket_query() needs to work without custom provider

    from ticket_agent import run_ticket_query

    result = await run_ticket_query("What urgent tickets do we have?")

    assert result, "run_ticket_query returned empty"
    assert "error" not in result.lower()[:50], f"Got error: {result[:200]}"
    # TKT-1025 is the only urgent ticket in stub data
    assert "1025" in result or "SQL" in result or "urgent" in result.lower(), (
        f"Expected TKT-1025 (urgent) in response, got: {result[:200]}"
    )
