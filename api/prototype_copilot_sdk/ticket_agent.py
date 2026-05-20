"""
Ticket Agent — rewritten for github-copilot-sdk.

Dual-mode: uses MCP via Copilot CLI (stdio) when FRESHDESK_MCP_URI is configured,
otherwise falls back to @define_tool stub functions.

Key differences from agent-framework version:
  - @tool decorator replaced with @define_tool + Pydantic models
  - Agent loop managed by CopilotClient session
  - Same dual-mode (MCP vs stubs) design preserved
  - No user token needed — MCP servers run as stdio subprocesses
"""
import os
import asyncio
import logging

from typing import Optional

from pydantic import BaseModel, Field
from copilot import CopilotClient, define_tool
from copilot.generated.session_events import (
    AssistantMessageData,
    SessionIdleData,
)
from copilot.session import PermissionHandler

logger = logging.getLogger(__name__)

TICKET_AGENT_INSTRUCTIONS = """You are a support ticket assistant for a technical support team.
You have access to the Freshdesk ticket system and can search for and retrieve ticket details.

Guidelines:
- Search for tickets using keywords, ticket IDs, or customer information.
- Present ticket details clearly: ticket ID, subject, status, priority, assignee, and last update.
- Summarize ticket threads concisely — highlight the problem, key updates, and current status.
- When multiple tickets match, list them sorted by relevance or recency.
- Correlate tickets when asked — find related issues, recurring problems, or patterns.
- Never modify ticket data unless the user explicitly asks.
- If the ticket system is unavailable, inform the user clearly."""


# ================== Stub Tools (Pydantic + @define_tool) ==================

# ---- Stub data ----
_STUB_TICKETS = [
    {"id": "TKT-1042", "subject": "VM not responding in East US region",
     "status": "open", "priority": "high", "assignee": "Sarah Chen",
     "created": "2026-05-03", "customer": "Contoso Ltd",
     "description": "Production VM in East US has been unresponsive since 2AM. Multiple services affected."},
    {"id": "TKT-1038", "subject": "Storage account access denied after key rotation",
     "status": "pending", "priority": "medium", "assignee": "Mike Johnson",
     "created": "2026-05-02", "customer": "Fabrikam Inc",
     "description": "Customer rotated storage keys and now their app can't connect."},
    {"id": "TKT-1035", "subject": "App Service scaling issues during peak hours",
     "status": "open", "priority": "high", "assignee": "Sarah Chen",
     "created": "2026-05-01", "customer": "Contoso Ltd",
     "description": "Web app not scaling beyond 3 instances despite auto-scale rules set to 10."},
    {"id": "TKT-1029", "subject": "SSL certificate expiring on custom domain",
     "status": "resolved", "priority": "medium", "assignee": "Alex Rivera",
     "created": "2026-04-28", "customer": "Northwind Traders",
     "description": "Custom domain SSL cert expires in 5 days. Need renewal or switch to managed cert."},
    {"id": "TKT-1025", "subject": "Azure SQL database performance degradation",
     "status": "open", "priority": "urgent", "assignee": "Mike Johnson",
     "created": "2026-04-27", "customer": "Fabrikam Inc",
     "description": "Query response times increased 10x. DTU usage at 98%."},
]

_STUB_TICKET_DETAILS = {
    "TKT-1042": {
        "thread": [
            {"from": "John Smith (Contoso)", "date": "2026-05-03 02:15",
             "message": "Our production VM 'prod-web-01' in East US is not responding."},
            {"from": "Sarah Chen (Support)", "date": "2026-05-03 02:45",
             "message": "VM is 'Running' but network connectivity tests failing. Checking NSG rules."},
            {"from": "Sarah Chen (Support)", "date": "2026-05-03 03:30",
             "message": "Found the issue — recent NSG rule change blocked inbound traffic on ports 22 and 3389."},
        ]
    },
    "TKT-1025": {
        "thread": [
            {"from": "Lisa Park (Fabrikam)", "date": "2026-04-27 09:00",
             "message": "Azure SQL database 'fabrikam-prod-db' extremely slow. Average query time 50ms → 500ms."},
            {"from": "Mike Johnson (Support)", "date": "2026-04-27 10:15",
             "message": "DTU at 98%. Missing indexes + long-running query from new deployment. Recommending S3 tier."},
        ]
    },
}


class SearchTicketsParams(BaseModel):
    query: str = Field(description="Search keywords (ticket subject, description, or customer name)")
    status: Optional[str] = Field(None, description="Filter by status (open, pending, resolved, closed)")
    priority: Optional[str] = Field(None, description="Filter by priority (low, medium, high, urgent)")


@define_tool(description="Search for support tickets by keyword, status, or priority", skip_permission=True)
async def search_tickets(params: SearchTicketsParams) -> str:
    results = _STUB_TICKETS
    if params.status:
        results = [t for t in results if t["status"] == params.status.lower()]
    if params.query:
        q = params.query.lower()
        results = [t for t in results if q in t["subject"].lower() or q in t["description"].lower()
                    or q in t["customer"].lower() or q in t["id"].lower()]

    if not results:
        return "No tickets found matching the search criteria."

    lines = [f"Found {len(results)} ticket(s):\n"]
    for t in results:
        lines.append(f"- **{t['id']}** [{t['status'].upper()}] (Priority: {t['priority']}) — {t['subject']}")
        lines.append(f"  Customer: {t['customer']} | Assignee: {t['assignee']} | Created: {t['created']}")
    return "\n".join(lines)


class GetTicketDetailsParams(BaseModel):
    ticket_id: str = Field(description="The ticket ID (e.g., TKT-1042)")


@define_tool(description="Get full details and conversation thread for a specific ticket", skip_permission=True)
async def get_ticket_details(params: GetTicketDetailsParams) -> str:
    tid = params.ticket_id.upper()

    # Find the ticket summary
    ticket = next((t for t in _STUB_TICKETS if t["id"] == tid), None)
    if not ticket:
        return f"Ticket {params.ticket_id} not found."

    details = _STUB_TICKET_DETAILS.get(tid, {})
    thread = details.get("thread", [])

    lines = [
        f"# {ticket['id']}: {ticket['subject']}",
        f"**Status:** {ticket['status']} | **Priority:** {ticket['priority']}",
        f"**Customer:** {ticket['customer']} | **Assignee:** {ticket['assignee']}",
        f"**Created:** {ticket['created']}",
        f"\n## Description\n{ticket['description']}",
        "\n## Conversation Thread"
    ]
    for msg in thread:
        lines.append(f"\n**{msg['from']}** ({msg['date']}):\n{msg['message']}")

    return "\n".join(lines)


# ================== Agent Runner ==================

def _get_azure_provider_config() -> dict:
    """Build the custom provider config for Azure OpenAI."""
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

    config = {
        "type": "azure",
        "base_url": endpoint,
        "azure": {"api_version": "2024-10-21"},
    }

    if not api_key:
        client_id = os.environ.get("AZURE_CLIENT_ID")
        credential = (
            ManagedIdentityCredential(client_id=client_id)
            if client_id
            else DefaultAzureCredential()
        )
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        config["bearer_token"] = token.token
    else:
        config["api_key"] = api_key

    return config


async def run_ticket_query(query: str) -> str:
    """
    Run a query against the Freshdesk ticket system.

    If FRESHDESK_MCP_URI is configured, the Copilot CLI will use its built-in
    MCP support to connect. Otherwise, uses stub tools.

    No user token needed — MCP servers run as stdio subprocesses
    authenticated via the container's managed identity.
    """
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    freshdesk_mcp_url = os.environ.get("FRESHDESK_MCP_URI")

    response_parts: list[str] = []
    done = asyncio.Event()

    def on_event(event):
        match event.data:
            case AssistantMessageData() as data:
                response_parts.append(data.content)
            case SessionIdleData():
                done.set()

    try:
        async with CopilotClient() as client:
            session_kwargs = {
                "model": deployment_name,
                "provider": _get_azure_provider_config(),
                "system_message": {"content": TICKET_AGENT_INSTRUCTIONS},
            }

            if freshdesk_mcp_url:
                # MCP mode — CLI handles MCP connection via stdio
                # MCP server configured in CLI's mcp.json
                session_kwargs["on_permission_request"] = PermissionHandler.approve_all
                logger.info(f"Using Freshdesk MCP server (stdio) at {freshdesk_mcp_url}")
            else:
                # Stub mode — register local tools
                if not freshdesk_mcp_url:
                    logger.info("FRESHDESK_MCP_URI not configured — using stub ticket tools")
                session_kwargs["tools"] = [search_tickets, get_ticket_details]
                session_kwargs["on_permission_request"] = PermissionHandler.approve_all

            async with await client.create_session(**session_kwargs) as session:
                session.on(on_event)
                await session.send(query)
                await done.wait()

        return "".join(response_parts) or "No response from agent."

    except Exception as ex:
        logger.error(f"Ticket agent error: {ex}", exc_info=True)
        return f"Error querying ticket system: {ex}"
