"""
Ticket Agent — searches and retrieves support tickets via a Freshdesk MCP server.

Uses MCPStreamableHTTPTool to connect to a Freshdesk MCP server.
When FRESHDESK_MCP_URI is not configured, falls back to stub tools
that return mock data for development and testing.
"""
import os
import logging
from typing import Optional

import httpx
from agent_framework import MCPStreamableHTTPTool, FunctionTool, tool
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

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


# ================== Stub Tools (used when Freshdesk MCP is not configured) ==================

@tool
def search_tickets(query: str, status: Optional[str] = None, priority: Optional[str] = None) -> str:
    """Search for support tickets by keyword, status, or priority.

    Args:
        query: Search keywords (ticket subject, description, or customer name).
        status: Filter by status (open, pending, resolved, closed). Optional.
        priority: Filter by priority (low, medium, high, urgent). Optional.
    """
    # Stub data for development
    tickets = [
        {"id": "TKT-1042", "subject": "VM not responding in East US region",
         "status": "open", "priority": "high", "assignee": "Sarah Chen",
         "created": "2026-05-03", "customer": "Contoso Ltd",
         "description": "Production VM in East US has been unresponsive since 2AM. Multiple services affected."},
        {"id": "TKT-1038", "subject": "Storage account access denied after key rotation",
         "status": "pending", "priority": "medium", "assignee": "Mike Johnson",
         "created": "2026-05-02", "customer": "Fabrikam Inc",
         "description": "Customer rotated storage keys and now their app can't connect. Need to update connection strings."},
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
         "description": "Query response times increased 10x. DTU usage at 98%. Need to investigate and scale."},
    ]

    results = tickets
    if status:
        results = [t for t in results if t["status"] == status.lower()]
    if query:
        q = query.lower()
        results = [t for t in results if q in t["subject"].lower() or q in t["description"].lower()
                    or q in t["customer"].lower() or q in t["id"].lower()]

    if not results:
        return "No tickets found matching the search criteria."

    lines = [f"Found {len(results)} ticket(s):\n"]
    for t in results:
        lines.append(f"- **{t['id']}** [{t['status'].upper()}] (Priority: {t['priority']}) — {t['subject']}")
        lines.append(f"  Customer: {t['customer']} | Assignee: {t['assignee']} | Created: {t['created']}")
    return "\n".join(lines)


@tool
def get_ticket_details(ticket_id: str) -> str:
    """Get full details and conversation thread for a specific ticket.

    Args:
        ticket_id: The ticket ID (e.g., TKT-1042).
    """
    tickets = {
        "TKT-1042": {
            "id": "TKT-1042", "subject": "VM not responding in East US region",
            "status": "open", "priority": "high", "assignee": "Sarah Chen",
            "created": "2026-05-03", "customer": "Contoso Ltd",
            "description": "Production VM in East US has been unresponsive since 2AM. Multiple services affected.",
            "thread": [
                {"from": "John Smith (Contoso)", "date": "2026-05-03 02:15",
                 "message": "Our production VM 'prod-web-01' in East US is not responding. We can't SSH or RDP into it. Multiple downstream services are affected."},
                {"from": "Sarah Chen (Support)", "date": "2026-05-03 02:45",
                 "message": "I can see the VM is in a 'Running' state in the portal but network connectivity tests are failing. Checking NSG rules and VM diagnostics."},
                {"from": "Sarah Chen (Support)", "date": "2026-05-03 03:30",
                 "message": "Found the issue — a recent NSG rule change blocked inbound traffic on ports 22 and 3389. Working with the customer to verify the intended rules."},
            ]
        },
        "TKT-1025": {
            "id": "TKT-1025", "subject": "Azure SQL database performance degradation",
            "status": "open", "priority": "urgent", "assignee": "Mike Johnson",
            "created": "2026-04-27", "customer": "Fabrikam Inc",
            "description": "Query response times increased 10x. DTU usage at 98%. Need to investigate and scale.",
            "thread": [
                {"from": "Lisa Park (Fabrikam)", "date": "2026-04-27 09:00",
                 "message": "Our Azure SQL database 'fabrikam-prod-db' has become extremely slow. Average query time went from 50ms to 500ms overnight."},
                {"from": "Mike Johnson (Support)", "date": "2026-04-27 10:15",
                 "message": "DTU usage is at 98%. Identified several missing indexes and a long-running query from a new deployment. Recommending scaling to S3 tier and adding indexes."},
            ]
        }
    }

    tid = ticket_id.upper()
    if tid not in tickets:
        return f"Ticket {ticket_id} not found."

    t = tickets[tid]
    lines = [
        f"# {t['id']}: {t['subject']}",
        f"**Status:** {t['status']} | **Priority:** {t['priority']}",
        f"**Customer:** {t['customer']} | **Assignee:** {t['assignee']}",
        f"**Created:** {t['created']}",
        f"\n## Description\n{t['description']}",
        "\n## Conversation Thread"
    ]
    for msg in t.get("thread", []):
        lines.append(f"\n**{msg['from']}** ({msg['date']}):\n{msg['message']}")

    return "\n".join(lines)


# ================== Agent Runner ==================

def _get_credential():
    client_id = os.environ.get("AZURE_CLIENT_ID")
    if client_id:
        return ManagedIdentityCredential(client_id=client_id)
    return DefaultAzureCredential()


async def run_ticket_query(query: str, user_access_token: Optional[str] = None) -> str:
    """
    Run a query against the Freshdesk ticket system.

    If FRESHDESK_MCP_URI is configured, connects via MCP with the user's token.
    Otherwise, uses stub tools for development.

    Args:
        query: Natural language query about support tickets.
        user_access_token: OAuth 2.0 access token (used when connecting to real MCP server).

    Returns:
        The agent's response as a string.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    freshdesk_mcp_url = os.environ.get("FRESHDESK_MCP_URI")

    client = AzureOpenAIChatClient(
        endpoint=endpoint,
        deployment_name=deployment_name,
        credential=_get_credential(),
    )

    # If Freshdesk MCP server is configured, use it; otherwise use stubs
    if freshdesk_mcp_url and user_access_token:
        http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {user_access_token}"},
            timeout=120.0,
        )
        mcp_tool = MCPStreamableHTTPTool(
            name="freshdesk-mcp",
            url=freshdesk_mcp_url,
            description="Search and retrieve support tickets from Freshdesk.",
            http_client=http_client,
        )
        agent = client.as_agent(
            name="TicketAgent",
            instructions=TICKET_AGENT_INSTRUCTIONS,
            tools=mcp_tool,
        )
        try:
            async with mcp_tool:
                response = await agent.run(query)
                return response.text if hasattr(response, "text") else str(response)
        except Exception as ex:
            logger.error(f"Ticket agent MCP error: {ex}", exc_info=True)
            return f"Error querying ticket system: {ex}"
        finally:
            await http_client.aclose()
    else:
        # Use stub tools
        if not freshdesk_mcp_url:
            logger.info("FRESHDESK_MCP_URI not configured — using stub ticket tools")

        agent = client.as_agent(
            name="TicketAgent",
            instructions=TICKET_AGENT_INSTRUCTIONS,
            tools=[search_tickets, get_ticket_details],
        )
        try:
            response = await agent.run(query)
            return response.text if hasattr(response, "text") else str(response)
        except Exception as ex:
            logger.error(f"Ticket agent error: {ex}", exc_info=True)
            return f"Error querying ticket system: {ex}"
