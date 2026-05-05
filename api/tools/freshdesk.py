"""
Freshdesk ticket search tool — stub implementation.

Returns mock ticket data for now. Replace the search_tickets function body
with real Freshdesk API calls when ready.

Freshdesk API docs: https://developers.freshdesk.com/api/
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Mock ticket data for development
_MOCK_TICKETS = [
    {
        "id": 10234,
        "subject": "Web App failing health checks after deployment",
        "description": "Customer reports their App Service web app started returning 503 errors after a deployment. Health check endpoint /health is timing out.",
        "status": "Open",
        "priority": "High",
        "created_at": "2026-05-01T14:23:00Z",
        "updated_at": "2026-05-03T09:15:00Z",
        "requester": "alice@contoso.com",
        "tags": ["app-service", "health-check", "503-error", "deployment"],
        "resolution": None,
    },
    {
        "id": 10189,
        "subject": "Storage account access denied after key rotation",
        "description": "After rotating storage account keys, several function apps lost connectivity to blob storage. Managed identity was not configured.",
        "status": "Resolved",
        "priority": "High",
        "created_at": "2026-04-28T10:05:00Z",
        "updated_at": "2026-04-29T16:30:00Z",
        "requester": "bob@contoso.com",
        "tags": ["storage-account", "access-denied", "key-rotation", "managed-identity"],
        "resolution": "Configured managed identity for the function apps and assigned Storage Blob Data Contributor role.",
    },
    {
        "id": 10301,
        "subject": "High CPU on Linux App Service Plan",
        "description": "App Service Plan showing sustained 95%+ CPU usage. Multiple web apps on the same plan are affected.",
        "status": "Open",
        "priority": "Medium",
        "created_at": "2026-05-04T08:45:00Z",
        "updated_at": "2026-05-04T11:20:00Z",
        "requester": "carol@contoso.com",
        "tags": ["app-service-plan", "high-cpu", "performance", "linux"],
        "resolution": None,
    },
    {
        "id": 10156,
        "subject": "Cannot connect to Azure SQL from Virtual Network",
        "description": "Application in a VNet-integrated App Service cannot reach the Azure SQL database. Private endpoint DNS resolution appears to be failing.",
        "status": "Resolved",
        "priority": "High",
        "created_at": "2026-04-25T13:10:00Z",
        "updated_at": "2026-04-26T17:45:00Z",
        "requester": "dave@contoso.com",
        "tags": ["azure-sql", "vnet", "private-endpoint", "dns", "connectivity"],
        "resolution": "Added private DNS zone link for privatelink.database.windows.net to the VNet.",
    },
    {
        "id": 10287,
        "subject": "Function App cold start taking 30+ seconds",
        "description": "Python Function App on Consumption plan experiencing very long cold starts. First request after idle period takes over 30 seconds.",
        "status": "Open",
        "priority": "Medium",
        "created_at": "2026-05-03T07:30:00Z",
        "updated_at": "2026-05-03T15:00:00Z",
        "requester": "eve@contoso.com",
        "tags": ["function-app", "cold-start", "performance", "python", "consumption-plan"],
        "resolution": None,
    },
]


def search_tickets(query: str) -> str:
    """
    Search Freshdesk tickets related to a query.

    Args:
        query: Natural language search query describing the issue or topic.

    Returns:
        A formatted string with matching ticket details.

    TODO: Replace this stub with real Freshdesk API calls:
        GET https://{domain}.freshdesk.com/api/v2/search/tickets?query="..."
        Authorization: Basic base64(api_key:X)
    """
    logger.info(f"Searching tickets for: {query}")

    query_lower = query.lower()

    # Simple keyword matching against mock data
    matches = []
    for ticket in _MOCK_TICKETS:
        searchable = " ".join(
            [
                ticket["subject"],
                ticket["description"],
                " ".join(ticket["tags"]),
                ticket.get("resolution") or "",
            ]
        ).lower()

        # Count keyword hits
        keywords = query_lower.split()
        hits = sum(1 for kw in keywords if kw in searchable)
        if hits > 0:
            matches.append((hits, ticket))

    matches.sort(key=lambda x: x[0], reverse=True)

    if not matches:
        return "No matching support tickets found for this query."

    results = []
    for _, ticket in matches[:5]:
        status_icon = "✅" if ticket["status"] == "Resolved" else "🔴"
        result = (
            f"{status_icon} **Ticket #{ticket['id']}** [{ticket['status']}] "
            f"(Priority: {ticket['priority']})\n"
            f"  **Subject:** {ticket['subject']}\n"
            f"  **Description:** {ticket['description']}\n"
            f"  **Requester:** {ticket['requester']}\n"
            f"  **Tags:** {', '.join(ticket['tags'])}\n"
            f"  **Created:** {ticket['created_at']}"
        )
        if ticket["resolution"]:
            result += f"\n  **Resolution:** {ticket['resolution']}"
        results.append(result)

    return f"Found {len(matches)} matching ticket(s):\n\n" + "\n\n".join(results)
