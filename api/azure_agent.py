"""
Azure Resource Agent — queries Azure resources via the Azure MCP server.

Uses MCPStreamableHTTPTool to connect to the deployed Azure MCP container app.
The user's OAuth access token (audience: api://{MCP_SERVER_CLIENT_ID}/Mcp.Tools.ReadWrite)
is forwarded as a Bearer token so the MCP server can perform OBO to call ARM
with the user's identity and RBAC.
"""
import os
import logging

import httpx
from agent_framework import MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

logger = logging.getLogger(__name__)

AZURE_AGENT_INSTRUCTIONS = """You are an Azure infrastructure assistant for a support team.
You have access to Azure MCP tools that can query and inspect the user's Azure resources.

Guidelines:
- Use the available tools to look up real data. Do NOT guess or hallucinate resource names.
- Be concise and factual. Format results as clear lists or tables.
- Include resource names, types, resource groups, locations, and key properties.
- If a query is ambiguous, ask the user to clarify (e.g., which subscription).
- If you encounter permission errors, explain that the user may lack RBAC access.
- When listing resources, summarize counts and highlight anything unusual (stopped VMs, unhealthy resources, etc.).
- Never modify or delete resources unless the user explicitly asks and confirms."""


def _get_credential():
    client_id = os.environ.get("AZURE_CLIENT_ID")
    if client_id:
        return ManagedIdentityCredential(client_id=client_id)
    return DefaultAzureCredential()


async def run_azure_query(query: str, user_access_token: str) -> str:
    """
    Run a query against the Azure MCP server using the user's token.

    Args:
        query: Natural language query about Azure resources.
        user_access_token: OAuth 2.0 access token for the MCP server audience.

    Returns:
        The agent's response as a string.
    """
    mcp_url = os.environ.get("MCP_SERVER_URI")
    if not mcp_url:
        return "Error: MCP_SERVER_URI is not configured. The Azure MCP server URL must be set."

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

    http_client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {user_access_token}"},
        timeout=120.0,
    )

    mcp_tool = MCPStreamableHTTPTool(
        name="azure-mcp",
        url=mcp_url,
        description="Query and inspect Azure resources on behalf of the authenticated user.",
        http_client=http_client,
        load_prompts=False,
    )

    client = OpenAIChatClient(
        azure_endpoint=endpoint,
        model=deployment_name,
        credential=_get_credential(),
    )

    agent = client.as_agent(
        name="AzureResourceAgent",
        instructions=AZURE_AGENT_INSTRUCTIONS,
        tools=mcp_tool,
    )

    try:
        async with mcp_tool:
            response = await agent.run(query)
            return response.text if hasattr(response, "text") else str(response)
    except Exception as ex:
        logger.error(f"Azure agent error: {ex}", exc_info=True)
        return f"Error querying Azure resources: {ex}"
    finally:
        await http_client.aclose()
