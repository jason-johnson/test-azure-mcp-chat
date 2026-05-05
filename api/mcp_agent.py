"""
MCP Agent activity — runs Azure MCP queries on behalf of the authenticated user.

This runs as a Durable Functions activity because:
1. Durable orchestrations must be deterministic (no I/O)
2. MCPStreamableHTTPTool needs async HTTP connections
3. The user's access token is passed in and forwarded to the MCP server
"""
import os
import logging

import httpx
from agent_framework import MCPStreamableHTTPTool
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

logger = logging.getLogger(__name__)


def _get_credential():
    client_id = os.environ.get("AZURE_CLIENT_ID")
    if client_id:
        return ManagedIdentityCredential(client_id=client_id)
    return DefaultAzureCredential()


async def run_mcp_query(query: str, user_access_token: str) -> str:
    """
    Run a query against the Azure MCP server using the user's token.

    Args:
        query: The natural language query about Azure resources.
        user_access_token: OAuth 2.0 access token with audience api://{MCP_SERVER_CLIENT_ID}/Mcp.Tools.ReadWrite.

    Returns:
        The agent's response as a string.
    """
    mcp_url = os.environ.get("MCP_SERVER_URI")
    if not mcp_url:
        return "Error: MCP_SERVER_URI environment variable is not configured."

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

    # Create an httpx client that injects the user's bearer token
    http_client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {user_access_token}"},
        timeout=60.0,
    )

    mcp_tool = MCPStreamableHTTPTool(
        name="azure-mcp",
        url=mcp_url,
        description="Query and manage Azure resources on behalf of the user.",
        http_client=http_client,
    )

    client = AzureOpenAIChatClient(
        endpoint=endpoint,
        deployment_name=deployment_name,
        credential=_get_credential(),
    )

    agent = client.as_agent(
        name="AzureResourceAgent",
        instructions="""You are an Azure resource assistant for a support team.
Use the Azure MCP tools to answer questions about the user's Azure resources.
Be concise and factual. Format results clearly with resource names, types, and key properties.
If you encounter permission errors, explain that the user may not have access to that resource or subscription.""",
        tools=mcp_tool,
    )

    try:
        async with mcp_tool:
            response = await agent.run(query)
            return response.text if hasattr(response, "text") else str(response)
    except Exception as ex:
        logger.error(f"MCP agent error: {ex}", exc_info=True)
        return f"Error querying Azure resources: {ex}"
    finally:
        await http_client.aclose()
