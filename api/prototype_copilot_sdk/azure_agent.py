"""
Azure Resource Agent — rewritten for github-copilot-sdk.

Uses CopilotClient with Azure OpenAI as a custom provider.
Azure MCP server runs as a LOCAL STDIO SUBPROCESS managed by the Copilot CLI —
no separate Container App, no OBO flow, no app registrations for MCP.

Dual auth mode for ARM access:
  - Web UI (user logged in):  user's ARM token passed via AZURE_ACCESS_TOKEN env var
  - Webhook (no user):        container's managed identity via DefaultAzureCredential

Architecture (before → after):
  BEFORE: React → Function App → MCPStreamableHTTPTool → MCP Container App (OBO) → ARM
  AFTER:  React → Container App → CopilotClient → CLI → azure-mcp (stdio) → ARM

What this eliminates:
  - MCP Container App (mcp.bicep)
  - MCP Container Apps Environment
  - MCP Server App Registration + Service Principal
  - MCP Client App Registration + Service Principal
  - Federated Identity Credential for OBO
  - User-assigned Managed Identity for OBO
  - Custom MCP audience scopes (api://.../Mcp.Tools.ReadWrite)

Frontend auth simplification:
  - MSAL scope changes from api://{mcp}/Mcp.Tools.ReadWrite
    to https://management.azure.com/user_impersonation (ARM direct)
"""
import os
import asyncio
import logging
from typing import Optional

from copilot import CopilotClient, SubprocessConfig
from copilot.generated.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    SessionIdleData,
)
from copilot.session import PermissionHandler, PermissionRequestResult
from copilot.generated.session_events import PermissionRequest

logger = logging.getLogger(__name__)

# Path to the azure-mcp binary in the container image
AZURE_MCP_PATH = os.getenv("AZURE_MCP_PATH", "/usr/local/bin/azure-mcp")

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


def _get_azure_provider_config() -> dict:
    """Build the custom provider config for Azure OpenAI."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

    config = {
        "type": "azure",
        "base_url": endpoint,
        "azure": {
            "api_version": "2024-10-21",
        },
    }

    if not api_key:
        from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

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


def _mcp_permission_handler(
    request: PermissionRequest, invocation: dict
) -> PermissionRequestResult:
    """Approve MCP and read operations; deny shell/write for safety."""
    if request.kind.value in ("mcp", "custom-tool", "read"):
        return PermissionRequestResult(kind="approved")
    return PermissionRequestResult(kind="denied-by-rules")


def _get_copilot_config(user_access_token: Optional[str] = None) -> SubprocessConfig:
    """
    Build CopilotClient config with dual auth support.

    The Copilot CLI spawns azure-mcp as a stdio subprocess. The env vars
    passed here control how azure-mcp authenticates to ARM:

    1. If user_access_token is provided (web UI → user logged in):
       - AZURE_ACCESS_TOKEN is set → azure-mcp uses it directly for ARM calls
       - ARM calls run as the user → per-user RBAC enforced

    2. If user_access_token is None (webhook → no user context):
       - AZURE_ACCESS_TOKEN is not set → azure-mcp falls back to
         DefaultAzureCredential → picks up AZURE_CLIENT_ID → managed identity
       - ARM calls run as the MI → MI's RBAC applies

    The MCP server config in the CLI's config file (mcp.json):
    {
      "mcpServers": {
        "azure": {
          "command": "/opt/azure-mcp/azure-mcp",
          "args": ["--mode", "all", "--read-only"]
        }
      }
    }
    """
    env = {
        # Always pass MI client ID so DefaultAzureCredential can find it
        "AZURE_CLIENT_ID": os.getenv("AZURE_CLIENT_ID", ""),
    }

    if user_access_token:
        # User is logged in — azure-mcp will use this token for ARM calls
        # instead of DefaultAzureCredential. Per-user RBAC enforced.
        env["AZURE_ACCESS_TOKEN"] = user_access_token
        logger.info("MCP auth: using user's ARM token (per-user RBAC)")
    else:
        # No user context (webhook) — fall back to managed identity
        logger.info("MCP auth: using managed identity (service RBAC)")

    return SubprocessConfig(env=env)


async def run_azure_query(query: str, user_access_token: Optional[str] = None) -> str:
    """
    Run a query against Azure resources via the local MCP server.

    Args:
        query: Natural language query about Azure resources.
        user_access_token: Optional ARM-scoped OAuth token from the logged-in user.
            If provided, azure-mcp uses it for per-user RBAC.
            If None (webhook trigger), falls back to managed identity.

    Returns:
        The agent's response as a string.
    """
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

    response_parts: list[str] = []
    done = asyncio.Event()

    def on_event(event):
        match event.data:
            case AssistantMessageData() as data:
                response_parts.append(data.content)
            case SessionIdleData():
                done.set()

    try:
        async with CopilotClient(_get_copilot_config(user_access_token)) as client:
            async with await client.create_session(
                model=deployment_name,
                provider=_get_azure_provider_config(),
                system_message={"content": AZURE_AGENT_INSTRUCTIONS},
                on_permission_request=_mcp_permission_handler,
                # MCP servers configured in CLI config — azure-mcp runs
                # as a stdio subprocess. Auth depends on caller:
                #   - Web UI: user's ARM token via AZURE_ACCESS_TOKEN
                #   - Webhook: managed identity via AZURE_CLIENT_ID
            ) as session:
                session.on(on_event)
                await session.send(query)
                await done.wait()

        return "".join(response_parts) or "No response from agent."

    except Exception as ex:
        logger.error(f"Azure agent error: {ex}", exc_info=True)
        return f"Error querying Azure resources: {ex}"


async def run_azure_query_streaming(query: str, user_access_token: Optional[str] = None):
    """
    Generator version that yields streaming chunks for SSE responses.

    Args:
        query: Natural language query about Azure resources.
        user_access_token: Optional ARM token. User's token if web UI, None if webhook.
    """
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

    done = asyncio.Event()
    chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_event(event):
        match event.data:
            case AssistantMessageDeltaData() as data:
                if data.delta_content:
                    chunk_queue.put_nowait(data.delta_content)
            case SessionIdleData():
                chunk_queue.put_nowait(None)  # sentinel
                done.set()

    try:
        async with CopilotClient(_get_copilot_config(user_access_token)) as client:
            async with await client.create_session(
                model=deployment_name,
                provider=_get_azure_provider_config(),
                system_message={"content": AZURE_AGENT_INSTRUCTIONS},
                on_permission_request=_mcp_permission_handler,
                streaming=True,
            ) as session:
                session.on(on_event)
                await session.send(query)

                while True:
                    chunk = await chunk_queue.get()
                    if chunk is None:
                        break
                    yield chunk

    except Exception as ex:
        logger.error(f"Azure agent streaming error: {ex}", exc_info=True)
        yield f"Error querying Azure resources: {ex}"
