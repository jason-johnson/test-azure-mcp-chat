"""
Azure Resource Agent — rewritten for github-copilot-sdk.

Azure MCP server runs as a LOCAL STDIO SUBPROCESS configured via mcp_servers
on create_session() — the Copilot CLI manages the subprocess lifecycle.

Dual auth mode for ARM access:
  - Web UI (user logged in):  user's ARM token passed via AZURE_ACCESS_TOKEN env var
  - Webhook (no user):        container's managed identity via DefaultAzureCredential
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
    SessionErrorData,
)
from copilot.session import PermissionHandler

logger = logging.getLogger(__name__)

# Command to launch the azure-mcp server (npx or binary path)
AZURE_MCP_COMMAND = os.getenv("AZURE_MCP_COMMAND", "npx")
AZURE_MCP_ARGS = os.getenv("AZURE_MCP_ARGS", "-y @azure/mcp server start --mode all --read-only").split()

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


def _get_mcp_server_config(user_access_token: Optional[str] = None) -> dict:
    """
    Build the azure MCP server config for create_session(mcp_servers=...).

    The env vars control how azure-mcp authenticates to ARM:
    1. If user_access_token provided: AZURE_ACCESS_TOKEN set → per-user RBAC
    2. If None (webhook): falls back to DefaultAzureCredential → managed identity
    """
    env = {}

    if os.getenv("AZURE_CLIENT_ID"):
        env["AZURE_CLIENT_ID"] = os.environ["AZURE_CLIENT_ID"]

    if user_access_token:
        env["AZURE_ACCESS_TOKEN"] = user_access_token
        logger.info("MCP auth: using user's ARM token (per-user RBAC)")
    else:
        logger.info("MCP auth: using managed identity / DefaultAzureCredential")

    config = {
        "type": "local",
        "command": AZURE_MCP_COMMAND,
        "args": AZURE_MCP_ARGS,
        "tools": ["*"],
    }
    if env:
        config["env"] = env

    return config


def _get_copilot_config() -> SubprocessConfig:
    """Build CopilotClient subprocess config."""
    kwargs = {}
    gh_token = os.environ.get("GITHUB_TOKEN")
    if gh_token:
        kwargs["github_token"] = gh_token
    return SubprocessConfig(**kwargs)


async def run_azure_query(query: str, user_access_token: Optional[str] = None) -> str:
    """
    Run a query against Azure resources via the local MCP server.

    Args:
        query: Natural language query about Azure resources.
        user_access_token: Optional ARM-scoped OAuth token from the logged-in user.

    Returns:
        The agent's response as a string.
    """
    model = os.getenv("COPILOT_MODEL", "gpt-5-mini")

    response_parts: list[str] = []
    done = asyncio.Event()

    def on_event(event):
        match event.data:
            case SessionErrorData() as data:
                logger.error(f"Session error: {data.error_type}: {data.message}")
                response_parts.append(f"Error: {data.message}")
                done.set()
            case AssistantMessageData() as data:
                response_parts.append(data.content)
            case SessionIdleData():
                done.set()

    try:
        session_kwargs = {
            "model": model,
            "system_message": {"content": AZURE_AGENT_INSTRUCTIONS},
            "on_permission_request": PermissionHandler.approve_all,
            "mcp_servers": {
                "azure": _get_mcp_server_config(user_access_token),
            },
        }

        # Only use Azure OpenAI provider if explicitly configured
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            session_kwargs["provider"] = _get_azure_provider_config()

        async with CopilotClient(_get_copilot_config()) as client:
            async with await client.create_session(**session_kwargs) as session:
                session.on(on_event)
                await session.send(query)
                await asyncio.wait_for(done.wait(), timeout=120)

        return "".join(response_parts) or "No response from agent."

    except Exception as ex:
        logger.error(f"Azure agent error: {ex}", exc_info=True)
        return f"Error querying Azure resources: {ex}"


async def run_azure_query_streaming(query: str, user_access_token: Optional[str] = None):
    """
    Generator version that yields streaming chunks for SSE responses.
    """
    model = os.getenv("COPILOT_MODEL", "gpt-5-mini")

    done = asyncio.Event()
    chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_event(event):
        match event.data:
            case SessionErrorData() as data:
                logger.error(f"Session error: {data.error_type}: {data.message}")
                chunk_queue.put_nowait(f"Error: {data.message}")
                chunk_queue.put_nowait(None)
                done.set()
            case AssistantMessageDeltaData() as data:
                if data.delta_content:
                    chunk_queue.put_nowait(data.delta_content)
            case SessionIdleData():
                chunk_queue.put_nowait(None)  # sentinel
                done.set()

    try:
        session_kwargs = {
            "model": model,
            "system_message": {"content": AZURE_AGENT_INSTRUCTIONS},
            "on_permission_request": PermissionHandler.approve_all,
            "streaming": True,
            "mcp_servers": {
                "azure": _get_mcp_server_config(user_access_token),
            },
        }

        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            session_kwargs["provider"] = _get_azure_provider_config()

        async with CopilotClient(_get_copilot_config()) as client:
            async with await client.create_session(**session_kwargs) as session:
                session.on(on_event)
                await session.send(query)

                while True:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=120)
                    if chunk is None:
                        break
                    yield chunk

    except Exception as ex:
        logger.error(f"Azure agent streaming error: {ex}", exc_info=True)
        yield f"Error: {ex}"


def _get_azure_provider_config() -> dict:
    """Build the custom provider config for Azure OpenAI (only used if AZURE_OPENAI_ENDPOINT is set)."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

    config = {
        "type": "azure",
        "base_url": endpoint,
        "azure": {"api_version": "2024-10-21"},
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
