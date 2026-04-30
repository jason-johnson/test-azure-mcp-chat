"""
Foundry Agent Service proxy API.

Thin Function App that proxies chat requests to a Foundry Agent Service agent.
Uses the azure-ai-projects v2 SDK with the OpenAI Responses API pattern.
Conversations are managed by Foundry — no local state needed.
"""
import os
import json
import logging

import azure.functions as func
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MCPTool, PromptAgentDefinition

logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Cached agent name — created once, reused across requests
_agent_name: str | None = None


def _get_credential():
    """Get credential based on environment."""
    client_id = os.environ.get("AZURE_CLIENT_ID")
    if client_id:
        return ManagedIdentityCredential(client_id=client_id)
    return DefaultAzureCredential()


def _get_project_client() -> AIProjectClient:
    """Get Foundry project client."""
    return AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        credential=_get_credential(),
    )


def _ensure_agent(project: AIProjectClient) -> str:
    """Create or retrieve the agent. Returns the agent name."""
    global _agent_name
    if _agent_name:
        return _agent_name

    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
    name = "azure-mcp-agent"

    # Build tools list — add MCP server if configured
    tools = []
    mcp_url = os.environ.get("MCP_SERVER_URL")
    if mcp_url:
        tools.append(
            MCPTool(
                server_label="azure-mcp",
                server_url=mcp_url,
                server_description="Azure MCP Server for managing Azure resources",
                require_approval="never",
            )
        )
        logger.info("MCP tool configured: %s", mcp_url)

    agent = project.agents.create_version(
        agent_name=name,
        definition=PromptAgentDefinition(
            model=model,
            instructions=(
                "You are a helpful assistant that can manage Azure resources. "
                "Use the available tools to list, inspect, and manage Azure resources "
                "on behalf of the user. Be concise and format responses in markdown."
            ),
            tools=tools if tools else None,
        ),
    )
    _agent_name = agent.name
    logger.info("Created agent: %s (version: %s)", agent.name, agent.version)
    return _agent_name


# --------------- HTTP Endpoints ---------------


@app.function_name("CreateThread")
@app.route(route="threads", methods=["POST"])
def create_thread(req: func.HttpRequest) -> func.HttpResponse:
    """Create a new conversation."""
    try:
        project = _get_project_client()
        openai = project.get_openai_client()
        conversation = openai.conversations.create()
        return func.HttpResponse(
            json.dumps({"threadId": conversation.id}),
            status_code=201,
            mimetype="application/json",
        )
    except Exception as ex:
        logger.error("Error creating conversation: %s", ex)
        return func.HttpResponse(
            json.dumps({"error": str(ex)}),
            status_code=500,
            mimetype="application/json",
        )


@app.function_name("Chat")
@app.route(route="chat/{threadId?}", methods=["POST"])
def chat(req: func.HttpRequest) -> func.HttpResponse:
    """Send a message and get a response.

    If threadId is provided, continues an existing conversation.
    Otherwise creates a new conversation.
    """
    try:
        body = req.get_json()
        message = body.get("message", "").strip()
        if not message:
            return func.HttpResponse(
                json.dumps({"error": "message is required"}),
                status_code=400,
                mimetype="application/json",
            )

        project = _get_project_client()
        agent_name = _ensure_agent(project)
        openai = project.get_openai_client()

        # Get or create conversation
        conversation_id = req.route_params.get("threadId")
        if not conversation_id:
            conversation = openai.conversations.create()
            conversation_id = conversation.id

        # Send message and get response via the Responses API
        response = openai.responses.create(
            input=message,
            conversation=conversation_id,
            extra_body={
                "agent_reference": {
                    "name": agent_name,
                    "type": "agent_reference",
                }
            },
        )

        response_text = response.output_text or "No response generated."

        return func.HttpResponse(
            json.dumps({
                "threadId": conversation_id,
                "response": response_text,
                "status": "completed",
            }),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as ex:
        logger.error("Error in chat: %s", ex)
        return func.HttpResponse(
            json.dumps({"error": str(ex)}),
            status_code=500,
            mimetype="application/json",
        )
