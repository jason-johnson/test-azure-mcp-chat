"""
Foundry Agent Service proxy API.

Thin Function App that proxies chat requests to a Foundry Agent Service agent.
The agent is created lazily on first request and cached. Threads (conversations)
are managed by Foundry — no local state needed.
"""
import os
import json
import logging

import azure.functions as func
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.ai.projects import AIProjectClient

logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Cached agent ID — created once, reused across requests
_agent_id: str | None = None


def _get_credential():
    """Get credential based on environment."""
    client_id = os.environ.get("AZURE_CLIENT_ID")
    if client_id:
        return ManagedIdentityCredential(client_id=client_id)
    return DefaultAzureCredential()


def _get_client() -> AIProjectClient:
    """Get Foundry project client."""
    return AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        credential=_get_credential(),
    )


def _ensure_agent(client: AIProjectClient) -> str:
    """Create or retrieve the Azure MCP agent."""
    global _agent_id
    if _agent_id:
        return _agent_id

    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")

    agent = client.agents.create_agent(
        model=model,
        name="azure-mcp-agent",
        instructions=(
            "You are a helpful assistant that can manage Azure resources. "
            "Use the available tools to list, inspect, and manage Azure resources "
            "on behalf of the user. Be concise and format responses in markdown."
        ),
    )
    _agent_id = agent.id
    logger.info("Created agent: %s", _agent_id)
    return _agent_id


# --------------- HTTP Endpoints ---------------


@app.function_name("CreateThread")
@app.route(route="threads", methods=["POST"])
def create_thread(req: func.HttpRequest) -> func.HttpResponse:
    """Create a new conversation thread."""
    try:
        client = _get_client()
        thread = client.agents.threads.create()
        return func.HttpResponse(
            json.dumps({"threadId": thread.id}),
            status_code=201,
            mimetype="application/json",
        )
    except Exception as ex:
        logger.error("Error creating thread: %s", ex)
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
    Otherwise creates a new thread.
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

        client = _get_client()
        agent_id = _ensure_agent(client)

        # Get or create thread
        thread_id = req.route_params.get("threadId")
        if not thread_id:
            thread = client.agents.threads.create()
            thread_id = thread.id

        # Add user message
        client.agents.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message,
        )

        # Run the agent
        run = client.agents.threads.runs.create_and_process(
            thread_id=thread_id,
            agent_id=agent_id,
        )

        # Get the assistant's response messages
        messages = client.agents.threads.messages.list(thread_id=thread_id)
        assistant_messages = []
        for msg in messages.data:
            if msg.role == "assistant":
                for content in msg.content:
                    if hasattr(content, "text"):
                        assistant_messages.append(content.text.value)
                break  # Only get the latest assistant message

        response_text = assistant_messages[0] if assistant_messages else "No response generated."

        return func.HttpResponse(
            json.dumps({
                "threadId": thread_id,
                "response": response_text,
                "status": run.status,
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


@app.function_name("GetThread")
@app.route(route="threads/{threadId}", methods=["GET"])
def get_thread(req: func.HttpRequest) -> func.HttpResponse:
    """Get conversation history for a thread."""
    try:
        thread_id = req.route_params.get("threadId")
        if not thread_id:
            return func.HttpResponse(
                json.dumps({"error": "threadId is required"}),
                status_code=400,
                mimetype="application/json",
            )

        client = _get_client()
        messages = client.agents.threads.messages.list(thread_id=thread_id)

        history = []
        for msg in reversed(messages.data):
            content_parts = []
            for content in msg.content:
                if hasattr(content, "text"):
                    content_parts.append(content.text.value)
            if content_parts:
                history.append({
                    "role": msg.role,
                    "content": "\n".join(content_parts),
                })

        return func.HttpResponse(
            json.dumps({"threadId": thread_id, "messages": history}),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as ex:
        logger.error("Error getting thread: %s", ex)
        return func.HttpResponse(
            json.dumps({"error": str(ex)}),
            status_code=500,
            mimetype="application/json",
        )
