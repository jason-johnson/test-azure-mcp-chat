"""
Azure Support Assistant — two-agent system using Microsoft Agent Framework.

Uses AgentFunctionApp as the application host. Two agents are registered:
  - AzureResourceAgent: Queries Azure resources via Azure MCP server (OBO user identity)
  - TicketAgent: Searches support tickets via Freshdesk MCP server (or stub tools)

Agents run inside durable function activities to support async MCP connections
with per-request user tokens. The registered agents provide the framework with
metadata for health checks and discovery, while actual MCP-connected execution
happens in the activity functions.
"""
import os
import logging
import json

import azure.functions as func
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import AgentFunctionApp
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

from azure_agent import run_azure_query, AZURE_AGENT_INSTRUCTIONS
from ticket_agent import run_ticket_query, TICKET_AGENT_INSTRUCTIONS

logger = logging.getLogger(__name__)


def _get_credential():
    """Get credential — ManagedIdentity when deployed, DefaultAzureCredential locally."""
    client_id = os.environ.get("AZURE_CLIENT_ID")
    if client_id:
        return ManagedIdentityCredential(client_id=client_id)
    return DefaultAzureCredential()


# ================== Agent Registration ==================

_client = OpenAIChatClient(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
    credential=_get_credential(),
)

_azure_agent = _client.as_agent(
    name="AzureResourceAgent",
    instructions=AZURE_AGENT_INSTRUCTIONS,
)

_ticket_agent = _client.as_agent(
    name="TicketAgent",
    instructions=TICKET_AGENT_INSTRUCTIONS,
)

app = AgentFunctionApp(
    agents=[_azure_agent, _ticket_agent],
    http_auth_level=func.AuthLevel.ANONYMOUS,
    enable_http_endpoints=False,
)


# ================== Activity Functions ==================

@app.activity_trigger(input_name="request")
async def query_azure_resources(request: dict) -> dict:
    """Activity: run a query against the Azure MCP server."""
    try:
        query = request.get("query", "")
        user_token = request.get("user_access_token", "")

        if not user_token:
            return {"error": "No user access token provided", "result": ""}

        result = await run_azure_query(query, user_token)
        return {"result": result}
    except Exception as ex:
        logger.error(f"Azure activity error: {ex}", exc_info=True)
        return {"error": str(ex), "result": ""}


@app.activity_trigger(input_name="request")
async def search_tickets_activity(request: dict) -> dict:
    """Activity: run a query against the Freshdesk ticket system."""
    try:
        query = request.get("query", "")
        user_token = request.get("user_access_token", "")

        result = await run_ticket_query(query, user_token or None)
        return {"result": result}
    except Exception as ex:
        logger.error(f"Ticket activity error: {ex}", exc_info=True)
        return {"error": str(ex), "result": ""}


# ================== Orchestrations ==================

@app.orchestration_trigger(context_name="context")
def azure_query_orchestration(context):
    """Orchestration: run an Azure resource query."""
    request = context.get_input()
    result = yield context.call_activity("query_azure_resources", request)
    return result


@app.orchestration_trigger(context_name="context")
def ticket_query_orchestration(context):
    """Orchestration: run a ticket search query."""
    request = context.get_input()
    result = yield context.call_activity("search_tickets_activity", request)
    return result


@app.orchestration_trigger(context_name="context")
def combined_query_orchestration(context):
    """Orchestration: run both agents in parallel and combine results."""
    request = context.get_input()

    azure_task = context.call_activity("query_azure_resources", request)
    ticket_task = context.call_activity("search_tickets_activity", request)

    results = yield context.task_all([azure_task, ticket_task])

    return {
        "azure": results[0],
        "tickets": results[1],
    }


# ================== HTTP Endpoints ==================


@app.function_name(name="AzureQuery")
@app.route(route="query/azure", methods=["POST"])
@app.durable_client_input(client_name="client")
async def azure_query(req: func.HttpRequest, client) -> func.HttpResponse:
    """Start an Azure resource query.

    Body: { "query": "...", "userAccessToken": "..." }
    """
    try:
        body = req.get_json()
        query = body.get("query", "")
        token = body.get("userAccessToken", "")

        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'query'"}), status_code=400, mimetype="application/json"
            )
        if not token:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'userAccessToken'"}), status_code=401, mimetype="application/json"
            )

        instance_id = await client.start_new(
            "azure_query_orchestration",
            client_input={"query": query, "user_access_token": token},
        )
        return func.HttpResponse(
            json.dumps({"id": instance_id}), status_code=202, mimetype="application/json"
        )
    except Exception as ex:
        logger.error(f"Error starting Azure query: {ex}")
        return func.HttpResponse(
            json.dumps({"error": str(ex)}), status_code=500, mimetype="application/json"
        )


@app.function_name(name="TicketQuery")
@app.route(route="query/tickets", methods=["POST"])
@app.durable_client_input(client_name="client")
async def ticket_query(req: func.HttpRequest, client) -> func.HttpResponse:
    """Start a ticket search query.

    Body: { "query": "...", "userAccessToken": "..." }
    """
    try:
        body = req.get_json()
        query = body.get("query", "")
        token = body.get("userAccessToken", "")

        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'query'"}), status_code=400, mimetype="application/json"
            )

        instance_id = await client.start_new(
            "ticket_query_orchestration",
            client_input={"query": query, "user_access_token": token},
        )
        return func.HttpResponse(
            json.dumps({"id": instance_id}), status_code=202, mimetype="application/json"
        )
    except Exception as ex:
        logger.error(f"Error starting ticket query: {ex}")
        return func.HttpResponse(
            json.dumps({"error": str(ex)}), status_code=500, mimetype="application/json"
        )


@app.function_name(name="CombinedQuery")
@app.route(route="query/both", methods=["POST"])
@app.durable_client_input(client_name="client")
async def combined_query(req: func.HttpRequest, client) -> func.HttpResponse:
    """Start a combined query that runs both agents in parallel.

    Body: { "query": "...", "userAccessToken": "..." }
    """
    try:
        body = req.get_json()
        query = body.get("query", "")
        token = body.get("userAccessToken", "")

        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'query'"}), status_code=400, mimetype="application/json"
            )

        instance_id = await client.start_new(
            "combined_query_orchestration",
            client_input={"query": query, "user_access_token": token},
        )
        return func.HttpResponse(
            json.dumps({"id": instance_id}), status_code=202, mimetype="application/json"
        )
    except Exception as ex:
        logger.error(f"Error starting combined query: {ex}")
        return func.HttpResponse(
            json.dumps({"error": str(ex)}), status_code=500, mimetype="application/json"
        )


@app.function_name(name="QueryStatus")
@app.route(route="query/status/{instance_id}", methods=["GET"])
@app.durable_client_input(client_name="client")
async def query_status(req: func.HttpRequest, client) -> func.HttpResponse:
    """Get the status of any query orchestration."""
    try:
        instance_id = req.route_params.get("instance_id")
        status = await client.get_status(instance_id)

        if not status:
            return func.HttpResponse(
                json.dumps({"error": "Query not found"}), status_code=404, mimetype="application/json"
            )

        return func.HttpResponse(
            json.dumps({
                "id": status.instance_id,
                "runtimeStatus": status.runtime_status.name,
                "output": status.output,
            }),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as ex:
        logger.error(f"Error checking status: {ex}")
        return func.HttpResponse(
            json.dumps({"error": str(ex)}), status_code=500, mimetype="application/json"
        )
