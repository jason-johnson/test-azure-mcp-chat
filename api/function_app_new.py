"""
Azure Support Assistant — routes user queries to Azure MCP and/or Freshdesk agents.

Architecture:
- A single router agent decides whether a query needs Azure resource info,
  ticket search, or both.
- Azure resources are queried via MCPStreamableHTTPTool (user's OBO token).
- Ticket search uses a Freshdesk stub tool (to be replaced with real API).
- The orchestration runs as a durable function so it can do async I/O
  in activity functions.
"""
import os
import logging

import azure.functions as func
import azure.durable_functions as df
import httpx
import json

from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from agent_framework import MCPStreamableHTTPTool
from agent_framework.azure import AzureOpenAIChatClient

from tools.freshdesk import search_tickets

logger = logging.getLogger(__name__)

# ================== Helpers ==================


def _get_credential():
    """Get credential — ManagedIdentity when deployed, DefaultAzureCredential locally."""
    client_id = os.environ.get("AZURE_CLIENT_ID")
    if client_id:
        return ManagedIdentityCredential(client_id=client_id)
    return DefaultAzureCredential()


# ================== Function App ==================

# Use the base azure.functions app since we no longer need AgentFunctionApp's
# automatic agent endpoint registration — our agents run inside activities.
app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# ================== Activity: Query Azure Resources ==================


@app.activity_trigger(input_name="request")
async def query_azure_resources(request: dict) -> dict:
    """Run an Azure resource query via MCP server with the user's OBO token."""
    try:
        query = request.get("query", "")
        user_token = request.get("user_access_token", "")

        if not user_token:
            return {"result": "", "error": "No user access token provided."}

        mcp_url = os.environ.get("MCP_SERVER_URI")
        if not mcp_url:
            return {"result": "", "error": "MCP_SERVER_URI not configured."}

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

        http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {user_token}"},
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
            deployment_name=deployment,
            credential=_get_credential(),
        )

        agent = client.as_agent(
            name="AzureResourceAgent",
            instructions="""You are an Azure resource assistant for a support team.
Use the Azure MCP tools to answer questions about the user's Azure resources.
Be concise and factual. Format results clearly with resource names, types, locations, and key properties.
If you encounter permission errors, explain that the user may not have access to that resource or subscription.""",
            tools=mcp_tool,
        )

        try:
            async with mcp_tool:
                response = await agent.run(query)
                return {"result": response.text if hasattr(response, "text") else str(response)}
        finally:
            await http_client.aclose()

    except Exception as ex:
        logger.error(f"Azure resource query error: {ex}", exc_info=True)
        return {"result": "", "error": str(ex)}


# ================== Activity: Search Tickets ==================


@app.activity_trigger(input_name="request")
async def search_support_tickets(request: dict) -> dict:
    """Search Freshdesk tickets matching the query."""
    try:
        query = request.get("query", "")
        result = search_tickets(query)
        return {"result": result}
    except Exception as ex:
        logger.error(f"Ticket search error: {ex}", exc_info=True)
        return {"result": "", "error": str(ex)}


# ================== Activity: Route & Synthesize ==================


@app.activity_trigger(input_name="request")
async def route_query(request: dict) -> dict:
    """
    Use the LLM to decide which sources to query and return the routing decision.
    Returns: {"azure": bool, "tickets": bool}
    """
    try:
        query = request.get("query", "")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

        client = AzureOpenAIChatClient(
            endpoint=endpoint,
            deployment_name=deployment,
            credential=_get_credential(),
        )

        agent = client.as_agent(
            name="RouterAgent",
            instructions="""You are a routing agent. Given a support query, decide which information sources to check.
You MUST respond with ONLY a JSON object, no other text:
{"azure": true/false, "tickets": true/false}

Rules:
- Set "azure" to true if the query is about Azure resources, infrastructure, deployments, configuration, or status.
- Set "tickets" to true if the query is about past issues, known problems, previous resolutions, or support history.
- Set both to true if the query could benefit from both live Azure data and historical ticket context.
- At least one must be true.

Examples:
- "How many web apps do I have?" → {"azure": true, "tickets": false}
- "Has anyone seen this error before?" → {"azure": false, "tickets": true}
- "My app service is returning 503, any known issues?" → {"azure": true, "tickets": true}""",
        )

        response = await agent.run(query)
        text = response.text if hasattr(response, "text") else str(response)

        # Parse the JSON from the response
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        routing = json.loads(text)
        return {
            "azure": bool(routing.get("azure", False)),
            "tickets": bool(routing.get("tickets", False)),
        }
    except Exception as ex:
        logger.error(f"Routing error: {ex}", exc_info=True)
        # Default to both on error
        return {"azure": True, "tickets": True}


@app.activity_trigger(input_name="request")
async def synthesize_response(request: dict) -> dict:
    """Combine results from Azure and ticket searches into a coherent answer."""
    try:
        query = request.get("query", "")
        azure_result = request.get("azure_result", "")
        ticket_result = request.get("ticket_result", "")

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

        client = AzureOpenAIChatClient(
            endpoint=endpoint,
            deployment_name=deployment,
            credential=_get_credential(),
        )

        context_parts = []
        if azure_result:
            context_parts.append(f"## Azure Resource Information\n{azure_result}")
        if ticket_result:
            context_parts.append(f"## Related Support Tickets\n{ticket_result}")

        context = "\n\n".join(context_parts)

        agent = client.as_agent(
            name="SynthesisAgent",
            instructions="""You are a support assistant that synthesizes information from multiple sources.
Given the user's query and the gathered context (Azure resource data and/or support ticket history),
provide a clear, actionable answer.

Guidelines:
- Lead with the most relevant finding.
- If there are related past tickets, mention resolutions that might apply.
- If Azure resource data shows issues, highlight them clearly.
- Be concise but thorough. Use markdown formatting.
- If data from a source is missing or errored, work with what you have.""",
        )

        prompt = f"User query: {query}\n\n{context}"
        response = await agent.run(prompt)
        return {"result": response.text if hasattr(response, "text") else str(response)}

    except Exception as ex:
        logger.error(f"Synthesis error: {ex}", exc_info=True)
        # Fall back to raw results
        parts = []
        if request.get("azure_result"):
            parts.append(f"## Azure Resources\n{request['azure_result']}")
        if request.get("ticket_result"):
            parts.append(f"## Support Tickets\n{request['ticket_result']}")
        return {"result": "\n\n".join(parts) if parts else f"Error: {ex}"}


# ================== Orchestration ==================


@app.orchestration_trigger(context_name="context")
def support_query_orchestration(context: df.DurableOrchestrationContext):
    """
    Main orchestration:
    1. Router decides which sources to query.
    2. Queries run in parallel (Azure MCP + Freshdesk tickets).
    3. Synthesis agent combines results into a final answer.
    """
    request = context.get_input()
    query = request.get("query", "")
    user_token = request.get("user_access_token", "")

    # Step 1: Route
    context.set_custom_status({"step": "Routing", "message": "Analyzing your query..."})
    routing = yield context.call_activity("route_query", {"query": query})

    # Step 2: Fan out — query sources in parallel
    tasks = []
    source_labels = []

    if routing.get("azure"):
        context.set_custom_status({"step": "Querying", "message": "Checking Azure resources..."})
        tasks.append(
            context.call_activity(
                "query_azure_resources",
                {"query": query, "user_access_token": user_token},
            )
        )
        source_labels.append("azure")

    if routing.get("tickets"):
        context.set_custom_status({"step": "Querying", "message": "Searching support tickets..."})
        tasks.append(
            context.call_activity("search_support_tickets", {"query": query})
        )
        source_labels.append("tickets")

    # Wait for all tasks
    results = yield context.task_all(tasks)

    # Map results back to source names
    azure_result = ""
    ticket_result = ""
    for label, result in zip(source_labels, results):
        if label == "azure":
            azure_result = result.get("result", result.get("error", ""))
        elif label == "tickets":
            ticket_result = result.get("result", "")

    # Step 3: Synthesize
    context.set_custom_status({"step": "Synthesizing", "message": "Preparing your answer..."})
    synthesis = yield context.call_activity(
        "synthesize_response",
        {
            "query": query,
            "azure_result": azure_result,
            "ticket_result": ticket_result,
        },
    )

    context.set_custom_status({"step": "Completed", "message": "Done"})
    return {"result": synthesis.get("result", ""), "sources": source_labels}


# ================== HTTP Endpoints ==================


@app.function_name(name="SubmitQuery")
@app.route(route="query", methods=["POST"])
@app.durable_client_input(client_name="client")
async def submit_query(req: func.HttpRequest, client) -> func.HttpResponse:
    """
    Start a support query.
    Body: { "query": "...", "userAccessToken": "..." }
    Returns: { "id": "<orchestration-id>" }
    """
    try:
        body = req.get_json()
        query = body.get("query", "")
        user_token = body.get("userAccessToken", "")

        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'query' in request body"}),
                status_code=400,
                mimetype="application/json",
            )

        instance_id = await client.start_new(
            "support_query_orchestration",
            client_input={"query": query, "user_access_token": user_token},
        )

        return func.HttpResponse(
            json.dumps({"id": instance_id}),
            status_code=202,
            mimetype="application/json",
        )
    except Exception as ex:
        logger.error(f"Error starting query: {ex}")
        return func.HttpResponse(
            json.dumps({"error": str(ex)}),
            status_code=500,
            mimetype="application/json",
        )


@app.function_name(name="GetQueryStatus")
@app.route(route="query/status/{instance_id}", methods=["GET"])
@app.durable_client_input(client_name="client")
async def get_query_status(req: func.HttpRequest, client) -> func.HttpResponse:
    """Check the status of a support query orchestration."""
    try:
        instance_id = req.route_params.get("instance_id")
        status = await client.get_status(instance_id)

        if not status:
            return func.HttpResponse(
                json.dumps({"error": "Query not found"}),
                status_code=404,
                mimetype="application/json",
            )

        return func.HttpResponse(
            json.dumps({
                "id": status.instance_id,
                "runtimeStatus": status.runtime_status.name,
                "output": status.output,
                "customStatus": status.custom_status,
            }),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as ex:
        logger.error(f"Error getting status: {ex}")
        return func.HttpResponse(
            json.dumps({"error": str(ex)}),
            status_code=500,
            mimetype="application/json",
        )
