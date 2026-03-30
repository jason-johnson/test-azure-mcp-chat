# =============================================================================
# Azure Functions App with Microsoft Agent Framework
# =============================================================================

import os
import logging
import sys

import azure.functions as func
from azure.identity import DefaultAzureCredential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Agent Instructions
# =============================================================================
SRE_INSTRUCTIONS = """
Role: Azure Service Reliability Engineer (SRE)

You are an expert Azure SRE assistant with direct access to Azure operations through MCP tools.

Capabilities:
- Query and manage Azure resources (subscriptions, resource groups, web apps, AKS, storage)
- Diagnose issues and troubleshoot Azure resources
- Provide operational insights and recommendations
- Execute Azure CLI commands and ARM operations

Guidelines:
1. Use appropriate tools based on the user's request
2. For Azure operations, use the provided MCP tools
3. Present results clearly with relevant details
4. Offer follow-up assistance and recommendations
5. If an operation fails, explain the error and suggest alternatives

Always prioritize safety - confirm destructive operations before executing.
"""

# =============================================================================
# Create the Function App
# =============================================================================
def create_app():
    """Create either AgentFunctionApp or simple FunctionApp."""
    # Try to create AgentFunctionApp first
    try:
        from agent_framework import tool
        from agent_framework.azure import AgentFunctionApp, AzureOpenAIResponsesClient
        
        project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
        deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        if not project_endpoint:
            logger.warning("AZURE_AI_PROJECT_ENDPOINT not set - falling back to simple app")
            raise ValueError("No project endpoint")
        
        logger.info(f"Initializing Agent Framework with endpoint: {project_endpoint}")
        
        @tool(description="Get current agent status and configuration")
        def get_agent_status() -> str:
            """Returns the agent's current status and configuration."""
            return f"""
Agent Status: Online
Project Endpoint: {os.environ.get('AZURE_AI_PROJECT_ENDPOINT', 'Not configured')}
Model Deployment: {os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', 'Not configured')}
MCP Server: {os.environ.get('MCP_URL', 'Not configured')}
"""
        
        client = AzureOpenAIResponsesClient(
            project_endpoint=project_endpoint,
            deployment_name=deployment_name,
            credential=DefaultAzureCredential(),
        )
        
        sre_agent = client.as_agent(
            name="SREAgent",
            instructions=SRE_INSTRUCTIONS,
            tools=[get_agent_status],
        )
        
        logger.info("SRE Agent created successfully")
        
        agent_app = AgentFunctionApp(
            agents=[sre_agent],
            enable_health_check=True,
        )
        
        logger.info("AgentFunctionApp initialized")
        return agent_app
        
    except Exception as e:
        logger.warning(f"AgentFunctionApp not available ({e}), using simple FunctionApp")
        
        # Create simple FunctionApp as fallback
        simple_app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
        
        @simple_app.route(route="health", methods=["GET"])
        def health_check(req: func.HttpRequest) -> func.HttpResponse:
            """Simple health check."""
            return func.HttpResponse(
                '{"status": "healthy", "message": "Python worker is running (simple mode)"}',
                mimetype="application/json",
                status_code=200
            )
        
        @simple_app.route(route="test", methods=["GET"])
        def test_endpoint(req: func.HttpRequest) -> func.HttpResponse:
            """Test endpoint."""
            endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "NOT SET")
            return func.HttpResponse(
                f'{{"status": "ok", "python_version": "{sys.version}", "endpoint": "{endpoint[:50] if endpoint else "NOT SET"}...", "mode": "simple"}}',
                mimetype="application/json",
                status_code=200
            )
        
        return simple_app

# Single app instance - this is what Azure Functions expects
app = create_app()
