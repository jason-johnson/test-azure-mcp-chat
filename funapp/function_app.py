"""
Azure Durable Functions Chat Agent using Microsoft Agent Framework.

This is a minimal implementation that leverages the Agent Framework to:
- Automatically handle HTTP endpoints via AgentFunctionApp
- Persist conversation threads using Durable Functions
- Integrate with MCP tools via Azure AI Foundry

Usage:
  POST /api/agents/SREAgent/run
  Body: "Your question here"
  
  # Continue conversation with thread_id:
  POST /api/agents/SREAgent/run?thread_id=<thread_id>
  Body: "Follow-up question"
"""

import os
import logging
from azure.identity import DefaultAzureCredential
from agent_framework.azure import AzureOpenAIChatClient, AgentFunctionApp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Agent Configuration
# =============================================================================

SRE_AGENT_INSTRUCTIONS = """
You are an expert Azure Site Reliability Engineer (SRE) assistant.

**Your Capabilities:**
- Query and manage Azure resources across subscriptions
- Diagnose issues with Azure services (App Services, AKS, Storage, etc.)
- Provide operational insights and recommendations
- Execute Azure operations using available tools

**Guidelines:**
- Always use tools to get real-time information about Azure resources
- Be concise but thorough in your responses
- Offer follow-up actions when appropriate
- If an operation requires elevated permissions, explain what's needed
"""


def create_app() -> AgentFunctionApp:
    """
    Create the AgentFunctionApp with SRE Agent.
    
    The AgentFunctionApp automatically:
    - Creates HTTP endpoints at /api/agents/{name}/run
    - Persists conversation threads via Durable Functions
    - Handles failure recovery and state management
    """
    
    # Get configuration from environment
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    mcp_connection_id = os.environ.get("MCP_TOOL_CONNECTION_ID")
    
    if not endpoint:
        raise ValueError(
            "AZURE_AI_PROJECT_ENDPOINT is not set. "
            "Expected format: https://<resource>.services.ai.azure.com/api/projects/<project>"
        )
    
    logger.info(f"Initializing agent with endpoint: {endpoint}")
    logger.info(f"Using model deployment: {deployment_name}")
    
    # Create Azure OpenAI client with Managed Identity
    client = AzureOpenAIChatClient(
        endpoint=endpoint,
        deployment_name=deployment_name,
        credential=DefaultAzureCredential(),
    )
    
    # Check if MCP tools are configured via Foundry
    # If MCP_TOOL_CONNECTION_ID is set, use Foundry Tools which handles auth automatically
    if mcp_connection_id:
        logger.info(f"Using Foundry Tools with MCP connection: {mcp_connection_id}")
        # use_foundry_tools() loads MCP tools from the Foundry project's connected resources
        # Auth is handled by Foundry - no manual token management needed
        tools_client = client.use_foundry_tools()
        sre_agent = tools_client.as_agent(
            name="SREAgent",
            instructions=SRE_AGENT_INSTRUCTIONS,
        )
    else:
        logger.info("No MCP_TOOL_CONNECTION_ID set - running without MCP tools")
        logger.info("To enable MCP, configure an MCP connection in Azure AI Foundry portal")
        sre_agent = client.as_agent(
            name="SREAgent",
            instructions=SRE_AGENT_INSTRUCTIONS,
        )
    
    # Create the Function App - this handles everything automatically:
    # - POST /api/agents/SREAgent/run endpoint
    # - GET /api/health endpoint  
    # - Durable conversation persistence via thread_id
    # - Async mode via x-ms-wait-for-response header
    app = AgentFunctionApp(
        agents=[sre_agent],
        enable_health_check=True,
    )
    
    logger.info("AgentFunctionApp initialized successfully")
    logger.info("Endpoints available:")
    logger.info("  POST /api/agents/SREAgent/run - Chat with the SRE agent")
    logger.info("  GET /api/health - Health check")
    
    return app


# Initialize the app
# This is the only line needed to expose the agent as an Azure Function
app = create_app()