"""
Azure Functions Durable Agent using Microsoft Agent Framework.

This implementation uses AgentFunctionApp from agent-framework for durable agents
with hosted MCP tools configured in Azure AI Foundry portal.

Features:
- Durable agents with automatic state persistence
- Auto-generated HTTP endpoints: /api/agents/{agentName}/run
- Hosted MCP tools via Foundry connection (OAuth identity passthrough)
- Thread continuity via thread_id parameter

Endpoints:
  POST /api/agents/SREAgent/run
  - Body: Plain text message OR JSON {"input": "message", "thread_id": "optional"}
  - Response headers: x-ms-thread-id

  GET /api/health
  - Health check with agent and tool status
"""

import os
import logging
from azure.identity import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.azure import AzureOpenAIResponsesClient, AgentFunctionApp

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

# MCP Server configuration - must match the connection name configured in Foundry portal
MCP_TOOL_CONNECTION_ID = os.environ.get("MCP_TOOL_CONNECTION_ID", "AzureMCP")

# =============================================================================
# Create Agent with Foundry-hosted MCP Tools
# =============================================================================

def create_sre_agent() -> Agent:
    """
    Create the SRE agent with Foundry-hosted MCP tools.
    
    The MCP tools are configured in Azure AI Foundry portal with OAuth identity
    passthrough. The agent-framework automatically handles:
    - Tool discovery from MCP server
    - OAuth consent flow when user first invokes MCP tools
    - Credential caching after user consents
    """
    # Initialize Azure OpenAI Responses client
    # This client connects to Azure AI Foundry project for hosted MCP tools
    client = AzureOpenAIResponsesClient(
        project_endpoint=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"),
        deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        credential=DefaultAzureCredential(),
    )
    
    # Get MCP tools from Foundry connection
    # The connection is configured in Foundry portal with OAuth identity passthrough
    tools = []
    if MCP_TOOL_CONNECTION_ID:
        try:
            mcp_tool = client.get_mcp_tool(
                connection_id=MCP_TOOL_CONNECTION_ID,
                require_approval="never",  # Tools auto-execute without confirmation
            )
            tools.append(mcp_tool)
            logger.info(f"Loaded MCP tools from connection: {MCP_TOOL_CONNECTION_ID}")
        except Exception as e:
            logger.warning(f"Failed to load MCP tools: {e}")
    
    # Create agent with the client and MCP tools
    agent = client.as_agent(
        name="SREAgent",
        instructions=SRE_AGENT_INSTRUCTIONS,
        tools=tools if tools else None,
    )
    
    return agent


# =============================================================================
# Create AgentFunctionApp (Durable Agent Host)
# =============================================================================

# Create the SRE agent
sre_agent = create_sre_agent()

# Create AgentFunctionApp - this automatically:
# - Registers agents with Durable Task worker
# - Generates HTTP endpoints: /api/agents/{agentName}/run
# - Handles thread persistence via thread_id
# - Provides health check endpoint
app = AgentFunctionApp(
    agents=[sre_agent],
    enable_health_check=True,
)