# =============================================================================
# Azure Functions App with Microsoft Agent Framework
# =============================================================================
# This creates an SRE agent that can interact with Azure resources via MCP tools

import os
import logging
from typing import Annotated

from pydantic import Field
from azure.identity import DefaultAzureCredential
from agent_framework import tool
from agent_framework.azure import AgentFunctionApp, AzureOpenAIResponsesClient

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
# Custom Tools (can be extended with MCP tool wrappers)
# =============================================================================
@tool(description="Get current agent status and configuration")
def get_agent_status() -> str:
    """Returns the agent's current status and configuration."""
    return f"""
Agent Status: Online
Project Endpoint: {os.environ.get('AZURE_AI_PROJECT_ENDPOINT', 'Not configured')}
Model Deployment: {os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', 'Not configured')}
MCP Server: {os.environ.get('MCP_URL', 'Not configured')}
"""


# =============================================================================
# Create Agent Function App
# =============================================================================
def create_app() -> AgentFunctionApp:
    """Create and configure the AgentFunctionApp with SRE Agent."""
    
    # Get configuration from environment
    project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    
    if not project_endpoint:
        raise ValueError("AZURE_AI_PROJECT_ENDPOINT environment variable is required")
    
    logger.info(f"Initializing Agent Framework with endpoint: {project_endpoint}")
    
    # Create Azure OpenAI client with managed identity
    client = AzureOpenAIResponsesClient(
        project_endpoint=project_endpoint,
        deployment_name=deployment_name,
        credential=DefaultAzureCredential(),
    )
    
    # Create the SRE Agent
    # TODO: Add MCP tools integration when Foundry MCP connection is configured
    sre_agent = client.as_agent(
        name="SREAgent",
        instructions=SRE_INSTRUCTIONS,
        tools=[get_agent_status],  # Add more tools here
    )
    
    logger.info("SRE Agent created successfully")
    
    # Create the Function App
    # This automatically generates:
    # - POST /api/agents/SREAgent/run - Main agent endpoint
    # - GET /api/health - Health check (if enabled)
    app = AgentFunctionApp(
        agents=[sre_agent],
        enable_health_check=True,
    )
    
    logger.info("AgentFunctionApp initialized")
    
    return app


# Initialize the app
app = create_app()
