"""Post-provision script: create or update the Azure AI Foundry Agent with MCP tool."""

import os
import sys

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import McpTool
from azure.identity import DefaultAzureCredential

AGENT_NAME = "azure-mcp-agent"
AGENT_MODEL = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

INSTRUCTIONS_TEMPLATE = """\
You are a helpful assistant that can manage Azure resources. Use the available tools to list, \
inspect, and manage Azure resources on behalf of the user. Be concise and format responses in markdown.

Always use tenant ID '{tenant_id}' when calling Azure tools unless the user specifies a different tenant.\
"""


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"ERROR: Required environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return value


def main() -> None:
    project_endpoint = get_required_env("FOUNDRY_PROJECT_ENDPOINT")
    mcp_server_uri = get_required_env("MCP_SERVER_URI")
    tenant_id = get_required_env("AZURE_TENANT_ID")

    instructions = INSTRUCTIONS_TEMPLATE.format(tenant_id=tenant_id)

    credential = DefaultAzureCredential()
    client = AgentsClient(endpoint=project_endpoint, credential=credential)

    # Configure MCP tool pointing to the Azure MCP server
    mcp_tool = McpTool(server_label="azure-mcp", server_url=mcp_server_uri)
    mcp_tool.set_approval_mode("never")

    # Check for an existing agent (stored from a previous run)
    existing_agent_id = os.environ.get("AZURE_AI_AGENT_ID")
    if existing_agent_id:
        try:
            existing = client.get_agent(existing_agent_id)
            print(f"Updating existing agent '{existing.name}' ({existing.id})...")
            agent = client.update_agent(
                assistant_id=existing_agent_id,
                model=AGENT_MODEL,
                name=AGENT_NAME,
                instructions=instructions,
                tools=mcp_tool.definitions,
                tool_resources=mcp_tool.resources,
            )
            print(f"Agent updated: {agent.id}")
            return
        except Exception:
            print("Previous agent not found, creating a new one...")

    # List existing agents to avoid duplicates
    agents_list = client.list_agents()
    for existing in agents_list.data:
        if existing.name == AGENT_NAME:
            print(f"Found existing agent '{AGENT_NAME}' ({existing.id}), updating...")
            agent = client.update_agent(
                assistant_id=existing.id,
                model=AGENT_MODEL,
                name=AGENT_NAME,
                instructions=instructions,
                tools=mcp_tool.definitions,
                tool_resources=mcp_tool.resources,
            )
            print(f"Agent updated: {agent.id}")
            # Output for azd env set
            print(f"AZURE_AI_AGENT_ID={agent.id}")
            return

    # Create new agent
    print(f"Creating new agent '{AGENT_NAME}'...")
    agent = client.create_agent(
        model=AGENT_MODEL,
        name=AGENT_NAME,
        instructions=instructions,
        tools=mcp_tool.definitions,
        tool_resources=mcp_tool.resources,
    )
    print(f"Agent created: {agent.id}")
    # Output for azd env set
    print(f"AZURE_AI_AGENT_ID={agent.id}")


if __name__ == "__main__":
    main()
