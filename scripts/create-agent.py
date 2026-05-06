"""Post-provision script: create OAuth Identity Passthrough MCP connection and Foundry Agent."""

import json
import os
import subprocess
import sys

import requests
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import McpTool
from azure.identity import DefaultAzureCredential

AGENT_NAME = "azure-mcp-agent"
AGENT_MODEL = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
MCP_CONNECTION_NAME = "azure-mcp-connection"

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


def az_cli(*args: str) -> str:
    """Run an az CLI command and return stdout."""
    result = subprocess.run(
        ["az", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def create_client_secret(client_app_id: str) -> str:
    """Create a client secret on the MCP client app registration."""
    print("Creating client secret on MCP client app registration...")
    output = az_cli(
        "ad", "app", "credential", "reset",
        "--id", client_app_id,
        "--display-name", "foundry-oauth-passthrough",
        "--years", "2",
        "--query", "password",
        "-o", "tsv",
    )
    return output


def create_oauth_connection(
    credential: DefaultAzureCredential,
    resource_group: str,
    account_name: str,
    project_name: str,
    mcp_server_uri: str,
    client_app_id: str,
    client_secret: str,
    server_app_id: str,
    tenant_id: str,
) -> None:
    """Create or update the OAuth Identity Passthrough connection on the Foundry project."""
    print(f"Creating OAuth Identity Passthrough connection '{MCP_CONNECTION_NAME}'...")

    token = credential.get_token("https://management.azure.com/.default").token
    subscription_id = get_required_env("AZURE_SUBSCRIPTION_ID")

    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.CognitiveServices/accounts/{account_name}"
        f"/projects/{project_name}"
        f"/connections/{MCP_CONNECTION_NAME}"
        f"?api-version=2025-06-01"
    )

    body = {
        "properties": {
            "category": "RemoteTool",
            "target": mcp_server_uri,
            "authType": "OAuth2",
            "isSharedToAll": True,
            "credentials": {
                "clientId": client_app_id,
                "clientSecret": client_secret,
                "tenantId": tenant_id,
                "authUrl": f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize",
                "refreshToken": f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            },
            "metadata": {
                "ApiType": "Azure",
                "Scope": f"api://{server_app_id}/Mcp.Tools.ReadWrite",
                "TokenUrl": f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            },
        }
    }

    resp = requests.put(url, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }, json=body, timeout=60)
    resp.raise_for_status()

    result = resp.json()
    print(f"Connection '{MCP_CONNECTION_NAME}' created/updated successfully.")

    # Check for redirect URI in the response
    props = result.get("properties", {})
    redirect_uri = props.get("metadata", {}).get("RedirectUri", "")
    if redirect_uri:
        print(f"Adding redirect URI to client app registration: {redirect_uri}")
        try:
            # Get existing redirect URIs
            existing = az_cli(
                "ad", "app", "show",
                "--id", client_app_id,
                "--query", "web.redirectUris",
                "-o", "json",
            )
            uris = json.loads(existing) if existing else []
            if redirect_uri not in uris:
                uris.append(redirect_uri)
                az_cli(
                    "ad", "app", "update",
                    "--id", client_app_id,
                    "--web-redirect-uris", *uris,
                )
                print("Redirect URI added.")
            else:
                print("Redirect URI already present.")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Could not update redirect URIs: {e}", file=sys.stderr)
            print(f"Manually add this redirect URI to app {client_app_id}: {redirect_uri}")


def main() -> None:
    project_endpoint = get_required_env("FOUNDRY_PROJECT_ENDPOINT")
    mcp_server_uri = get_required_env("MCP_SERVER_URI")
    tenant_id = get_required_env("AZURE_TENANT_ID")
    client_app_id = get_required_env("MCP_CLIENT_CLIENT_ID")
    server_app_id = get_required_env("MCP_SERVER_CLIENT_ID")
    resource_group = get_required_env("RESOURCE_GROUP")

    # Parse account name and project name from the project endpoint
    # Format: https://<account>.cognitiveservices.azure.com/api/projects/<project>
    parts = project_endpoint.rstrip("/").split("/")
    project_name = parts[-1]
    account_host = parts[2]  # <account>.cognitiveservices.azure.com
    account_name = account_host.split(".")[0]

    instructions = INSTRUCTIONS_TEMPLATE.format(tenant_id=tenant_id)

    credential = DefaultAzureCredential()

    # Step 1: Create client secret for OAuth flow
    client_secret = create_client_secret(client_app_id)

    # Step 2: Create OAuth Identity Passthrough connection on the Foundry project
    create_oauth_connection(
        credential=credential,
        resource_group=resource_group,
        account_name=account_name,
        project_name=project_name,
        mcp_server_uri=mcp_server_uri,
        client_app_id=client_app_id,
        client_secret=client_secret,
        server_app_id=server_app_id,
        tenant_id=tenant_id,
    )

    # Step 3: Create or update the agent with MCP tool referencing the connection
    client = AgentsClient(endpoint=project_endpoint, credential=credential)

    mcp_tool = McpTool(server_label="azure-mcp", server_url=mcp_server_uri)
    mcp_tool.set_approval_mode("never")
    mcp_tool._definition["project_connection_id"] = MCP_CONNECTION_NAME

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
