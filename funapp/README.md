# Azure Functions Durable Agent

A chat agent using **Microsoft Agent Framework** with **AgentFunctionApp** for durable agents and **Foundry-hosted MCP tools** with **OAuth Identity Passthrough**.

## Architecture

This implementation uses **Agent Framework with Azure Functions** which provides:

| Feature | How It's Handled |
|---------|------------------|
| HTTP Endpoints | `AgentFunctionApp` auto-generates `/api/agents/{name}/run` |
| State Persistence | Durable Task framework with automatic replay |
| Conversation State | Thread continuity via `thread_id` |
| MCP Tools | Foundry-hosted with OAuth identity passthrough |
| Token Management | Foundry handles consent, tokens, refresh |

## Why Agent Framework + Foundry MCP?

| Concern | Agent Framework Handles It |
|---------|---------------------------|
| HTTP endpoint creation | ✅ Auto-generated |
| Thread management | ✅ Built-in |
| Error recovery | ✅ Durable replay |
| MCP tool discovery | ✅ `get_mcp_tool()` from Foundry |
| User identity | ✅ OAuth identity passthrough |

## OAuth Identity Passthrough

From [Microsoft docs](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/mcp-authentication):

With OAuth identity passthrough:
- Each user signs in with their own account
- MCP tools access Azure resources with user's permissions
- Full Azure RBAC enforcement
- Audit trail shows who did what

## Endpoints

### Run the Agent
```bash
# Simple text request
curl -X POST https://<function-app>.azurewebsites.net/api/agents/SREAgent/run \
  -H "Content-Type: text/plain" \
  -d "How many web apps are in my subscription?"

# JSON request with thread_id for conversation continuity
curl -X POST https://<function-app>.azurewebsites.net/api/agents/SREAgent/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Show details of the first one", "thread_id": "<thread-id>"}'

# Response includes x-ms-thread-id header for follow-up requests
```

### Health Check
```bash
curl https://<function-app>.azurewebsites.net/api/health
# Returns agent status and tool configuration
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_AI_PROJECT_ENDPOINT` | Azure AI Foundry project endpoint | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Model deployment name (default: gpt-4o) | Yes |
| `MCP_TOOL_CONNECTION_ID` | Name of MCP connection in Foundry (default: AzureMCP) | Yes |

## Setup MCP Connection with OAuth Identity Passthrough

### 1. Configure MCP Connection in Foundry Portal

1. Go to [Azure AI Foundry](https://ai.azure.com)
2. Select your project
3. Navigate to **Connected resources** → **Tools**
4. Click **Connect a tool** → **Custom** → **MCP**
5. Configure:
   - **Name**: `AzureMCP` (must match `MCP_CONNECTION_NAME` env var)
   - **MCP Server endpoint**: `https://<mcp-app>.azurewebsites.net`
   - **Authentication**: **OAuth Identity Passthrough**

### 2. For Custom OAuth (with Entra ID)

If using your own Entra app registration:

1. Create an [app registration](https://learn.microsoft.com/en-us/entra/identity-platform/quickstart-register-app)
2. Configure API permissions for the MCP server
3. In Foundry connection, provide:
   - **Client ID**: Your app's client ID
   - **Client Secret**: Your app's client secret
   - **Token URL**: `https://login.microsoftonline.com/{tenantId}/oauth2/v2.0/token`
   - **Auth URL**: `https://login.microsoftonline.com/{tenantId}/oauth2/v2.0/authorize`
   - **Refresh URL**: `https://login.microsoftonline.com/{tenantId}/oauth2/v2.0/token`
   - **Scopes**: `api://{MCP_API_CLIENT_ID}/.default`
4. Add the **redirect URL** (provided by Foundry) to your app registration

### 3. Role Requirements

Users need at least **Azure AI User** role on the Foundry project to use OAuth identity passthrough.

## Local Development

### Prerequisites
1. Azure CLI logged in (`az login`)
2. Python 3.11+
3. Azure Functions Core Tools
4. MCP connection configured in Foundry portal

### Setup
```bash
cd funapp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AZURE_AI_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"
export MCP_URL="https://<mcp-app>.azurewebsites.net"
export MCP_CONNECTION_NAME="AzureMCP"

# Start function app
func start
```

### Test Locally
```bash
# Chat with agent (no bearer token needed - consent handled by Agent Service)
curl -X POST http://localhost:7071/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What Azure subscriptions do I have access to?"}'

# If response includes requires_consent=true, open consent_link in browser,
# sign in, then continue with previous_response_id
```

## Comparison: Old vs New

| Aspect | Old (app/agent.py) | New (funapp/) |
|--------|-------------------|---------------|
| Framework | Semantic Kernel | Azure AI Foundry Agent Service |
| HTTP Routing | Manual FastAPI routes | Azure Functions triggers |
| User Authentication | Manual MSAL + OBO | OAuth identity passthrough |
| Token Management | Manual caching w/ TTL | Agent Service handles it |
| Conversation State | Manual in-memory cache | Agent Service threads |
| MCP Setup | Code-based configuration | Foundry portal configuration |
| User Identity | ✅ Preserved via OBO | ✅ Preserved via OAuth passthrough |

## Deployment

Deployment is via ADO Pipeline:
1. Commit changes: `git add . && git commit -m "Update agent"`
2. Push: `git push origin main`
3. Monitor pipeline in Azure DevOps
4. Verify: `az webapp log tail --name <function-app-name> --resource-group <rg>`

## References

- [MCP Authentication in Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/mcp-authentication)
- [Azure AI Projects SDK](https://pypi.org/project/azure-ai-projects/)
