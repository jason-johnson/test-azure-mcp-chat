# Azure Durable Functions Agent

A minimal chat agent implementation using the Microsoft Agent Framework with Azure Durable Functions.

## Architecture

This implementation uses **AgentFunctionApp** from the Microsoft Agent Framework, which provides:

| Feature | How It's Handled |
|---------|------------------|
| HTTP Endpoints | Auto-generated: `POST /api/agents/SREAgent/run` |
| Conversation Persistence | Automatic via `thread_id` parameter |
| State Management | Durable Functions - survives restarts/failures |
| MCP Tools | Azure AI Foundry Tool Connection |
| Authentication | Managed by Azure AI Foundry |

## Endpoints

### Chat with the Agent
```bash
# Start a new conversation
curl -X POST https://<function-app>.azurewebsites.net/api/agents/SREAgent/run \
  -H "Content-Type: text/plain" \
  -d "How many web apps are in my subscription?"

# Response includes thread_id in x-ms-thread-id header
```

### Continue a Conversation
```bash
# Use thread_id from previous response
curl -X POST "https://<function-app>.azurewebsites.net/api/agents/SREAgent/run?thread_id=<thread-id>" \
  -H "Content-Type: text/plain" \
  -d "Show me details of the first one"
```

### Async Mode (Long-running queries)
```bash
# Set header to get 202 Accepted immediately
curl -X POST https://<function-app>.azurewebsites.net/api/agents/SREAgent/run \
  -H "Content-Type: text/plain" \
  -H "x-ms-wait-for-response: false" \
  -d "Analyze all my Azure resources"
# Returns 202 with status URL to poll
```

### Health Check
```bash
curl https://<function-app>.azurewebsites.net/api/health
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_AI_PROJECT_ENDPOINT` | Azure AI Foundry project endpoint | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Model deployment name (default: gpt-4o) | Yes |
| `MCP_TOOL_CONNECTION_ID` | Foundry MCP tool connection name | For MCP tools |

## Local Development

### Prerequisites
1. Azure CLI logged in (`az login`)
2. Python 3.11+
3. Azure Functions Core Tools

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
export MCP_TOOL_CONNECTION_ID="AzureMCP"  # Optional

# Start function app
func start
```

### Test Locally
```bash
# Chat with agent
curl -X POST http://localhost:7071/api/agents/SREAgent/run \
  -H "Content-Type: text/plain" \
  -d "What Azure subscriptions do I have access to?"
```

## MCP Tools Setup

MCP tools are configured in Azure AI Foundry as a Tool Connection. This means:
- **No manual token management** - Foundry handles authentication
- **No MCP code in the app** - Tools are loaded automatically
- **Centralized configuration** - Manage in Foundry portal

### Configure MCP Connection (via Terraform)
The MCP connection is created automatically via `infrastructure/ai_foundry.tf`:
```hcl
resource "azapi_resource" "mcp_connection" {
  type      = "Microsoft.MachineLearningServices/workspaces/connections@2024-10-01"
  name      = "AzureMCP"
  parent_id = azapi_resource.ai_hub.id
  body = {
    properties = {
      category = "ModelContextProtocol"
      target   = "https://<mcp-app>.azurewebsites.net"
      authType = "AAD"
    }
  }
}
```

### Manual Configuration (Azure Portal)
1. Go to [Azure AI Foundry](https://ai.azure.com)
2. Select your project
3. Navigate to **Connected resources**
4. Add **Model Context Protocol tool**
5. Configure:
   - Name: `AzureMCP`
   - Endpoint: `https://<mcp-server>.azurewebsites.net`
   - Authentication: Azure AD

## Comparison: Old vs New

| Aspect | Old (app/agent.py) | New (funapp/) |
|--------|-------------------|---------------|
| Lines of Code | ~400+ | ~90 |
| State Management | Manual in-memory cache | Automatic Durable Functions |
| HTTP Routing | Manual FastAPI routes | Auto-generated |
| Token Refresh | Manual handling | Managed by Foundry |
| MCP Authentication | Manual OBO flow | Foundry handles it |
| Conversation History | Manual ChatHistory | Automatic thread persistence |
| Error Recovery | None | Built-in replay |

## Deployment

Deployment is via ADO Pipeline:
1. Commit changes: `git add . && git commit -m "Update agent"`
2. Push: `git push origin main`
3. Monitor pipeline in Azure DevOps
4. Verify: `az webapp log tail --name <function-app-name> --resource-group <rg>`
