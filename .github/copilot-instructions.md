# Instructions for coding agents

This repository is a Semantic Kernel agent application that connects to remote azure-mcp server, designed to run on azure app service.

## ⚠️ CRITICAL: Azure DevOps (ADO) Pipeline Deployment Only

**DO NOT run local deployment scripts like `redeploy-app.sh` - they are not used for deployment!**
**DO NOT run git commands like `git add`, `git commit`, or `git push` - user handles git workflow!**

### Proper Deployment Process
1. Make code changes in `/app/` directory
2. User commits: `git add . && git commit -m "message"`
3. User pushes: `git push origin main`
4. Monitor ADO pipeline (auto-triggers on push)
5. Verify via logs: `az webapp log tail --name $(az webapp list --resource-group rg-mcpchat-dev-chn-main --query "[?contains(name, 'fe')].name" -o tsv) --resource-group rg-mcpchat-dev-chn-main`

**⚠️ IMPORTANT: Never use `az webapp log download` - the zip file will be huge and cause problems. Use `az webapp log tail` or other non-download methods to check logs.**

---

## Authentication Architecture

This app uses **MSAL (Microsoft Authentication Library)** for direct OAuth 2.0 authentication with Microsoft Entra ID.

### Two Authentication Modes:

| Mode | Environment | How It Works |
|------|-------------|--------------|
| **DEV_MODE** | Local development | Accept bearer tokens via `Authorization: Bearer <token>` header |
| **Production** | Azure App Service | MSAL browser-based OAuth flow with session cookies |

### Entra ID Requirements

The following App Registrations are required (created by Terraform):

1. **Frontend App Registration** (`azuread_application.fe`)
   - Client ID: `AZURE_CLIENT_ID` env var
   - Client Secret: `MICROSOFT_PROVIDER_AUTHENTICATION_SECRET` env var
   - Redirect URIs:
     - `http://localhost:8000/auth/callback` (local development)
     - `https://<app-name>.azurewebsites.net/auth/callback` (production)
   - API Permissions: `api://{MCP_API_CLIENT_ID}/Mcp.Tools.ReadWrite` (delegated)

2. **MCP API App Registration** (`azuread_application.mcp`)
   - Client ID: `MCP_API_CLIENT_ID` env var
   - Exposes scope: `Mcp.Tools.ReadWrite`

### Adding Localhost Redirect URI

To test locally with browser auth, add the localhost redirect URI to the frontend App Registration:

```bash
# Get the app registration object ID
APP_ID=$(az ad app list --display-name "your-fe-app-name" --query "[0].id" -o tsv)

# Add localhost redirect URI
az ad app update --id $APP_ID --web-redirect-uris \
  "http://localhost:8000/auth/callback" \
  "https://your-app.azurewebsites.net/.auth/login/aad/callback"
```

---

## Testing the Application

### Local Development (DEV_MODE)

**Start the server in DEV_MODE:**
```bash
cd /workspaces/azure-mcp-chat/app
DEV_MODE=true uvicorn agent:app --reload --port 8000
```

**Test with bearer token (similar to the old approach):**
```bash
# Get a token for the MCP API
ACCESS_TOKEN=$(az account get-access-token --resource "api://923d2ffc-9173-49c6-94bb-06a2dee07a50" --query accessToken -o tsv)

# Test the chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "x-user-id: test-user-789" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "user_input=how many linux web apps do I have in subscription 2daa7beb-ac1d-473e-84e2-f3cd40e584de?"
```

**Test debug endpoints:**
```bash
# Test auth
curl http://localhost:8000/debug/test-auth \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "x-user-id: test-user"

# Test MCP connection
curl http://localhost:8000/debug/test-mcp \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "x-user-id: test-user"
```

### Production Mode (MSAL Browser Auth)

**Start the server in production mode:**
```bash
cd /workspaces/azure-mcp-chat/app
uvicorn agent:app --reload --port 8000
```

**Required environment variables:**
```bash
export AZURE_CLIENT_ID="your-frontend-client-id"
export MICROSOFT_PROVIDER_AUTHENTICATION_SECRET="your-client-secret"
export TENANT_ID="your-tenant-id"
export MCP_API_CLIENT_ID="your-mcp-api-client-id"
export BASE_URL="http://localhost:8000"  # or production URL
export SESSION_SECRET_KEY="a-random-secret-key"  # optional, generates random if not set
```

Then visit `http://localhost:8000` in a browser - you'll be redirected to Microsoft login.

---

### Key Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEV_MODE` | Set to `true` for bearer token auth mode | No (default: false) |
| `AZURE_CLIENT_ID` | Frontend app registration client ID | Yes |
| `MICROSOFT_PROVIDER_AUTHENTICATION_SECRET` | Frontend app client secret | Yes (production) |
| `TENANT_ID` | Azure AD tenant ID | Yes |
| `MCP_API_CLIENT_ID` | MCP server app registration client ID | Yes |
| `MCP_URL` | URL of the MCP server | Yes |
| `BASE_URL` | Base URL of this app (for redirect URI) | Yes (production) |
| `SESSION_SECRET_KEY` | Secret for signing session cookies | Recommended |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Azure OpenAI model deployment name | Yes |

### Current Debug Context
- Issue: Semantic Kernel MCPStreamableHttpPlugin reports 0 functions despite successful MCP backend connection
- Enhanced debugging added to `test_mcp_connection()` in `app/agent.py`
- Test endpoint: `/debug/test-mcp`