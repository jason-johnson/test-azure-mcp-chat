# Instructions for coding agents

This repository is a Semantic Kernel agent application that connects to remote azure-mcp server, designed to run on azure app service.

## ⚠️ CRITICAL: Azure DevOps (ADO) Pipeline Deployment Only

**DO NOT run local deployment scripts like `redeploy-app.sh` - they are not used for deployment!**
**DO NOT run git commands like `git add`, `git commit`, or `git push` - user handles git workflow!**
**DO NOT run Terraform commands (`terraform init`, `plan`, `apply`, `validate`, etc.) - state is managed remotely via ADO pipeline!**

### Proper Deployment Process
1. Make code changes in `/app/` or `/funapp/` directory
2. User commits: `git add . && git commit -m "message"`
3. User pushes: `git push origin main`
4. Monitor ADO pipeline (auto-triggers on push)
5. Verify via logs: `az webapp log tail --name $(az webapp list --resource-group rg-mcpchat-dev-chn-main --query "[?contains(name, 'fe')].name" -o tsv) --resource-group rg-mcpchat-dev-chn-main`

**⚠️ IMPORTANT: Never use `az webapp log download` - the zip file will be huge and cause problems. Use `az webapp log tail` or other non-download methods to check logs.**

### Terraform State Management
- Terraform state is stored in a **remote Azure Storage Account** (configured in ADO pipeline)
- Running `terraform` commands locally will fail or corrupt state
- Infrastructure changes: Edit files in `/infrastructure/`, commit, push, let ADO pipeline apply

---

## 🔐 Security: Passwordless Authentication Only

**ALWAYS use passwordless/secretless authentication. Never use passwords or connection strings unless there is absolutely no alternative.**

### Preferred Authentication Methods (in order):
1. **Managed Identity** - For Azure-to-Azure communication (App Service → Storage, Function App → OpenAI, etc.)
2. **Workload Identity Federation** - For CI/CD pipelines and external services
3. **DefaultAzureCredential** - In code, automatically uses MI in Azure, CLI locally
4. **User-delegated tokens** - For user-context operations (OBO flow)

### Examples in This Repo:
- Storage Account: `storage_uses_managed_identity = true`, `shared_access_key_enabled = false`
- Function App → Storage: Role assignments (Storage Blob/Queue/Table Data Contributor)
- Function App → AI Services: Role assignment (Cognitive Services OpenAI User)
- AI Foundry connections: `authType = "AAD"` (managed identity)

### When Secrets Are Unavoidable:
- Store in **Azure Key Vault** with Key Vault reference: `@Microsoft.KeyVault(SecretUri=...)`
- Grant access via managed identity, not access policies with secrets
- Example: `MICROSOFT_PROVIDER_AUTHENTICATION_SECRET` for MSAL client secret (required for OBO flow)

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
   - Client ID: `MSAL_CLIENT_ID` env var (NOT `AZURE_CLIENT_ID` to avoid conflict with DefaultAzureCredential)
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
export MSAL_CLIENT_ID="your-frontend-client-id"  # Note: NOT AZURE_CLIENT_ID to avoid conflict with DefaultAzureCredential
export MICROSOFT_PROVIDER_AUTHENTICATION_SECRET="your-client-secret"
export TENANT_ID="your-tenant-id"
export MCP_API_CLIENT_ID="your-mcp-api-client-id"
export APP_BASE_URL="http://localhost:8000"  # or production URL
export SESSION_SECRET_KEY="a-random-secret-key"  # optional, generates random if not set
```

Then visit `http://localhost:8000` in a browser - you'll be redirected to Microsoft login.

---

### Key Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEV_MODE` | Set to `true` for bearer token auth mode | No (default: false) |
| `MSAL_CLIENT_ID` | Frontend app registration client ID (for MSAL auth) | Yes |
| `MICROSOFT_PROVIDER_AUTHENTICATION_SECRET` | Frontend app client secret | Yes (production) |
| `TENANT_ID` | Azure AD tenant ID | Yes |
| `MCP_API_CLIENT_ID` | MCP server app registration client ID | Yes |
| `MCP_URL` | URL of the MCP server | Yes |
| `APP_BASE_URL` | Base URL of this app (for redirect URI) | Yes (production) |
| `SESSION_SECRET_KEY` | Secret for signing session cookies | Recommended |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Azure OpenAI model deployment name | Yes |

**Note:** We use `MSAL_CLIENT_ID` instead of `AZURE_CLIENT_ID` because `DefaultAzureCredential` uses `AZURE_CLIENT_ID` for service principal/managed identity auth, which would conflict with our MSAL browser auth client ID.

### Current Debug Context
- Issue: Semantic Kernel MCPStreamableHttpPlugin reports 0 functions despite successful MCP backend connection
- Enhanced debugging added to `test_mcp_connection()` in `app/agent.py`
- Test endpoint: `/debug/test-mcp`