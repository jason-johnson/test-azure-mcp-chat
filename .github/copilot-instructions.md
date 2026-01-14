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

### Testing the Application

**Always use authenticated requests when testing the chat and test-mcp endpoint:**

```bash
ACCESS_TOKEN=$(az account get-access-token --resource "api://f29dd31f-0747-46d7-8cbc-c846e315eb88" --query accessToken -o tsv) && curl -X POST http://localhost:8000/chat \
  -H "x-ms-token-aad-access-token: $ACCESS_TOKEN" \
  -H "x-ms-client-principal-id: test-user-789" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "user_input=how many linux web apps do I have in subscription 2daa7beb-ac1d-473e-84e2-f3cd40e584de?"
```

**Key points:**
- The `x-ms-token-aad-access-token` header is required for Azure MCP Server authentication (OBO flow)
- The `api://f29dd31f-0747-46d7-8cbc-c846e315eb88` resource must match the MCP server's App Registration client ID
- The `x-ms-client-principal-id` header identifies the user
- Replace `user_input` with your test query
- For deployed app, replace `localhost:8000` with the App Service URL

### Current Debug Context
- Issue: Semantic Kernel MCPStreamableHttpPlugin reports 0 functions despite successful MCP backend connection
- Enhanced debugging added to `test_mcp_connection()` in `app/agent.py`
- Test endpoint: `/debug/test-mcp`