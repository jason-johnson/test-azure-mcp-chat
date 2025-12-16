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

### Current Debug Context
- Issue: Semantic Kernel MCPStreamableHttpPlugin reports 0 functions despite successful MCP backend connection
- Enhanced debugging added to `test_mcp_connection()` in `app/agent.py`
- Test endpoint: `/debug/test-mcp`