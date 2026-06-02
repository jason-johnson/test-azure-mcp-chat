---
name: azure-cli
description: Guides the agent to use Azure CLI to answer Azure resource questions.
---

# Azure CLI Skill

You have access to the Azure CLI (`az`) to query and manage Azure resources.

## How to use az CLI

- Use `az` commands to look up real data. Do NOT guess or hallucinate resource names, IDs, or states.
- Prefer read-only commands unless the user has explicitly asked you to change something.
- Always use `--output json` so results are machine-readable and easy to parse.
- If a query spans multiple subscriptions, run the command per subscription or use `--subscription <id>`.
- Run commands directly — **do not ask the user for permission yourself**. The runtime will handle approval prompts automatically.

### Common read-only commands

```bash
# List subscriptions
az account list --output json

# List resource groups in current subscription
az group list --output json

# List all resources in a resource group
az resource list --resource-group <rg> --output json

# List web apps
az webapp list --output json

# Show details of a specific resource
az resource show --ids <resource-id> --output json

# List VMs and their power state
az vm list --show-details --output json

# List storage accounts
az storage account list --output json

# Show diagnostic logs / activity log
az monitor activity-log list --resource-group <rg> --output json
```

## Handling denied commands

- If the runtime denies a command, explain what you would have done and ask if the user wants a different approach.
- Never chain commands (e.g. `az ... | az ...`) in a single shell call; issue them individually so each can be approved.

## Output formatting

- Summarize JSON results in plain language after showing relevant fields.
- For lists of resources, include: name, resource group, location, type, and any status field.
- Highlight anything unusual: stopped VMs, failed deployments, resources in unexpected states.
