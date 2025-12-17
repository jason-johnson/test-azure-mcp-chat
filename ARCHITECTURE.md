# Azure MCP Chat - Architecture Documentation

This document describes the architecture, deployment process, and infrastructure management for the Azure MCP Chat application.

## Table of Contents

- [Overview](#overview)
- [Architecture Diagram](#architecture-diagram)
- [Application Components](#application-components)
- [Infrastructure Components](#infrastructure-components)
- [Authentication Flow](#authentication-flow)
- [Deployment Pipeline (ADO)](#deployment-pipeline-ado)
- [Infrastructure Management (Terraform)](#infrastructure-management-terraform)
- [Local Development](#local-development)

---

## Overview

Azure MCP Chat is a **Semantic Kernel agent application** that provides an AI-powered Azure Service Reliability Engineer (SRE) assistant. It connects to the Azure MCP (Model Context Protocol) server to execute Azure operations and provide intelligent responses about Azure resources.

### Key Technologies

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI (Python 3.13) |
| AI Framework | Semantic Kernel |
| AI Model | Azure OpenAI (GPT-4o) |
| MCP Server | Azure MCP (`mcr.microsoft.com/azure-sdk/azure-mcp:2.0.0-beta.7`) |
| Infrastructure as Code | Terraform |
| CI/CD | Azure DevOps Pipelines |
| Hosting | Azure App Service (Linux) |
| Authentication | Azure AD / Entra ID |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    Azure Cloud                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐           │
│  │   Azure AD /    │     │   Azure Key     │     │  Application    │           │
│  │   Entra ID      │     │     Vault       │     │    Insights     │           │
│  │                 │     │                 │     │                 │           │
│  │  • App Regs     │     │  • Client       │     │  • Logging      │           │
│  │  • OAuth 2.0    │     │    Secrets      │     │  • Metrics      │           │
│  │  • RBAC         │     │                 │     │  • Tracing      │           │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘           │
│           │                       │                       │                     │
│           ▼                       ▼                       ▼                     │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                        Azure App Service Plan (B1)                      │    │
│  │  ┌──────────────────────────┐    ┌──────────────────────────┐         │    │
│  │  │  Frontend Web App (fe)   │    │    MCP Web App (mcp)     │         │    │
│  │  │                          │    │                          │         │    │
│  │  │  • Python 3.13           │───▶│  • Docker Container      │         │    │
│  │  │  • FastAPI + Gunicorn    │    │  • Azure MCP Server      │         │    │
│  │  │  • Semantic Kernel       │    │  • On-Behalf-Of Auth     │         │    │
│  │  │  • Agent Logic           │    │  • Azure Tool Execution  │         │    │
│  │  │                          │    │                          │         │    │
│  │  │  System Assigned MI      │    │  System + User MI        │         │    │
│  │  └──────────────────────────┘    └──────────────────────────┘         │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                    │                              │                             │
│                    ▼                              ▼                             │
│           ┌─────────────────┐           ┌─────────────────┐                    │
│           │  Azure OpenAI   │           │  Azure Resources │                    │
│           │                 │           │  (via MCP Tools) │                    │
│           │  • GPT-4o Model │           │                  │                    │
│           │  • Chat API     │           │  • VMs, Storage  │                    │
│           └─────────────────┘           │  • Networking    │                    │
│                                         │  • Databases     │                    │
│                                         └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ HTTPS
                                    │
                              ┌─────┴─────┐
                              │   User    │
                              │ (Browser) │
                              └───────────┘
```

---

## Application Components

### Frontend Application (`app/`)

The frontend is a **FastAPI-based Python application** that serves as the AI chat interface.

#### Key Files

| File | Purpose |
|------|---------|
| `agent.py` | Main application with FastAPI routes, Semantic Kernel agent, MCP plugin integration |
| `index.html` | Web UI for the chat interface |
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Python project configuration |

#### Core Features

1. **Semantic Kernel Agent**: Uses `ChatCompletionAgent` with Azure OpenAI (GPT-4o)
2. **MCP Integration**: `MCPStreamableHttpPlugin` connects to Azure MCP server
3. **User Session Management**: Per-user agent caching with token-based invalidation
4. **Token Management**: Automatic token refresh via Microsoft identity platform

#### Agent Configuration

```python
# SRE Instructions define the agent's behavior
sre_instructions = """
Role: Azure Service Reliability Engineer (SRE)
- Investigating and resolving incidents and outages
- Responding to monitoring alerts with actionable insights
- Proactively identifying potential issues
- Executing Azure operations directly when requested
"""
```

### MCP Server

The MCP (Model Context Protocol) server is a **containerized Azure service** that provides tools for Azure resource management.

- **Image**: `mcr.microsoft.com/azure-sdk/azure-mcp:2.0.0-beta.7`
- **Transport**: HTTP with Streamable HTTP plugin
- **Authentication**: On-Behalf-Of (OBO) flow for user impersonation
- **Mode**: Read-only by default (`--read-only` flag)

---

## Infrastructure Components

All infrastructure is defined in the `infrastructure/` directory using Terraform.

### Resource Overview

| Resource | File | Description |
|----------|------|-------------|
| Resource Group | `resources.tf` | Main resource group container |
| App Service Plan | `resources.tf` | Linux B1 SKU hosting plan |
| Frontend Web App | `frontend.tf` | Python FastAPI application |
| MCP Web App | `mcp.tf` | Docker container for Azure MCP |
| Azure OpenAI | `openai.tf` | GPT-4o model deployment |
| Key Vault | `keyvault.tf` | Secret management with RBAC |
| Application Insights | `app_insights.tf` | Monitoring and logging |
| Azure AD Apps | `frontend.tf`, `mcp.tf` | OAuth app registrations |

### Naming Convention

The project uses the `namep` Terraform provider for consistent resource naming:

```terraform
# Example: Creates names like "app-mcpchat-dev-chn-fe"
local.frontend_app_name = provider::namep::namestring("azurerm_linux_web_app", local.namep_config, { name = "fe" })
```

---

## Authentication Flow

The application uses **Azure AD authentication** with **On-Behalf-Of (OBO)** flow:

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  User    │────▶│ Frontend App │────▶│   MCP App    │────▶│Azure Resources│
│ Browser  │     │   (agent)    │     │  (tools)     │     │              │
└──────────┘     └──────────────┘     └──────────────┘     └──────────────┘
     │                  │                    │                    │
     │  1. Login        │                    │                    │
     │─────────────────▶│                    │                    │
     │                  │                    │                    │
     │  2. AAD Token    │                    │                    │
     │◀─────────────────│                    │                    │
     │                  │                    │                    │
     │  3. Chat Request │                    │                    │
     │  (with token)    │                    │                    │
     │─────────────────▶│                    │                    │
     │                  │                    │                    │
     │                  │ 4. MCP Call        │                    │
     │                  │ (OBO token)        │                    │
     │                  │───────────────────▶│                    │
     │                  │                    │                    │
     │                  │                    │ 5. Azure API Call  │
     │                  │                    │ (user context)     │
     │                  │                    │───────────────────▶│
     │                  │                    │                    │
     │                  │                    │ 6. Response        │
     │                  │                    │◀───────────────────│
     │                  │                    │                    │
     │                  │ 7. Tool Result     │                    │
     │                  │◀───────────────────│                    │
     │                  │                    │                    │
     │  8. AI Response  │                    │                    │
     │◀─────────────────│                    │                    │
```

### App Registrations

1. **Frontend App (`fe`)**: Authenticates users, requests MCP API scope
2. **MCP App (`mcp`)**: Validates frontend tokens, executes Azure operations as user

---

## Deployment Pipeline (ADO)

### ⚠️ CRITICAL: ADO Pipeline is the Only Deployment Method

**DO NOT use local deployment scripts or manual `az webapp deploy` commands.**

### Pipeline Files

| File | Purpose |
|------|---------|
| `ado-pipelines/main.yml` | Main pipeline entry point |
| `ado-pipelines/terraform.yml` | Terraform deployment template |
| `ado-pipelines/variables.yml` | Pipeline variables |
| `ado-pipelines/destroy.yml` | Infrastructure destruction |

### Deployment Process

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Developer  │────▶│  Git Push   │────▶│ ADO Pipeline│────▶│   Azure     │
│             │     │  to main    │     │  Triggered  │     │  Deployed   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

#### Pipeline Jobs

1. **Terraform Job** (`Infrastructure Deployment`)
   - Installs Terraform v1.14.0
   - Initializes with Azure backend (Storage Account)
   - Plans and applies infrastructure changes
   - Outputs resource names for next job

2. **Deploy Python App Job** (`Deploy Python Application`)
   - Depends on Terraform job completion
   - Sets up Python 3.13
   - Installs dependencies from `requirements.txt`
   - Archives application to ZIP
   - Deploys to Azure Web App

### How to Deploy

```bash
# 1. Make code changes in /app/ directory

# 2. Commit changes
git add .
git commit -m "Your commit message"

# 3. Push to trigger pipeline
git push origin main

# 4. Monitor deployment in Azure DevOps

# 5. Verify via logs (DO NOT use az webapp log download!)
az webapp log tail \
  --name $(az webapp list --resource-group rg-mcpchat-dev-chn-main --query "[?contains(name, 'fe')].name" -o tsv) \
  --resource-group rg-mcpchat-dev-chn-main
```

---

## Infrastructure Management (Terraform)

### State Management

Terraform state is stored in Azure Storage:

| Setting | Value |
|---------|-------|
| Storage Account | `sajasonmcapterraform` |
| Container | `terraform-mcpchat` |
| Resource Group | `rg-manual` |
| State File | `terraform.tfstate` |

### Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `app_name` | `mcpchat` | Application name prefix |
| `environment` | `dev` | Environment (dev/prod) |
| `location` | `eastus` | Azure region |
| `salt` | `""` | Optional naming salt |

### Terraform Commands (via ADO only)

```bash
# These are executed by the ADO pipeline, not manually:
terraform init       # Initialize with Azure backend
terraform plan       # Preview changes
terraform apply      # Apply infrastructure changes
terraform destroy    # Remove all resources (destroy.yml)
```

### Workspace Support

The pipeline supports Terraform workspaces for environment isolation:

```yaml
parameters:
  - name: useWorkspace
    default: false
  - name: salt
    default: ""
```

---

## Local Development

### Prerequisites

- Python 3.13
- Azure CLI (`az login`)
- Docker (for MCP server testing)
- VS Code with Dev Containers extension

### Environment Variables

Create a `.env` file in the `app/` directory:

```bash
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
MCP_URL=http://localhost:5008
```

### Running Locally

```bash
cd app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
gunicorn -w 2 -k uvicorn.workers.UvicornWorker --timeout 600 -b 0.0.0.0:8000 agent:app
```

### Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Health check for App Service |
| `/ping` | Simple liveness check |
| `/alive` | Synchronous availability check |

---

## Monitoring & Debugging

### Application Insights

All applications are connected to Application Insights for:
- Request/response logging
- Error tracking
- Performance metrics
- Distributed tracing

### Log Access

```bash
# Stream live logs (RECOMMENDED)
az webapp log tail --name <app-name> --resource-group <rg-name>

# View recent logs
az webapp log show --name <app-name> --resource-group <rg-name>

# ⚠️ DO NOT use: az webapp log download (creates huge zip files)
```

### Debug Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/debug/test-mcp` | Test MCP server connection |

---

## Security Considerations

1. **Managed Identities**: Both apps use System Assigned Managed Identities
2. **Key Vault Integration**: Secrets stored in Key Vault with RBAC
3. **No Basic Auth**: FTP and WebDeploy basic auth disabled
4. **HTTPS Only**: All traffic encrypted
5. **AAD Authentication**: All endpoints require Azure AD authentication (except health checks)
6. **Principle of Least Privilege**: RBAC roles scoped to specific resources

---

## Repository Structure

```
azure-mcp-chat/
├── .github/
│   └── copilot-instructions.md    # AI coding assistant instructions
├── ado-pipelines/
│   ├── main.yml                   # Main pipeline definition
│   ├── terraform.yml              # Terraform deployment template
│   ├── variables.yml              # Pipeline variables
│   └── destroy.yml                # Infrastructure destruction
├── app/
│   ├── agent.py                   # Main FastAPI application
│   ├── index.html                 # Web UI
│   ├── requirements.txt           # Python dependencies
│   └── pyproject.toml             # Python project config
├── infrastructure/
│   ├── app_insights.tf            # Application Insights
│   ├── frontend.tf                # Frontend web app + AAD
│   ├── keyvault.tf                # Key Vault configuration
│   ├── locals.tf                  # Local variables
│   ├── mcp.tf                     # MCP server + AAD
│   ├── namep.tf                   # Naming convention
│   ├── openai.tf                  # Azure OpenAI
│   ├── outputs.tf                 # Terraform outputs
│   ├── provider.tf                # Provider configuration
│   ├── resources.tf               # Core resources
│   ├── variables.tf               # Input variables
│   └── README.md                  # Infrastructure docs
├── ARCHITECTURE.md                # This file
└── README.md                      # Project readme
```
