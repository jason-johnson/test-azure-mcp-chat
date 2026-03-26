# Azure Durable Functions Migration Plan

## Status: Infrastructure Complete, Testing Next

| Phase | Status |
|-------|--------|
| Phase 1: Foundation | ✅ Complete |
| Phase 2: Agent Config | 🔄 In Progress |
| Phase 3: Auth & Integration | ⏳ Pending |
| Phase 4: Migration | ⏳ Pending |

**Last Updated:** March 26, 2026

---

## Executive Summary

This document outlines the plan to migrate the current FastAPI-based Azure MCP Chat application to **Azure Durable Functions** using the **Microsoft Agent Framework** (the new `agent-framework` Python package, not Semantic Kernel). The migration will provide enhanced scalability, built-in state persistence, and native Azure Durable Functions integration.

---

## Table of Contents

1. [Current Architecture Overview](#current-architecture-overview)
2. [Microsoft Agent Framework vs Semantic Kernel](#microsoft-agent-framework-vs-semantic-kernel)
3. [Target Architecture](#target-architecture)
4. [Endpoints to Migrate](#endpoints-to-migrate)
5. [Agent Framework Design](#agent-framework-design)
6. [Directory Structure](#directory-structure)
7. [Implementation Plan](#implementation-plan)
8. [Dependencies](#dependencies)
9. [Authentication Considerations](#authentication-considerations)
10. [Infrastructure Changes](#infrastructure-changes)
11. [Testing Strategy](#testing-strategy)

---

## Current Architecture Overview

### Technology Stack

| Component | Current Technology |
|-----------|-------------------|
| Runtime | Python 3.13 with FastAPI |
| Hosting | Azure App Service (Linux) |
| AI Framework | Semantic Kernel with `ChatCompletionAgent` |
| MCP Integration | `MCPStreamableHttpPlugin` |
| AI Model | Azure OpenAI (GPT-4o) |
| Authentication | MSAL (browser) / Bearer tokens (DEV_MODE) |
| Session Management | In-memory caching with TTL |

### Current Endpoints (from `agent.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves HTML chat UI |
| `/ping` | GET | Basic health check |
| `/alive` | GET | Synchronous alive check |
| `/health` | GET | Detailed health check with status |
| `/chat` | POST | Main AI chat endpoint |
| `/clear-cache` | GET | Clears user agent caches |
| `/login` | GET | MSAL login initiation |
| `/auth/callback` | GET | OAuth callback handler |
| `/debug/test-auth` | GET | Auth header testing |
| `/debug/test-agent` | GET | Agent creation testing |
| `/debug/test-mcp` | GET | MCP connection testing |

### Current Pain Points

1. **Session State**: In-memory caching doesn't scale across instances
2. **Long-running Operations**: AI responses can timeout on complex queries
3. **Token Management**: Manual token refresh handling
4. **Error Recovery**: No automatic retry or checkpointing for failed operations

---

## Microsoft Agent Framework vs Semantic Kernel

### ⚠️ Important: These Are Different Packages!

| Package | Install Command | Purpose |
|---------|----------------|---------|
| **Microsoft Agent Framework** | `pip install agent-framework --pre` | New framework for durable AI agents |
| **Semantic Kernel** | `pip install semantic-kernel` | General AI orchestration SDK |

### Microsoft Agent Framework (Recommended)

The **Microsoft Agent Framework** is a newer, purpose-built framework for durable AI agents:

```bash
# Install the full framework with Azure Functions support
pip install agent-framework --pre
pip install agent-framework-azurefunctions --pre
```

**Key Benefits:**
- ✅ **Built-in Azure Functions hosting** via `AgentFunctionApp`
- ✅ **Automatic HTTP endpoints**: `/api/agents/{agentName}/run`
- ✅ **Automatic thread/conversation persistence** via `thread_id`
- ✅ **Durable state management** - survives restarts and failures
- ✅ **MCP tool integration** via Foundry Tools
- ✅ **No custom orchestrators needed** - framework handles everything

### Current Implementation (Semantic Kernel)

The current `agent.py` uses Semantic Kernel:

```python
# Current approach - Semantic Kernel (to be replaced)
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin

agent = ChatCompletionAgent(service=chat_completion, kernel=kernel, ...)
# Manual caching, manual state management, etc.
```

### Target Implementation (Microsoft Agent Framework)

```python
# New approach - Microsoft Agent Framework
from agent_framework import Agent
from agent_framework.azure import AgentFunctionApp, AzureOpenAIResponsesClient
from azure.identity import DefaultAzureCredential

# Create agent (simple!)
client = AzureOpenAIResponsesClient(
    project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    credential=DefaultAzureCredential(),
)

sre_agent = client.as_agent(
    name="SREAgent",
    instructions="You are an Azure SRE assistant...",
    tools=[...],  # MCP tools or custom functions
)

# Host on Azure Functions (one line!)
app = AgentFunctionApp(agents=[sre_agent])
# Automatically creates: POST /api/agents/SREAgent/run
```

---

## Target Architecture

### Microsoft Agent Framework + Azure Functions Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Azure Cloud                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐           │
│  │   Azure AD /    │     │   Azure Key     │     │  Application    │           │
│  │   Entra ID      │     │     Vault       │     │    Insights     │           │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘           │
│           │                       │                       │                     │
│           ▼                       ▼                       ▼                     │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │       Azure Function App (with AgentFunctionApp)                        │    │
│  │                                                                         │    │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │    │
│  │  │  AgentFunctionApp (auto-generated)                                │  │    │
│  │  │                                                                    │  │    │
│  │  │  POST /api/agents/SREAgent/run  ← Main chat endpoint              │  │    │
│  │  │  GET  /api/health               ← Health check                    │  │    │
│  │  │                                                                    │  │    │
│  │  │  Features (handled automatically):                                 │  │    │
│  │  │  • Thread persistence via thread_id                               │  │    │
│  │  │  • Conversation history replay                                    │  │    │
│  │  │  • Failure recovery & retry                                       │  │    │
│  │  │  • Async mode (202 Accepted)                                      │  │    │
│  │  └──────────────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                    │                                                            │
│                    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │          Azure AI Foundry Project                                    │       │
│  │  ┌──────────────────┐    ┌──────────────────┐                       │       │
│  │  │   Azure OpenAI   │    │   MCP Tools via  │                       │       │
│  │  │   (GPT-4o)       │    │   Foundry Tools  │                       │       │
│  │  └──────────────────┘    └──────────────────┘                       │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│                                         │                                       │
│                                         ▼                                       │
│                               ┌─────────────────┐                               │
│                               │    MCP Server   │                               │
│                               │  (Azure MCP)    │                               │
│                               └────────┬────────┘                               │
│                                        │                                        │
│                                        ▼                                        │
│                               ┌─────────────────┐                               │
│                               │ Azure Resources │                               │
│                               └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Simplifications (Microsoft Agent Framework)

| Concern | Old (Semantic Kernel) | New (Agent Framework) |
|---------|----------------------|----------------------|
| **HTTP Endpoints** | Manual FastAPI routes | ✅ Auto-generated by `AgentFunctionApp` |
| **Chat History** | In-memory `ChatHistoryAgentThread` | ✅ Durable, auto via `thread_id` |
| **State Persistence** | Manual caching with TTL | ✅ Automatic durable entities |
| **Failure Recovery** | None | ✅ Built-in replay & recovery |
| **Async Pattern** | None | ✅ `x-ms-wait-for-response: false` → 202 |
| **Orchestrators/Activities** | Custom Durable Functions code | ✅ **Not needed** - framework handles |

### What `AgentFunctionApp` Does Automatically

When you write:
```python
app = AgentFunctionApp(agents=[sre_agent], enable_health_check=True)
```

The framework automatically:
1. Registers agents as **Durable Entities**
2. Generates HTTP endpoints at `/api/agents/{name}/run`
3. Handles **conversation thread persistence** via `thread_id`
4. Provides **async mode** (202 response) via header
5. Manages **failure recovery** and **history replay**
6. Generates a `/api/health` endpoint

---

## Endpoints to Migrate

### Endpoint Mapping

| Current Endpoint | New Endpoint | Notes |
|-----------------|--------------|-------|
| `POST /chat` | `POST /api/agents/SREAgent/run` | Auto-generated by AgentFunctionApp |
| `GET /` | `GET /api/static/index.html` | Static file or separate SWA |
| `GET /health` | `GET /api/health` | Auto-generated with `enable_health_check=True` |
| `GET /ping` | `GET /api/health` | Merged into health |
| `GET /clear-cache` | Not needed | Durable state handles this |
| `GET /login` | Custom HTTP Trigger | Auth flow |
| `GET /auth/callback` | Custom HTTP Trigger | OAuth callback |

---

## Agent Framework Design

### Complete function_app.py Example

```python
# function_app.py
import os
from typing import Annotated
from pydantic import Field
from azure.identity import DefaultAzureCredential
from agent_framework import tool
from agent_framework.azure import AgentFunctionApp, AzureOpenAIResponsesClient


# Define custom tools (can also integrate MCP tools)
@tool(description="Get Azure resource information")
def get_azure_resources(
    subscription_id: Annotated[str, Field(description="Azure subscription ID")],
    resource_type: Annotated[str, Field(description="Type of resource to query")]
) -> str:
    """Query Azure resources - placeholder for MCP tool integration."""
    # This will be replaced by MCP tool integration
    return f"Querying {resource_type} in subscription {subscription_id}"


# SRE Agent Instructions
SRE_INSTRUCTIONS = """
Role: Azure Service Reliability Engineer (SRE)

You are an expert Azure SRE assistant with direct access to Azure operations through tools.

Capabilities:
- Query and manage Azure resources (subscriptions, resource groups, web apps, AKS, storage)
- Diagnose issues and troubleshoot Azure resources
- Provide operational insights and recommendations

Guidelines:
1. Use appropriate tools based on the user's request
2. For Azure operations, use the provided tools
3. Present results clearly and offer follow-up assistance
"""


def create_app() -> AgentFunctionApp:
    """Create and configure the AgentFunctionApp with SRE Agent."""
    
    # Create Azure OpenAI client
    client = AzureOpenAIResponsesClient(
        project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        credential=DefaultAzureCredential(),
    )
    
    # Create the SRE Agent
    sre_agent = client.as_agent(
        name="SREAgent",
        instructions=SRE_INSTRUCTIONS,
        tools=[get_azure_resources],  # Add MCP tools here
    )
    
    # Create the Function App - this does EVERYTHING!
    app = AgentFunctionApp(
        agents=[sre_agent],
        enable_health_check=True,  # Adds /api/health endpoint
    )
    
    return app


# Initialize the app
app = create_app()
```

### MCP Tool Integration Options

#### Option 1: MCP via Foundry Tools (Azure Native)

```python
# Configure MCP connection in Azure AI Foundry
# Then use UseFoundryTools to load MCP tools

# Set environment variable:
# MCP_TOOL_CONNECTION_ID="AzureMCP"

client = AzureOpenAIResponsesClient(
    project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    credential=DefaultAzureCredential(),
)

# UseFoundryTools automatically loads MCP connections from Foundry
tools_client = client.use_foundry_tools()

sre_agent = tools_client.as_agent(
    name="SREAgent",
    instructions=SRE_INSTRUCTIONS,
)
```

#### Option 2: Direct MCP Server Integration

```python
# For direct MCP server integration (like current Azure MCP setup)
# Tools are configured per-request with bearer token

# Note: This may require custom tool wrapper
from agent_framework import tool

@tool(description="Execute Azure MCP operation")
async def mcp_azure_operation(
    operation: Annotated[str, Field(description="MCP operation to execute")],
    parameters: Annotated[dict, Field(description="Operation parameters")]
) -> str:
    """Wrapper to call Azure MCP server with user's token."""
    # Implementation would call MCP server HTTP endpoint
    # with the user's bearer token for OBO authentication
    pass
```

### Thread (Conversation) Persistence

The framework handles thread persistence automatically:

```bash
# First request - new conversation
curl -X POST http://localhost:7071/api/agents/SREAgent/run \
     -d "How many VMs do I have?"
     
# Response includes thread_id in header
# HTTP/1.1 200 OK
# x-ms-thread-id: abc123-thread-id
# 
# You have 5 VMs in your subscription...

# Second request - continue conversation
curl -X POST "http://localhost:7071/api/agents/SREAgent/run?thread_id=abc123-thread-id" \
     -d "Show me details of the first one"
     
# Response continues previous context
# HTTP/1.1 200 OK
# x-ms-thread-id: abc123-thread-id
#
# VM 'prod-web-1' details:
# - Size: Standard_D4s_v3
# - Location: eastus
# ...
```

### Async Mode (202 Accepted)

For long-running operations:

```bash
# Request with async header
curl -X POST http://localhost:7071/api/agents/SREAgent/run \
     -H "x-ms-wait-for-response: false" \
     -d "Analyze all my Azure resources and provide recommendations"

# Immediate 202 response
# HTTP/1.1 202 Accepted
# {
#   "status": "accepted",
#   "thread_id": "abc123",
#   "correlation_id": "xyz789"
# }

# Poll for completion (using Durable Functions status endpoint)
```

### No Custom Orchestrators/Activities Needed! 🎉

With **Microsoft Agent Framework**, we eliminate all custom Durable Functions code:

| Component | Old Plan (Semantic Kernel) | New Plan (Agent Framework) |
|-----------|---------------------------|---------------------------|
| `chat_orchestrator` | Custom orchestrator code | ✅ Built into framework |
| `process_chat` activity | Custom activity code | ✅ Built into framework |
| `validate_token` activity | Custom activity code | ⚠️ May need for OBO |
| `refresh_token` activity | Custom activity code | ⚠️ May need for OBO |
| Durable Entities | Custom entity code | ✅ Built into framework |

---

## Directory Structure

```
/workspaces/azure-mcp-chat/
├── app/                              # Current FastAPI app (to be deprecated)
│   ├── agent.py
│   ├── auth.py
│   └── ...
├── functions/                        # NEW: Microsoft Agent Framework app
│   ├── function_app.py              # Main app with AgentFunctionApp
│   ├── agents/                       # Agent definitions
│   │   ├── __init__.py
│   │   └── sre_agent.py             # SRE Agent configuration
│   ├── tools/                        # Custom tool definitions
│   │   ├── __init__.py
│   │   ├── azure_tools.py           # Azure resource tools
│   │   └── mcp_wrapper.py           # MCP server wrapper (if needed)
│   ├── auth/                         # Auth triggers (if needed for OBO)
│   │   ├── __init__.py
│   │   └── auth_triggers.py         # Login/callback endpoints
│   ├── host.json                    # Function host configuration
│   ├── local.settings.json          # Local development settings
│   └── requirements.txt             # Python dependencies
├── infrastructure/                   # Terraform (needs updates)
│   ├── functions.tf                 # Function app infrastructure
│   ├── ai_foundry.tf                # Azure AI Foundry project
│   └── ...
└── DURABLE_FUNCTIONS_MIGRATION_PLAN.md
```

**Note:** No orchestrators, activities, or entities folders needed - `AgentFunctionApp` handles everything!

---

## Implementation Plan

### Phase 1: Foundation ✅ COMPLETE

1. **Set up Azure AI Foundry infrastructure** ✅
   - ✅ Created AI Hub via `azapi_resource` (not `azurerm_ai_hub` - doesn't exist)
   - ✅ Configured AI Services account (kind = "AIServices")
   - ✅ Created AI Project linked to Hub
   - ✅ Storage with managed identity (Azure Policy requires no key auth)

2. **Set up Azure Functions project** ✅
   - ✅ Created `functions/` directory
   - ✅ Installed `agent-framework --pre` and `agent-framework-azurefunctions --pre`
   - ✅ Configured `host.json` and `requirements.txt`

3. **Create basic AgentFunctionApp** ✅
   - ✅ Implemented `function_app.py` with `AgentFunctionApp`
   - ⏳ Pending: Test local development with `func start`

4. **Update ADO Pipeline** ✅
   - ✅ Added `DeployFunctionsApp` job to `ado-pipelines/main.yml`
   - ⏳ Pending: Push to trigger deployment

### Phase 2: Agent Configuration (2-3 days) 🔄 IN PROGRESS

5. **Configure SRE Agent**
   - ✅ Basic agent structure created
   - ⏳ Define detailed agent instructions
   - ⏳ Integrate MCP tools (via Foundry or wrapper)
   - ⏳ Test tool calling locally

6. **Test conversation persistence**
   - ⏳ Verify thread IDs persist across requests
   - ⏳ Test conversation resumption via `thread_id`

### Phase 3: Auth & Integration (2-3 days)

7. **Implement OBO auth for MCP**
   - ⏳ Create auth helper for MCP bearer token passthrough
   - ⏳ Test MCP operations with user token

8. **Implement custom auth endpoints (if needed)**
   - ⏳ `/api/login` - OAuth redirect
   - ⏳ `/api/auth/callback` - Token exchange

9. **Update frontend**
   - ⏳ Store `thread_id` from response header
   - ⏳ Handle both sync and 202 async responses

### Phase 4: Migration & Cutover (1-2 days)

10. **Migrate traffic**
    - ⏳ Update DNS/routing to point to Function App
    - ⏳ Deprecate old FastAPI app

11. **Testing & Cutover**
    - E2E tests with real MCP server
    - Gradual traffic migration

**Total Estimated Time: 2 weeks** (down from 4 weeks with custom orchestrators!)

---

## Dependencies

### Python Requirements (`functions/requirements.txt`)

> **✅ DEPLOYED** - These packages are in the actual `functions/requirements.txt`

```txt
# Microsoft Agent Framework (pre-release required as of March 2026)
agent-framework --pre
agent-framework-azurefunctions --pre

# Azure Identity
azure-identity>=1.15.0

# Type validation (Pydantic 2.x)
pydantic>=2.0.0

# Planned additions (not yet added):
# azure-ai-projects>=1.0.0  # For Foundry MCP Tools if using managed discovery
# msal>=1.28.0              # If custom auth endpoints needed
```

**Important:** `agent-framework` still requires `--pre` flag - it's not yet a stable release.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_AI_PROJECT_ENDPOINT` | Azure AI Foundry project endpoint |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Model deployment name (GPT-4o) |
| `MCP_URL` | MCP server URL (for direct integration) |
| `TENANT_ID` | Azure AD tenant ID |
| `MSAL_CLIENT_ID` | Frontend app client ID (for auth endpoints) |
| `MCP_API_CLIENT_ID` | MCP API client ID (for OBO scope) |

### host.json Configuration

```json
{
  "version": "2.0",
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": true,
        "excludedTypes": "Request"
      }
    }
  },
  "extensions": {
    "durableTask": {
      "storageProvider": {
        "type": "AzureStorage"
      }
    }
  },
  "functionTimeout": "00:30:00"
}
```

---

## Authentication Considerations

### Token Flow with AgentFunctionApp

With Microsoft Agent Framework, authentication is simplified:

1. **HTTP Request** includes bearer token in Authorization header
2. **Custom tool** extracts token and passes to MCP server
3. **MCP server** performs OBO authentication with Azure resources

### MCP Integration with User Token

```python
# tools/mcp_wrapper.py
from typing import Annotated
from pydantic import Field
from agent_framework import tool
import httpx
from contextvars import ContextVar

# Store user token in context (set by middleware if needed)
current_user_token: ContextVar[str] = ContextVar("current_user_token")


@tool(description="Execute Azure operation via MCP server")
async def azure_mcp_operation(
    operation: Annotated[str, Field(description="The MCP tool operation to execute")],
    parameters: Annotated[dict, Field(description="Parameters for the operation")]
) -> str:
    """Execute an Azure MCP operation with user's token for OBO auth."""
    token = current_user_token.get(None)
    if not token:
        return "Error: No user token available"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{os.environ['MCP_URL']}/tools/call",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": operation, "arguments": parameters},
            timeout=300
        )
        return response.json()["content"][0]["text"]
```

### Alternative: Use Foundry Tools (Recommended)

If MCP is configured as a Foundry Tool connection, authentication flows automatically:

```python
# No token management needed - Foundry handles it
client = AzureOpenAIResponsesClient(
    project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    credential=DefaultAzureCredential(),
)

# Foundry Tools include MCP connections
tools_client = client.use_foundry_tools()
```

---

## Infrastructure Changes

> **✅ DEPLOYED** - Infrastructure is complete as of March 26, 2026

### Key Deployment Learnings

⚠️ **Important Gotchas We Encountered:**

1. **Storage Account - Azure Policy**: Our subscription blocks key-based authentication
   - Must set `shared_access_key_enabled = false`
   - Provider requires `storage_use_azuread = true`
   - Function App uses `storage_uses_managed_identity = true`

2. **Consumption Plan Limitation**: Cannot create Linux Consumption (Y1) plan in a resource group that already has App Service Plans
   - **Solution**: Reuse the existing `azurerm_service_plan.main` (B1)

3. **AI Hub/Project**: `azurerm_ai_hub` doesn't exist in azurerm provider
   - **Solution**: Use `azapi_resource` with `Microsoft.MachineLearningServices/workspaces@2024-10-01`
   - Hub has `kind = "Hub"`, Project has `kind = "Project"`

4. **AI Hub Role Assignments**: AI Hub automatically creates its own role assignments
   - Don't create role assignments for AI Hub identity manually (causes 409 conflicts)

### Terraform Resources Created (`infrastructure/ai_foundry.tf`)

#### 1. Storage Account (Managed Identity Auth)

```hcl
resource "azurerm_storage_account" "functions" {
  name                            = "..."
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false  # Required by Azure Policy
}
```

#### 2. AI Services Account

```hcl
resource "azurerm_cognitive_account" "ai_services" {
  name                  = local.ai_services_name
  kind                  = "AIServices"  # Multi-service, includes OpenAI
  sku_name              = "S0"
  custom_subdomain_name = local.ai_services_name
}
```

#### 3. AI Hub & Project (via azapi)

```hcl
# AI Hub
resource "azapi_resource" "ai_hub" {
  type      = "Microsoft.MachineLearningServices/workspaces@2024-10-01"
  name      = local.ai_hub_name
  body = {
    kind = "Hub"
    properties = {
      storageAccount      = azurerm_storage_account.functions.id
      keyVault            = azurerm_key_vault.main.id
      applicationInsights = azurerm_application_insights.main.id
    }
  }
}

# AI Project (child of Hub)
resource "azapi_resource" "ai_project" {
  type = "Microsoft.MachineLearningServices/workspaces@2024-10-01"
  body = {
    kind = "Project"
    properties = {
      hubResourceId = azapi_resource.ai_hub.id
    }
  }
}
```

#### 4. Function App (Managed Identity Storage)

```hcl
resource "azurerm_linux_function_app" "agent" {
  service_plan_id               = azurerm_service_plan.main.id  # Reuse existing!
  storage_account_name          = azurerm_storage_account.functions.name
  storage_uses_managed_identity = true  # No access keys

  app_settings = {
    "AzureWebJobsStorage__accountName" = azurerm_storage_account.functions.name
    "AZURE_AI_PROJECT_ENDPOINT"        = "https://..."
    "AZURE_OPENAI_DEPLOYMENT_NAME"     = "gpt-4o"
  }
}
```

#### 5. Required Role Assignments

```hcl
# Function App needs multiple storage roles for Durable Functions
resource "azurerm_role_assignment" "function_storage_blob" {
  role_definition_name = "Storage Blob Data Contributor"
}
resource "azurerm_role_assignment" "function_storage_queue" {
  role_definition_name = "Storage Queue Data Contributor"
}
resource "azurerm_role_assignment" "function_storage_table" {
  role_definition_name = "Storage Table Data Contributor"
}
resource "azurerm_role_assignment" "function_storage_account" {
  role_definition_name = "Storage Account Contributor"
}

# Function App needs AI roles
resource "azurerm_role_assignment" "function_ai_user" {
  role_definition_name = "Cognitive Services OpenAI User"
}
resource "azurerm_role_assignment" "function_hub_developer" {
  role_definition_name = "Azure AI Developer"
}

# NOTE: AI Hub creates its own role assignments automatically - don't duplicate!
```

### Provider Configuration

```hcl
# Required for storage without access keys
provider "azurerm" {
  storage_use_azuread = true
}

# Required for AI Hub/Project
terraform {
  required_providers {
    azapi = {
      source  = "Azure/azapi"
      version = "~> 2.4.0"
    }
  }
}
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_tools.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_azure_mcp_operation():
    """Test MCP tool wrapper."""
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_post.return_value.json.return_value = {
            "content": [{"text": "5 VMs found"}]
        }
        
        from functions.tools.mcp_wrapper import azure_mcp_operation
        from functions.tools.mcp_wrapper import current_user_token
        
        # Set user token
        token = current_user_token.set("mock-token")
        try:
            result = await azure_mcp_operation(
                operation="list_vms",
                parameters={"subscription_id": "sub-123"}
            )
            assert "5 VMs found" in result
        finally:
            current_user_token.reset(token)
```

### Integration Tests

```python
# tests/test_agent_app.py
import pytest
from agent_framework.testing import AgentTestClient


@pytest.mark.asyncio
async def test_sre_agent_conversation():
    """Test the full SRE agent conversation flow."""
    from functions.function_app import app
    
    async with AgentTestClient(app) as client:
        # First message
        response1 = await client.run_agent(
            "SREAgent",
            "How many VMs do I have?"
        )
        assert response1.status_code == 200
        thread_id = response1.headers.get("x-ms-thread-id")
        assert thread_id is not None
        
        # Continue conversation
        response2 = await client.run_agent(
            "SREAgent",
            "Show details of the first one",
            thread_id=thread_id
        )
        assert response2.status_code == 200
        assert response2.headers.get("x-ms-thread-id") == thread_id
```

### Local Development Testing

```bash
# Start functions locally
cd functions
func start

# Test health endpoint
curl http://localhost:7071/api/health

# Test agent endpoint
curl -X POST http://localhost:7071/api/agents/SREAgent/run \
     -d "List my Azure subscriptions"

# Continue conversation (use thread_id from previous response)
curl -X POST "http://localhost:7071/api/agents/SREAgent/run?thread_id=<thread_id>" \
     -d "How many resource groups in the first subscription?"
```

---

## Migration Checklist

### Infrastructure ✅ COMPLETE
- [x] Create Azure AI Foundry project (AI Hub) via Terraform
- [x] Create Function App with storage account
- [x] Configure managed identity RBAC
- [ ] (Optional) Configure MCP as Foundry Tool connection

### Function App ✅ COMPLETE
- [x] Create `functions/` directory
- [x] Install `agent-framework --pre` and `agent-framework-azurefunctions --pre`
- [x] Create `function_app.py` with `AgentFunctionApp`
- [x] Configure `host.json`
- [ ] Configure `local.settings.json` (for local dev)

### Agent & Tools 🔄 IN PROGRESS
- [x] Define SRE Agent with basic instructions
- [ ] Integrate MCP tools (Foundry or wrapper)
- [ ] Test agent locally with `func start`
- [ ] Test conversation persistence (thread_id)

### Auth Integration (if needed)
- [ ] Implement MCP token passthrough
- [ ] (Optional) Add custom auth endpoints

### Testing
- [ ] Unit tests for tools
- [ ] Integration tests with AgentTestClient
- [ ] E2E tests with real MCP server

### Deployment 🔄 IN PROGRESS
- [x] Update ADO pipeline
- [ ] Deploy to staging (push to trigger)
- [ ] Test with real MCP server
- [ ] Cutover from App Service

---

## References

- [Microsoft Agent Framework (PyPI)](https://pypi.org/project/agent-framework/) - Main framework package
- [Microsoft Agent Framework Azure Functions](https://pypi.org/project/agent-framework-azurefunctions/) - Azure Functions integration
- [Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-studio/) - AI project management and tool hosting
- [Azure Durable Functions - Python](https://learn.microsoft.com/en-us/azure/azure-functions/durable/durable-functions-overview?pivots=python)
- [Durable Entities](https://learn.microsoft.com/en-us/azure/azure-functions/durable/durable-functions-entities?pivots=python)

---

## Appendix: Current vs New API Contract

### Chat Endpoint

**Current (FastAPI - Synchronous):**
```bash
POST /chat
Content-Type: application/x-www-form-urlencoded
Authorization: Bearer <token>
x-user-id: <user-id>

user_input=How many VMs do I have?

Response (sync - waits for completion):
{"response": "You have 5 VMs..."}
```

**New (Agent Framework - Auto-generated):**
```bash
# Start new conversation (sync mode - default)
POST /api/agents/SREAgent/run
Content-Type: text/plain
Authorization: Bearer <token>

How many VMs do I have?

Response (200 OK):
You have 5 VMs in your subscription...

# Response headers include:
#   x-ms-thread-id: thread_abc123

# Continue conversation (use thread_id from header)
POST /api/agents/SREAgent/run?thread_id=thread_abc123
Content-Type: text/plain

Show me details of the first one

Response (200 OK):
VM 'prod-web-1' details:
- Size: Standard_D4s_v3
- Location: eastus
...

# Async mode (for long-running operations)
POST /api/agents/SREAgent/run
x-ms-wait-for-response: false
Content-Type: text/plain

Analyze all my Azure resources and provide recommendations

Response (202 Accepted):
{
  "status": "accepted",
  "thread_id": "thread_abc123",
  "correlation_id": "xyz789"
}
```

### Key Differences

| Aspect | Current | New |
|--------|---------|-----|
| Response Pattern | Synchronous | Sync by default, async with header |
| Conversation State | In-memory (lost on restart) | Durable (persists) |
| Resume Conversation | Per-session only | Via `thread_id` (anytime) |
| Timeout Handling | HTTP timeout | Built-in to framework |
| Retry on Failure | Manual | Automatic |
| Endpoint Discovery | Manual definition | Auto-generated |

---

*Document Version: 4.0*  
*Created: March 12, 2026*  
*Last Updated: March 26, 2026*  
*Major Changes:*
- *v3.0: Migrated to Microsoft Agent Framework (agent-framework --pre)*
- *v4.0: Infrastructure deployed, updated with deployment learnings*
