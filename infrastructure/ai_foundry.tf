# =============================================================================
# Azure AI Foundry Infrastructure
# =============================================================================
# This creates the AI Hub, Project, and AI Services for Microsoft Agent Framework
# Uses azapi provider for Hub/Project as azurerm doesn't have native support yet

locals {
  ai_services_name = provider::namep::namestring("azurerm_cognitive_account", local.namep_config, { name = "aisvc" })
  ai_hub_name      = provider::namep::namestring("azurerm_cognitive_account", local.namep_config, { name = "aihub" })
  ai_project_name  = provider::namep::namestring("azurerm_cognitive_account", local.namep_config, { name = "aiproj" })
}

# =============================================================================
# Storage Account for AI Hub/Durable Functions
# =============================================================================
resource "azurerm_storage_account" "functions" {
  name                            = provider::namep::namestring("azurerm_storage_account", local.namep_config, { name = "func" })
  resource_group_name             = azurerm_resource_group.main.name
  location                        = azurerm_resource_group.main.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false # Disabled per Azure Policy - use managed identity instead

  tags = {
    purpose = "ai-foundry-and-functions"
  }
}

# =============================================================================
# AI Services Account (replaces standalone OpenAI)
# =============================================================================
resource "azurerm_cognitive_account" "ai_services" {
  name                  = local.ai_services_name
  location              = azurerm_resource_group.main.location
  resource_group_name   = azurerm_resource_group.main.name
  kind                  = "AIServices"
  sku_name              = "S0"
  custom_subdomain_name = local.ai_services_name

  identity {
    type = "SystemAssigned"
  }

  tags = {
    purpose = "agent-framework"
  }
}

# =============================================================================
# AI Hub (Foundry Hub) - Using azapi
# =============================================================================
resource "azapi_resource" "ai_hub" {
  type      = "Microsoft.MachineLearningServices/workspaces@2024-10-01"
  name      = local.ai_hub_name
  location  = azurerm_resource_group.main.location
  parent_id = azurerm_resource_group.main.id

  identity {
    type = "SystemAssigned"
  }

  body = {
    kind = "Hub"
    properties = {
      friendlyName        = "AI Foundry Hub"
      description         = "AI Hub for Microsoft Agent Framework"
      storageAccount      = azurerm_storage_account.functions.id
      keyVault            = azurerm_key_vault.main.id
      applicationInsights = azurerm_application_insights.main.id
      publicNetworkAccess = "Enabled"
      v1LegacyMode        = false
      managedNetwork = {
        isolationMode = "Disabled"
      }
    }
  }

  response_export_values = ["properties.workspaceId", "properties.discoveryUrl"]

  tags = {
    purpose = "agent-framework"
  }
}

# =============================================================================
# AI Project (within the Hub) - Using azapi
# =============================================================================
resource "azapi_resource" "ai_project" {
  type      = "Microsoft.MachineLearningServices/workspaces@2024-10-01"
  name      = local.ai_project_name
  location  = azurerm_resource_group.main.location
  parent_id = azurerm_resource_group.main.id

  identity {
    type = "SystemAssigned"
  }

  body = {
    kind = "Project"
    properties = {
      friendlyName        = "Agent Framework Project"
      description         = "Project for Microsoft Agent Framework agents"
      hubResourceId       = azapi_resource.ai_hub.id
      publicNetworkAccess = "Enabled"
    }
  }

  response_export_values = ["properties.workspaceId", "properties.discoveryUrl"]

  depends_on = [azapi_resource.ai_hub]

  tags = {
    purpose = "agent-framework"
  }
}

# =============================================================================
# Connect AI Services to the Hub
# =============================================================================
resource "azapi_resource" "ai_services_connection" {
  type      = "Microsoft.MachineLearningServices/workspaces/connections@2024-10-01"
  name      = "AzureAIServices"
  parent_id = azapi_resource.ai_hub.id

  body = {
    properties = {
      category      = "AIServices"
      target        = azurerm_cognitive_account.ai_services.endpoint
      authType      = "AAD"
      isSharedToAll = true
      metadata = {
        ApiType    = "Azure"
        ResourceId = azurerm_cognitive_account.ai_services.id
      }
    }
  }

  depends_on = [azapi_resource.ai_hub, azurerm_cognitive_account.ai_services]
}

# =============================================================================
# Model Deployment (GPT-4o for Agent Framework)
# =============================================================================
resource "azurerm_cognitive_deployment" "gpt4o" {
  name                 = "gpt-4o"
  cognitive_account_id = azurerm_cognitive_account.ai_services.id

  model {
    format  = "OpenAI"
    name    = "gpt-4o"
    version = "2024-11-20"
  }

  sku {
    name     = "Standard"
    capacity = 100 # 100K TPM - needed for MCP tools
  }
}
# =============================================================================
# Function App Service Plan (Consumption)
# =============================================================================
resource "azurerm_service_plan" "functions" {
  name                = provider::namep::namestring("azurerm_service_plan", local.namep_config, { name = "func" })
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "Y1" # Consumption plan (or "EP1" for Premium)
}

# =============================================================================
# Function App (Agent Framework host)
# =============================================================================
resource "azurerm_linux_function_app" "agent" {
  name                          = provider::namep::namestring("azurerm_linux_web_app", local.namep_config, { name = "agent" })
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  service_plan_id               = azurerm_service_plan.functions.id
  storage_account_name          = azurerm_storage_account.functions.name
  storage_uses_managed_identity = true # Use managed identity instead of access keys

  site_config {
    application_stack {
      python_version = "3.11"
    }
    application_insights_connection_string = azurerm_application_insights.main.connection_string
  }

  app_settings = {
    # Azure Functions settings
    "FUNCTIONS_WORKER_RUNTIME" = "python"
    "AzureWebJobsFeatureFlags" = "EnableWorkerIndexing"

    # Storage connection via managed identity (no access keys)
    "AzureWebJobsStorage__accountName" = azurerm_storage_account.functions.name

    # Azure AI Foundry (for Agent Framework)
    # Construct endpoint from AI Services subdomain and project name
    "AZURE_AI_PROJECT_ENDPOINT"    = "https://${azurerm_cognitive_account.ai_services.custom_subdomain_name}.services.ai.azure.com/api/projects/${local.ai_project_name}"
    "AZURE_OPENAI_DEPLOYMENT_NAME" = azurerm_cognitive_deployment.gpt4o.name

    # MCP Server connection
    "MCP_URL"           = "https://${azurerm_linux_web_app.mcp_app.default_hostname}"
    "MCP_API_CLIENT_ID" = azuread_application.mcp.client_id

    # Auth (for OBO flow if implementing custom auth endpoints)
    "TENANT_ID"      = data.azuread_client_config.current.tenant_id
    "MSAL_CLIENT_ID" = azuread_application.fe.client_id
  }

  identity {
    type = "SystemAssigned"
  }

  tags = {
    purpose = "agent-framework"
  }
}

# =============================================================================
# Role Assignments
# =============================================================================

# Function App → AI Services (Cognitive Services User)
resource "azurerm_role_assignment" "function_ai_user" {
  scope                = azurerm_cognitive_account.ai_services.id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = azurerm_linux_function_app.agent.identity[0].principal_id
}

# Function App → AI Hub (Azure AI Developer role for agent operations)
resource "azurerm_role_assignment" "function_hub_developer" {
  scope                = azapi_resource.ai_hub.id
  role_definition_name = "Azure AI Developer"
  principal_id         = azurerm_linux_function_app.agent.identity[0].principal_id
}

# Function App → AI Project (Azure AI Developer role)
resource "azurerm_role_assignment" "function_project_developer" {
  scope                = azapi_resource.ai_project.id
  role_definition_name = "Azure AI Developer"
  principal_id         = azurerm_linux_function_app.agent.identity[0].principal_id
}

# Function App → Storage (for Durable Functions state)
resource "azurerm_role_assignment" "function_storage_blob" {
  scope                = azurerm_storage_account.functions.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_linux_function_app.agent.identity[0].principal_id
}

# Function App → Storage Queue (required for Durable Functions)
resource "azurerm_role_assignment" "function_storage_queue" {
  scope                = azurerm_storage_account.functions.id
  role_definition_name = "Storage Queue Data Contributor"
  principal_id         = azurerm_linux_function_app.agent.identity[0].principal_id
}

# Function App → Storage Table (required for Durable Functions)
resource "azurerm_role_assignment" "function_storage_table" {
  scope                = azurerm_storage_account.functions.id
  role_definition_name = "Storage Table Data Contributor"
  principal_id         = azurerm_linux_function_app.agent.identity[0].principal_id
}

# Function App → Storage Account Contributor (required for function runtime)
resource "azurerm_role_assignment" "function_storage_account" {
  scope                = azurerm_storage_account.functions.id
  role_definition_name = "Storage Account Contributor"
  principal_id         = azurerm_linux_function_app.agent.identity[0].principal_id
}

# AI Hub → Storage (required for hub operations)
resource "azurerm_role_assignment" "ai_hub_storage" {
  scope                = azurerm_storage_account.functions.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azapi_resource.ai_hub.identity[0].principal_id
}

# AI Hub → Key Vault (required for hub secrets)
resource "azurerm_role_assignment" "ai_hub_keyvault" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azapi_resource.ai_hub.identity[0].principal_id
}

# AI Hub → AI Services (required for model access)
resource "azurerm_role_assignment" "ai_hub_cognitive" {
  scope                = azurerm_cognitive_account.ai_services.id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = azapi_resource.ai_hub.identity[0].principal_id
}

# =============================================================================
# Outputs
# =============================================================================
output "ai_project_endpoint" {
  description = "Azure AI Foundry project endpoint for Agent Framework"
  value       = "https://${azurerm_cognitive_account.ai_services.custom_subdomain_name}.services.ai.azure.com/api/projects/${local.ai_project_name}"
}

output "ai_hub_name" {
  description = "AI Foundry Hub name"
  value       = azapi_resource.ai_hub.name
}

output "ai_project_name" {
  description = "AI Foundry Project name"
  value       = azapi_resource.ai_project.name
}

output "function_app_name" {
  description = "Name of the Agent Function App"
  value       = azurerm_linux_function_app.agent.name
}

output "function_app_url" {
  description = "URL of the Agent Function App"
  value       = "https://${azurerm_linux_function_app.agent.default_hostname}"
}
