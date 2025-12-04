# User Managed Identity for MCP app
resource "azurerm_user_assigned_identity" "mcp_app" {
  name                = provider::namep::namestring("azurerm_user_assigned_identity", local.namep_config, { name = "mcp" })
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
}

# Docker Web App for MCP server
resource "azurerm_linux_web_app" "mcp_app" {
  name                = provider::namep::namestring("azurerm_linux_web_app", local.namep_config, { name = "mcp" })
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id

  identity {
    type         = "SystemAssigned, UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.mcp_app.id]
  }

  site_config {
    application_stack {
      docker_image_name   = "azure-sdk/azure-mcp:2.0.0-beta.7"
      docker_registry_url = "https://mcr.microsoft.com"
    }

    app_command_line = "--transport http --outgoing-auth-strategy UseHostingEnvironmentIdentity --mode all --read-only"

    always_on = false
  }

  app_settings = merge({
    "WEBSITES_ENABLE_APP_SERVICE_STORAGE"             = "false"
    "ASPNETCORE_ENVIRONMENT"                          = "Production"
    "ASPNETCORE_URLS"                                 = "http://+:8080"
    "AZURE_TOKEN_CREDENTIALS"                         = "managedidentitycredential"
    "AZURE_MCP_INCLUDE_PRODUCTION_CREDENTIALS"        = "true"
    "AZURE_MCP_COLLECT_TELEMETRY"                     = "true"
    "AzureAd__Instance"                               = "https://login.microsoftonline.com/"
    "AzureAd__TenantId"                               = data.azuread_client_config.current.tenant_id
    "AzureAd__ClientId"                               = azurerm_user_assigned_identity.mcp_app.client_id
    "AZURE_LOG_LEVEL"                                 = "Verbose"
    "AZURE_MCP_DANGEROUSLY_DISABLE_HTTPS_REDIRECTION" = "true"
    "WEBSITES_PORT"                                   = "8080"
  }, local.app_insights_app_settings)

  logs {
    detailed_error_messages = true
    failed_request_tracing  = true

    application_logs {
      file_system_level = "Information"
    }

    http_logs {
      file_system {
        retention_in_days = 7
        retention_in_mb   = 35
      }
    }
  }

  lifecycle {
    ignore_changes = [
      tags["hidden-link: /app-insights-resource-id"]
    ]
  }
}
