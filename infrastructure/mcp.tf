# Docker Web App for MCP server
resource "azurerm_linux_web_app" "mcp_app" {
  name                = provider::namep::namestring("azurerm_linux_web_app", local.namep_config, { name = "mcp" })
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id

  identity {
    type = "SystemAssigned"
  }

  site_config {
    application_stack {
      docker_image_name   = "mcr.microsoft.com/azure-sdk/azure-mcp:2.0.0-beta.7"
      docker_registry_url = "https://mcr.microsoft.com"
    }

    always_on = false
  }

  app_settings = {
    "APPLICATIONINSIGHTS_CONNECTION_STRING" = azurerm_application_insights.main.connection_string
    "WEBSITES_ENABLE_APP_SERVICE_STORAGE"   = "false"
  }

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
}