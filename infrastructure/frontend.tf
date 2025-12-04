# Python Web App for the agent code
resource "azurerm_linux_web_app" "python_app" {
  name                = provider::namep::namestring("azurerm_linux_web_app", local.namep_config, { name = "fe" })
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id

  identity {
    type = "SystemAssigned"
  }

  site_config {
    application_stack {
      python_version = "3.13"
    }

    app_command_line = "gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080 agent:app"
  }

  app_settings = merge({
    "AZURE_OPENAI_ENDPOINT"                           = azurerm_cognitive_account.openai.endpoint
    "AZURE_OPENAI_DEPLOYMENT_NAME"                    = azurerm_cognitive_deployment.model.name
    "MCP_URL"                                         = "http://${azurerm_linux_web_app.mcp_app.default_hostname}"
    "SCM_DO_BUILD_DURING_DEPLOYMENT"                  = "true"
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
