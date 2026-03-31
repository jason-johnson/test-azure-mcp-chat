resource "azurerm_log_analytics_workspace" "main" {
  name                       = provider::namep::namestring("azurerm_log_analytics_workspace", local.namep_config)
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  sku                        = "PerGB2018"
  retention_in_days          = 30
  internet_ingestion_enabled = true # Required for AI Foundry portal access
  internet_query_enabled     = true # Required for AI Foundry portal access
}

resource "azurerm_application_insights" "main" {
  name                          = provider::namep::namestring("azurerm_application_insights", local.namep_config)
  location                      = var.location
  resource_group_name           = azurerm_resource_group.main.name
  workspace_id                  = azurerm_log_analytics_workspace.main.id
  application_type              = "web"
  internet_ingestion_enabled    = true # Required for AI Foundry portal access
  internet_query_enabled        = true # Required for AI Foundry portal access
}
