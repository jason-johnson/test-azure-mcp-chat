output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "service_plan_name" {
  description = "Name of the app service plan"
  value       = azurerm_service_plan.main.name
}

output "web_app_name" {
  description = "Name of the web app"
  value       = azurerm_linux_web_app.app.name
}

output "web_app_url" {
  description = "URL of the web app"
  value       = "https://${azurerm_linux_web_app.app.default_hostname}"
}

output "application_insights_connection_string" {
  description = "Application Insights connection string"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "key_vault_name" {
  description = "Name of the Key Vault"
  value       = azurerm_key_vault.main.name
}