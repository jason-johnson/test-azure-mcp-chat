output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "service_plan_name" {
  description = "Name of the app service plan"
  value       = azurerm_service_plan.main.name
}

output "python_web_app_name" {
  description = "Name of the Python web app"
  value       = azurerm_linux_web_app.python_app.name
}

output "python_web_app_url" {
  description = "URL of the Python web app"
  value       = "https://${azurerm_linux_web_app.python_app.default_hostname}"
}

output "mcp_web_app_name" {
  description = "Name of the MCP web app"
  value       = azurerm_linux_web_app.mcp_app.name
}

output "mcp_web_app_url" {
  description = "URL of the MCP web app"
  value       = "https://${azurerm_linux_web_app.mcp_app.default_hostname}"
}

output "openai_endpoint" {
  description = "Azure OpenAI service endpoint"
  value       = azurerm_cognitive_account.openai.endpoint
}

output "openai_deployment_name" {
  description = "Azure OpenAI deployment name"
  value       = azurerm_cognitive_deployment.model.name
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