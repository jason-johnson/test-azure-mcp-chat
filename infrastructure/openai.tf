resource "azurerm_cognitive_account" "openai" {
  name                = provider::namep::namestring("azurerm_cognitive_account", local.namep_config, { name = "openai" })
  location            = "East US"
  resource_group_name = azurerm_resource_group.main.name
  kind                = "OpenAI"
  sku_name            = "S0"

  custom_subdomain_name = provider::namep::namestring("azurerm_cognitive_account", local.namep_config, { name = "openai" })

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_cognitive_deployment" "model" {
  name                 = "gpt-4"
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = "gpt-4o"
    version = "2024-11-20"
  }

  sku {
    name = "Standard"
  }
}