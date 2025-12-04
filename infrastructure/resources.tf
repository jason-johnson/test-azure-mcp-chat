resource "azurerm_resource_group" "main" {
  name     = provider::namep::namestring("azurerm_resource_group", local.namep_config)
  location = var.location
}

resource "azurerm_service_plan" "main" {
  name                = provider::namep::namestring("azurerm_service_plan", local.namep_config)
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "B1"
}

resource "time_rotating" "main" {
  rotation_days = 60
}