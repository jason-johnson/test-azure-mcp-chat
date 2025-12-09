locals {
  mcp_app_name = provider::namep::namestring("azurerm_linux_web_app", local.namep_config, { name = "mcp" })
}
# User Managed Identity for MCP app
resource "azurerm_user_assigned_identity" "mcp_app" {
  name                = provider::namep::namestring("azurerm_user_assigned_identity", local.namep_config, { name = "mcp" })
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
}

# Docker Web App for MCP server
resource "azurerm_linux_web_app" "mcp_app" {
  name                = local.mcp_app_name
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

    app_command_line = "--transport http --outgoing-auth-strategy UseOnBehalfOf --mode all --read-only"

    always_on = false
  }

  auth_settings_v2 {
    auth_enabled           = true
    default_provider       = "azureactivedirectory"
    require_authentication = true

    active_directory_v2 {
      client_id                  = azuread_application.mcp.client_id
      client_secret_setting_name = "MICROSOFT_PROVIDER_AUTHENTICATION_SECRET"
      tenant_auth_endpoint       = "https://sts.windows.net/${data.azuread_client_config.current.tenant_id}/v2.0"
    }

    login {
      token_store_enabled = true
    }
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
    "WEBSITE_AUTH_AAD_ALLOWED_TENANTS"                = data.azuread_client_config.current.tenant_id
    "MICROSOFT_PROVIDER_AUTHENTICATION_SECRET"        = "@Microsoft.KeyVault(SecretUri=${azurerm_key_vault_secret.mcp_secret.id})"
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

resource "random_uuid" "fe_user_impersonation_id" {}

resource "azuread_application" "mcp" {
  display_name = provider::namep::namestring("azuread_application", local.namep_config, { name = "mcp" })
  owners       = [data.azuread_client_config.current.object_id]

  api {
    mapped_claims_enabled = true

    oauth2_permission_scope {
      admin_consent_description  = "Allow the application to access example on behalf of the signed-in user."
      admin_consent_display_name = "User Impersonation"
      enabled                    = true
      id                         = random_uuid.fe_user_impersonation_id.result
      type                       = "User"
      user_consent_description   = "Allow the application to access example on your behalf."
      user_consent_display_name  = "Access MCP backend"
      value                      = "user_impersonation"
    }
  }

  app_role {
    allowed_member_types = ["User", "Application"]
    description          = "Admins can manage roles and perform all task actions"
    display_name         = "Admin"
    enabled              = true
    id                   = "1b19509b-32b1-4e9f-b71d-4992aa991967"
    value                = "admin"
  }

  app_role {
    allowed_member_types = ["User"]
    description          = "ReadOnly roles have limited query access"
    display_name         = "ReadOnly"
    enabled              = true
    id                   = "497406e4-012a-4267-bf18-45a1cb148a01"
    value                = "User"
  }

  feature_tags {
    enterprise = true
    gallery    = true
  }

  required_resource_access {
    resource_app_id = "00000003-0000-0000-c000-000000000000" # Microsoft Graph

    resource_access {
      id   = "df021288-bdef-4463-88db-98f22de89214" # User.Read.All
      type = "Role"
    }

    resource_access {
      id   = "b4e74841-8e56-480b-be8b-910348b18b4c" # User.ReadWrite
      type = "Scope"
    }
  }

  web {
    redirect_uris = ["https://${local.mcp_app_name}.azurewebsites.net/.auth/login/aad/callback"]

    implicit_grant {
      id_token_issuance_enabled = true
    }
  }

  lifecycle {
    ignore_changes = [
      # This parameter is managed by `azuread_application_identifier_uri`.
      # Details: https://github.com/hashicorp/terraform-provider-azuread/issues/428#issuecomment-1788737766
      identifier_uris,
    ]
  }
}

/* resource "azuread_application_pre_authorized" "mcp" {
  application_id       = azuread_application.mcp.id
  authorized_client_id = azuread_application.frontend.client_id

  permission_ids = [
    random_uuid.fe_user_impersonation_id.result,
  ]
} */

resource "azuread_application_identifier_uri" "mcp" {
  application_id = azuread_application.mcp.id
  identifier_uri = "api://${azuread_application.mcp.client_id}"
  depends_on     = [azuread_service_principal.mcp]
}

resource "azuread_application_password" "mcp" {
  application_id = azuread_application.mcp.id
  rotate_when_changed = {
    rotation = time_rotating.main.id
  }
}

resource "azurerm_key_vault_secret" "mcp_secret" {
  name         = "mcp-entra-app-secret"
  key_vault_id = azurerm_key_vault.main.id
  value        = azuread_application_password.mcp.value

  depends_on = [azurerm_role_assignment.managed_admin, azurerm_role_assignment.managed_secrets]
}

resource "azuread_service_principal" "mcp" {
  client_id = azuread_application.mcp.client_id
  owners    = [data.azuread_client_config.current.object_id]
}
