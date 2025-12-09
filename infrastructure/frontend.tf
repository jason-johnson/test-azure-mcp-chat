# Python Web App for the agent code

locals {
  frontend_app_name = provider::namep::namestring("azurerm_linux_web_app", local.namep_config, { name = "fe" })
}

resource "azurerm_linux_web_app" "python_app" {
  name                = local.frontend_app_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id

  ftp_publish_basic_authentication_enabled       = false
  webdeploy_publish_basic_authentication_enabled = false

  identity {
    type = "SystemAssigned"
  }

  site_config {
    application_stack {
      python_version = "3.13"
    }

    app_command_line = "gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080 agent:app"
  }

  auth_settings_v2 {
    auth_enabled           = true
    default_provider       = "azureactivedirectory"
    require_authentication = true

    active_directory_v2 {
      client_id                  = azuread_application.fe.client_id
      client_secret_setting_name = "MICROSOFT_PROVIDER_AUTHENTICATION_SECRET"
      tenant_auth_endpoint       = "https://sts.windows.net/${data.azuread_client_config.current.tenant_id}/v2.0"
    }

    login {
      token_store_enabled = true
    }
  }

  app_settings = merge({
    "AZURE_OPENAI_ENDPOINT"                    = azurerm_cognitive_account.openai.endpoint
    "AZURE_OPENAI_DEPLOYMENT_NAME"             = azurerm_cognitive_deployment.model.name
    "MCP_URL"                                  = "http://${azurerm_linux_web_app.mcp_app.default_hostname}"
    "SCM_DO_BUILD_DURING_DEPLOYMENT"           = "true"
    "WEBSITES_PORT"                            = "8080"
    "WEBSITE_AUTH_AAD_ALLOWED_TENANTS"         = data.azuread_client_config.current.tenant_id
    "MICROSOFT_PROVIDER_AUTHENTICATION_SECRET" = "@Microsoft.KeyVault(SecretUri=${azurerm_key_vault_secret.fe_secret.id})"
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

resource "azuread_application" "fe" {
  display_name = provider::namep::namestring("azuread_application", local.namep_config, { name = "fe" })
  owners       = [data.azuread_client_config.current.object_id]

  api {
    mapped_claims_enabled = true
  }

  feature_tags {
    enterprise = true
    gallery    = true
  }

  required_resource_access {
    resource_app_id = azuread_application.mcp.client_id
    resource_access {
      id   = random_uuid.fe_user_impersonation_id.result
      type = "Scope"
    }

    resource_access {
      id   = local.mcp_role_id
      type = "Role"
    }
  }

  web {
    redirect_uris = ["https://${local.frontend_app_name}.azurewebsites.net/.auth/login/aad/callback"]

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

resource "azuread_application_identifier_uri" "fe" {
  application_id = azuread_application.fe.id
  identifier_uri = "api://${azuread_application.fe.client_id}"
  depends_on     = [azuread_service_principal.fe]
}

resource "azuread_application_password" "fe" {
  application_id = azuread_application.fe.id
  rotate_when_changed = {
    rotation = time_rotating.main.id
  }
}

resource "azurerm_key_vault_secret" "fe_secret" {
  name         = "fe-entra-app-secret"
  key_vault_id = azurerm_key_vault.main.id
  value        = azuread_application_password.fe.value

  depends_on = [azurerm_role_assignment.managed_admin, azurerm_role_assignment.managed_secrets]
}

resource "azuread_service_principal" "fe" {
  client_id = azuread_application.fe.client_id
  owners    = [data.azuread_client_config.current.object_id]
}
