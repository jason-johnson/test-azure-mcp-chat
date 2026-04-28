@description('Name of the MCP web app')
param name string

@description('Location for resources')
param location string

@description('Tags for resources')
param tags object = {}

@description('Resource ID of the App Service Plan')
param serverFarmId string

@description('Docker image name including registry path (e.g. azure-sdk/azure-mcp:2.0.0-beta.23)')
param dockerImageName string

@description('Startup command for the container')
param startupCommand string

@description('Client ID of the MCP App Registration in Entra ID')
param mcpAppClientId string

@description('Tenant ID for authentication')
param tenantId string

@description('Allowed audiences for token validation')
param allowedAudiences array

@description('App settings for the web app')
param appSettings array

resource mcpApp 'Microsoft.Web/sites@2023-12-01' = {
  name: name
  location: location
  tags: tags
  kind: 'app,linux,container'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: serverFarmId
    httpsOnly: true
    clientAffinityEnabled: false
    siteConfig: {
      linuxFxVersion: 'DOCKER|mcr.microsoft.com/${dockerImageName}'
      appCommandLine: startupCommand
      alwaysOn: false
      ftpsState: 'Disabled'
      appSettings: appSettings
    }
  }
}

// EasyAuth v2 - token validation only (no client secret needed for API-only bearer token validation)
resource authSettings 'Microsoft.Web/sites/config@2023-12-01' = {
  parent: mcpApp
  name: 'authsettingsV2'
  properties: {
    platform: {
      enabled: true
    }
    globalValidation: {
      requireAuthentication: true
      unauthenticatedClientAction: 'Return401'
    }
    identityProviders: {
      azureActiveDirectory: {
        enabled: true
        registration: {
          clientId: mcpAppClientId
          openIdIssuer: '${environment().authentication.loginEndpoint}${tenantId}/v2.0'
        }
        validation: {
          allowedAudiences: allowedAudiences
        }
      }
    }
    login: {
      tokenStore: {
        enabled: true
      }
    }
  }
}

output name string = mcpApp.name
output defaultHostname string = mcpApp.properties.defaultHostName
output principalId string = mcpApp.identity.principalId
output id string = mcpApp.id
