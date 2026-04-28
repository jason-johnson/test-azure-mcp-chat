@description('Name of the MCP container app')
param name string

@description('Location for resources')
param location string

@description('Tags for resources')
param tags object = {}

@description('Resource ID of the Container Apps environment')
param environmentResourceId string

@description('Docker image (e.g. mcr.microsoft.com/azure-sdk/azure-mcp:latest)')
param dockerImage string

@description('Startup arguments for the container')
param args array

@description('Azure AD Tenant ID')
param azureAdTenantId string

@description('Azure AD Client ID (server app registration)')
param azureAdClientId string

@description('Azure AD login endpoint')
param azureAdInstance string

@description('Resource ID of the user-assigned managed identity for OBO')
param userAssignedManagedIdentityId string

@description('Client ID of the user-assigned managed identity for OBO')
param userAssignedManagedIdentityClientId string

@description('FIC token exchange audience URI')
param tokenExchangeAudience string = 'api://AzureADTokenExchange'

@description('Application Insights connection string')
param appInsightsConnectionString string = ''

resource mcpApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userAssignedManagedIdentityId}': {}
    }
  }
  properties: {
    managedEnvironmentId: environmentResourceId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8080
        transport: 'http'
        allowInsecure: false
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
      }
    }
    template: {
      containers: [
        {
          image: dockerImage
          name: 'mcp-server'
          command: []
          args: args
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          env: concat([
              { name: 'ASPNETCORE_ENVIRONMENT', value: 'Production' }
              { name: 'ASPNETCORE_URLS', value: 'http://+:8080' }
              { name: 'AZURE_MCP_COLLECT_TELEMETRY', value: !empty(appInsightsConnectionString) ? 'true' : 'false' }
              { name: 'AzureAd__Instance', value: azureAdInstance }
              { name: 'AzureAd__TenantId', value: azureAdTenantId }
              { name: 'AzureAd__ClientId', value: azureAdClientId }
              { name: 'AzureAd__ClientCredentials__0__SourceType', value: 'SignedAssertionFromManagedIdentity' }
              { name: 'AzureAd__ClientCredentials__0__ManagedIdentityClientId', value: userAssignedManagedIdentityClientId }
              { name: 'AzureAd__ClientCredentials__0__TokenExchangeUrl', value: tokenExchangeAudience }
              { name: 'AZURE_LOG_LEVEL', value: 'Verbose' }
              { name: 'AZURE_MCP_DANGEROUSLY_DISABLE_HTTPS_REDIRECTION', value: 'true' }
              { name: 'AZURE_MCP_DANGEROUSLY_ENABLE_FORWARDED_HEADERS', value: 'true' }
            ], !empty(appInsightsConnectionString) ? [
              { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsightsConnectionString }
            ] : [])
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaler'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
    }
  }
}

output name string = mcpApp.name
output fqdn string = mcpApp.properties.configuration.ingress.fqdn
output id string = mcpApp.id
