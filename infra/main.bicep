targetScope = 'subscription'

extension microsoftGraphV1

@minLength(1)
@maxLength(64)
@description('Name of the the environment which is used to generate a short unique hash used in all resources.')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
@metadata({
  azd: {
    type: 'location'
  }
})
param location string

@description('Skip the creation of the virtual network and private endpoint')
param skipVnet bool = true

@description('Name of the API service')
param apiServiceName string = ''

@description('Name of the user assigned identity')
param apiUserAssignedIdentityName string = ''

@description('Name of the application insights resource')
param applicationInsightsName string = ''

@description('Name of the app service plan')
param appServicePlanName string = ''

@description('Name of the log analytics workspace')
param logAnalyticsName string = ''

@description('Name of the resource group')
param resourceGroupName string = ''

@description('Name of the storage account')
param storageAccountName string = ''

@description('Name of the virtual network')
param vNetName string = ''

@description('Disable local authentication for Azure Monitor')
param disableLocalAuth bool = true

@description('Name of the web service')
param webServiceName string = ''

@description('Id of the user or app to assign application roles')
param principalId string = deployer().objectId

@description('Name of the Azure AI Services account')
param aiServicesName string = 'agentaiservices'

@description('Model name for deployment')
param modelName string = 'gpt-4.1'

@description('Model format for deployment')
param modelFormat string = 'OpenAI'

@description('Model version for deployment')
param modelVersion string = '2025-04-14'

@description('Model deployment capacity')
param modelCapacity int = 10

@description('Model deployment location. If you want to deploy an Azure AI resource/model in different location than the rest of the resources created.')
param modelLocation string = location

// ============= MCP Server Parameters =============

@description('Azure MCP Docker image tag from mcr.microsoft.com/azure-sdk/azure-mcp')
param mcpImageTag string = 'latest'

@description('Name of the MCP app service')
param mcpServiceName string = ''

// Variables
var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, rg.id, environmentName, location))
var aiResourceToken = toLower(uniqueString(subscription().id, rg.id, environmentName, modelLocation))
var tags = { 'azd-env-name': environmentName }
var functionAppName = !empty(apiServiceName) ? apiServiceName : '${abbrs.webSitesFunctions}api-${resourceToken}'
var deploymentStorageContainerName = 'app-package-${take(functionAppName, 32)}-${take(toLower(uniqueString(functionAppName, resourceToken)), 7)}'
// Define the web app name first so we can construct the URL
var webAppName = !empty(webServiceName) ? webServiceName : '${abbrs.webStaticSites}web-${resourceToken}'
// Pre-compute the expected web URI for CORS settings
var webUri = 'https://${webAppName}.azurestaticapps.net'
var mcpAppName = !empty(mcpServiceName) ? mcpServiceName : '${abbrs.appContainerApps}mcp-${resourceToken}'

// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2025-04-01' = {
  name: !empty(resourceGroupName) ? resourceGroupName : '${abbrs.resourcesResourceGroups}${environmentName}'
  location: location
  tags: tags
}

// The application frontend webapp using AVM
module webapp 'br/public:avm/res/web/static-site:0.9.3' = {
  name: 'webapp-${resourceToken}'
  scope: rg
  params: {
    name: webAppName
    location: 'westus2' // Static Web Apps are global, but needs a specific region if a backend API is ever configured. Using westus2 as it is widely available.
    tags: union(tags, { 'azd-service-name': 'web' })
    sku: 'Standard'
    managedIdentities: {
      userAssignedResourceIds: [apiUserAssignedIdentity.outputs.resourceId]
    }
  }
}

// User assigned managed identity using AVM
module apiUserAssignedIdentity 'br/public:avm/res/managed-identity/user-assigned-identity:0.4.0' = {
  name: 'apiUserAssignedIdentity-${resourceToken}'
  scope: rg
  params: {
    name: !empty(apiUserAssignedIdentityName) ? apiUserAssignedIdentityName : '${abbrs.managedIdentityUserAssignedIdentities}api-${resourceToken}'
    location: location
    tags: tags
  }
}

// Backing storage for Azure functions using AVM
module storage 'br/public:avm/res/storage/storage-account:0.29.0' = {
  scope: rg
  name: 'storage-${resourceToken}'
  params: {
    name: !empty(storageAccountName) ? storageAccountName : '${abbrs.storageStorageAccounts}${resourceToken}'
    location: location
    tags: tags
    kind: 'StorageV2'
    skuName: 'Standard_LRS'
    allowSharedKeyAccess: false
    blobServices: {
      containers: [
        { name: deploymentStorageContainerName }
      ]
    }
    publicNetworkAccess: skipVnet ? 'Enabled' : 'Disabled'
    networkAcls: skipVnet ? {
        defaultAction: 'Allow'
        bypass: 'AzureServices'
      } : {
      defaultAction: 'Deny'
      bypass: 'AzureServices'
    }
    roleAssignments: [
      {
        principalId: apiUserAssignedIdentity.outputs.principalId
        roleDefinitionIdOrName: 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b' // Storage Blob Data Owner
        principalType: 'ServicePrincipal'
      }
      {
        principalId: apiUserAssignedIdentity.outputs.principalId
        roleDefinitionIdOrName: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor
        principalType: 'ServicePrincipal'
      }
      {
        principalId: apiUserAssignedIdentity.outputs.principalId
        roleDefinitionIdOrName: '0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3' // Storage Table Data Contributor
        principalType: 'ServicePrincipal'
      }
      {
        principalId: apiUserAssignedIdentity.outputs.principalId
        roleDefinitionIdOrName: '974c5e8b-45b9-4653-ba55-5f855dd0fb88' // Storage Queue Data Contributor
        principalType: 'ServicePrincipal'
      }
      {
        principalId: principalId
        roleDefinitionIdOrName: 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b' // Storage Blob Data Owner
        principalType: 'User'
      }
      {
        principalId: principalId
        roleDefinitionIdOrName: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor
        principalType: 'User'
      }
      {
        principalId: principalId
        roleDefinitionIdOrName: '0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3' // Storage Table Data Contributor
        principalType: 'User'
      }
      {
        principalId: principalId
        roleDefinitionIdOrName: '974c5e8b-45b9-4653-ba55-5f855dd0fb88' // Storage Queue Data Contributor
        principalType: 'User'
      }
    ]
  }
}

// App Service Plan using AVM - Flex Consumption
module appServicePlan 'br/public:avm/res/web/serverfarm:0.5.0' = {
  name: 'appserviceplan-${resourceToken}'
  scope: rg
  params: {
    name: !empty(appServicePlanName) ? appServicePlanName : '${abbrs.webServerFarms}${resourceToken}'
    location: location
    tags: tags
    skuName: 'FC1' // Flex Consumption
    reserved: true
  }
}

// Function App using AVM - Flex Consumption (Python)
module api 'br/public:avm/res/web/site:0.19.3' = {
  name: 'api-${resourceToken}'
  scope: rg
  params: {
    name: functionAppName
    location: location
    tags: union(tags, { 'azd-service-name': 'api' })
    kind: 'functionapp,linux'
    serverFarmResourceId: appServicePlan.outputs.resourceId
    managedIdentities: {
      userAssignedResourceIds: [apiUserAssignedIdentity.outputs.resourceId]
    }
    functionAppConfig: {
      deployment: {
        storage: {
          type: 'blobContainer'
          value: 'https://${storage.outputs.name}.blob.${environment().suffixes.storage}/${deploymentStorageContainerName}'
          authentication: {
            type: 'UserAssignedIdentity'
            userAssignedIdentityResourceId: apiUserAssignedIdentity.outputs.resourceId
          }
        }
      }
      scaleAndConcurrency: {
        instanceMemoryMB: 2048
        maximumInstanceCount: 100
      }
      runtime: {
        name: 'python'
        version: '3.11'
      }
    }
    virtualNetworkSubnetResourceId: skipVnet ? '' : '${serviceVirtualNetwork!.outputs.resourceId}/subnets/app-subnet'
    diagnosticSettings: [
      {
        workspaceResourceId: logAnalytics.outputs.resourceId
      }
    ]
    siteConfig: {
      alwaysOn: false
      cors: {
        allowedOrigins: [ webUri, 'https://${webapp.outputs.defaultHostname}' ]
      }
      appSettings: [
        { name: 'AzureWebJobsStorage__credential', value: 'managedidentity' }
        { name: 'AzureWebJobsStorage__clientId', value: apiUserAssignedIdentity.outputs.clientId }
        { name: 'AzureWebJobsStorage__blobServiceUri', value: 'https://${storage.outputs.name}.blob.${environment().suffixes.storage}' }
        { name: 'AzureWebJobsStorage__queueServiceUri', value: 'https://${storage.outputs.name}.queue.${environment().suffixes.storage}' }
        { name: 'AzureWebJobsStorage__tableServiceUri', value: 'https://${storage.outputs.name}.table.${environment().suffixes.storage}' }
        { name: 'AzureWebJobsStorage__accountName', value: storage.outputs.name }
        { name: 'FOUNDRY_PROJECT_ENDPOINT', value: foundry.outputs.projectEndpoint }
        { name: 'AZURE_OPENAI_DEPLOYMENT_NAME', value: modelName }
        { name: 'MCP_SERVER_URL', value: 'https://${mcpApp.outputs.fqdn}' }
        { name: 'AZURE_CLIENT_ID', value: apiUserAssignedIdentity.outputs.clientId }
        { name: 'APPLICATIONINSIGHTS_AUTHENTICATION_STRING', value: 'ClientId=${apiUserAssignedIdentity.outputs.clientId};Authorization=AAD' }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: monitoring.outputs.connectionString }
      ]
    }
  }
}

// AI Foundry resource + project
var aiServiceName = '${aiServicesName}${aiResourceToken}'

module foundry './app/foundry.bicep' = {
  scope: rg
  name: 'foundry-${aiResourceToken}'
  params: {
    name: aiServiceName
    projectName: '${environmentName}-project'
    location: modelLocation
    tags: tags
    modelName: modelName
    modelFormat: modelFormat
    modelVersion: modelVersion
    modelCapacity: modelCapacity
  }
}

// Role assignments for the Function App MI on the Foundry resource
module foundryRoleApi 'br/public:avm/ptn/authorization/resource-role-assignment:0.1.2' = {
  scope: rg
  name: 'foundryRoleApi-${resourceToken}'
  params: {
    principalId: apiUserAssignedIdentity.outputs.principalId
    roleDefinitionId: '53ca6127-db72-4b80-b1b0-d745d6d5456d' // Azure AI User
    principalType: 'ServicePrincipal'
    resourceId: foundry.outputs.resourceId
  }
}

module foundryRoleUser 'br/public:avm/ptn/authorization/resource-role-assignment:0.1.2' = {
  scope: rg
  name: 'foundryRoleUser-${resourceToken}'
  params: {
    principalId: principalId
    roleDefinitionId: '53ca6127-db72-4b80-b1b0-d745d6d5456d' // Azure AI User
    principalType: 'User'
    resourceId: foundry.outputs.resourceId
  }
}

// Storage role assignments using AVM pattern
// Blob and table roles now inline in storage module
var storageQueueDataContributorRole = '974c5e8b-45b9-4653-ba55-5f855dd0fb88'

// storageRoleAssignmentApi - now inline in storage module
// storageRoleAssignmentUser - now inline in storage module

module storageQueueRoleApi 'br/public:avm/ptn/authorization/resource-role-assignment:0.1.2' = {
  scope: rg
  name: 'storageQueueApi-${resourceToken}'
  params: {
    principalId: apiUserAssignedIdentity.outputs.principalId
    roleDefinitionId: storageQueueDataContributorRole
    principalType: 'ServicePrincipal'
    resourceId: storage.outputs.resourceId
  }
}

module storageQueueRoleUser 'br/public:avm/ptn/authorization/resource-role-assignment:0.1.2' = {
  scope: rg
  name: 'storageQueueUser-${resourceToken}'
  params: {
    principalId: principalId
    roleDefinitionId: storageQueueDataContributorRole
    principalType: 'User'
    resourceId: storage.outputs.resourceId
  }
}

// storageTableRoleApi - now inline in storage module

// Virtual Network using AVM
var vnetName = !empty(vNetName) ? vNetName : '${abbrs.networkVirtualNetworks}${resourceToken}'

module serviceVirtualNetwork 'br/public:avm/res/network/virtual-network:0.7.1' = if (!skipVnet) {
  scope: rg
  name: 'vnet-${resourceToken}'
  params: {
    name: vnetName
    location: location
    tags: tags
    addressPrefixes: ['10.0.0.0/16']
    subnets: [
      {
        name: 'app-subnet'
        addressPrefix: '10.0.0.0/24'
        delegation: 'Microsoft.App/environments'
      }
      {
        name: 'pe-subnet'
        addressPrefix: '10.0.1.0/24'
      }
    ]
  }
}

// Private DNS Zone for blob storage
module privateDnsZone 'br/public:avm/res/network/private-dns-zone:0.8.0' = if (!skipVnet) {
  scope: rg
  name: 'pdns-${resourceToken}'
  params: {
    name: 'privatelink.blob.${environment().suffixes.storage}'
    virtualNetworkLinks: [
      {
        virtualNetworkResourceId: serviceVirtualNetwork!.outputs.resourceId
      }
    ]
  }
}

// Private Endpoint for storage using AVM
module storagePrivateEndpoint 'br/public:avm/res/network/private-endpoint:0.11.1' = if (!skipVnet) {
  scope: rg
  name: 'pe-${resourceToken}'
  params: {
    name: 'pe-storage-${resourceToken}'
    location: location
    tags: tags
    subnetResourceId: '${serviceVirtualNetwork!.outputs.resourceId}/subnets/pe-subnet'
    privateLinkServiceConnections: [
      {
        name: 'storage-blob-connection'
        properties: {
          privateLinkServiceId: storage.outputs.resourceId
          groupIds: ['blob']
        }
      }
    ]
    privateDnsZoneGroup: {
      privateDnsZoneGroupConfigs: [
        {
          privateDnsZoneResourceId: privateDnsZone!.outputs.resourceId
        }
      ]
    }
  }
}

// Log Analytics Workspace using AVM
module logAnalytics 'br/public:avm/res/operational-insights/workspace:0.13.0' = {
  scope: rg
  name: 'logs-${resourceToken}'
  params: {
    name: !empty(logAnalyticsName) ? logAnalyticsName : '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
    location: location
    tags: tags
  }
}

// Application Insights using AVM
module monitoring 'br/public:avm/res/insights/component:0.7.1' = {
  scope: rg
  name: 'monitoring-${resourceToken}'
  params: {
    name: !empty(applicationInsightsName) ? applicationInsightsName : '${abbrs.insightsComponents}${resourceToken}'
    location: location
    tags: tags
    workspaceResourceId: logAnalytics.outputs.resourceId
    disableLocalAuth: disableLocalAuth
    roleAssignments: [
      {
        principalId: apiUserAssignedIdentity.outputs.principalId
        roleDefinitionIdOrName: '3913510d-42f4-4e42-8a64-420c390055eb' // Monitoring Metrics Publisher
        principalType: 'ServicePrincipal'
      }
    ]
  }
}

// ============= MCP Server Resources =============

// FIC token exchange audience varies by cloud
var tokenExchangeAudience = environment().name == 'AzureUSGovernment'
  ? 'api://AzureADTokenExchangeUSGov'
  : environment().name == 'AzureChinaCloud'
    ? 'api://AzureADTokenExchangeChina'
    : 'api://AzureADTokenExchange'

var mcpScopeId = guid(subscription().id, environmentName, 'Mcp.Tools.ReadWrite')
var mcpServerUniqueName = '${environmentName}-mcp-server'
var mcpClientUniqueName = '${environmentName}-mcp-client'

// User Assigned Managed Identity for OBO federated identity credential
module mcpManagedIdentity 'br/public:avm/res/managed-identity/user-assigned-identity:0.4.0' = {
  scope: rg
  params: {
    name: '${abbrs.managedIdentityUserAssignedIdentities}mcp-${resourceToken}'
    location: location
    tags: tags
  }
}

// Server App Registration — the OAuth 2.0 resource exposed to clients
resource mcpServerApp 'Microsoft.Graph/applications@v1.0' = {
  uniqueName: mcpServerUniqueName
  displayName: '${environmentName} MCP Server'
  signInAudience: 'AzureADMyOrg'
  api: {
    requestedAccessTokenVersion: 2
    oauth2PermissionScopes: [
      {
        id: mcpScopeId
        adminConsentDescription: 'Allow the application to access Azure MCP tools on behalf of the signed-in user.'
        adminConsentDisplayName: 'Azure MCP Tools ReadWrite'
        isEnabled: true
        type: 'User'
        userConsentDescription: 'Allow the application to access Azure MCP tools on your behalf.'
        userConsentDisplayName: 'Access Azure MCP tools'
        value: 'Mcp.Tools.ReadWrite'
      }
    ]
    preAuthorizedApplications: [
      {
        appId: mcpClientApp.appId
        delegatedPermissionIds: [
          mcpScopeId
        ]
      }
    ]
  }
  requiredResourceAccess: [
    {
      // Azure Resource Manager API — user_impersonation for OBO
      resourceAppId: '797f4846-ba00-4fd7-ba43-dac1f8f63013'
      resourceAccess: [
        {
          id: '41094075-9dad-400e-a0bd-54e686782033'
          type: 'Scope'
        }
      ]
    }
  ]
}

// Update server app to add identifierUris (requires appId to be known first)
resource mcpServerAppUpdate 'Microsoft.Graph/applications@v1.0' = {
  uniqueName: mcpServerUniqueName
  displayName: '${environmentName} MCP Server'
  identifierUris: ['api://${mcpServerApp.appId}']
  api: {
    oauth2PermissionScopes: mcpServerApp.api.oauth2PermissionScopes
    preAuthorizedApplications: mcpServerApp.api.preAuthorizedApplications
    requestedAccessTokenVersion: 2
  }
}

// Service principal for the server app
resource mcpServerSp 'Microsoft.Graph/servicePrincipals@v1.0' = {
  appId: mcpServerApp.appId
}

// Federated identity credential — passwordless OBO using managed identity
resource mcpFederatedCredential 'Microsoft.Graph/applications/federatedIdentityCredentials@v1.0' = {
  name: '${mcpServerApp.uniqueName}/McpServerOboCredential'
  audiences: [
    tokenExchangeAudience
  ]
  description: 'Federated credential for Azure MCP Server OBO flow'
  issuer: '${environment().authentication.loginEndpoint}${tenant().tenantId}/v2.0'
  subject: mcpManagedIdentity.outputs.principalId
}

// Client App Registration — used by Foundry/Copilot Studio to authenticate
resource mcpClientApp 'Microsoft.Graph/applications@v1.0' = {
  uniqueName: mcpClientUniqueName
  displayName: '${environmentName} MCP Client'
  signInAudience: 'AzureADMyOrg'
  publicClient: {
    redirectUris: [
      'http://localhost'
    ]
  }
  isFallbackPublicClient: true
}

// Service principal for the client app
resource mcpClientSp 'Microsoft.Graph/servicePrincipals@v1.0' = {
  appId: mcpClientApp.appId
}

// MCP Container Apps Environment (consumption, no VM quota needed)
module mcpEnvironment 'br/public:avm/res/app/managed-environment:0.8.0' = {
  scope: rg
  params: {
    name: '${abbrs.appManagedEnvironments}mcp-${resourceToken}'
    location: location
    tags: tags
    logAnalyticsWorkspaceResourceId: logAnalytics.outputs.resourceId
    zoneRedundant: false
  }
}

// MCP Container App — Azure MCP Server with OBO authentication
module mcpApp './app/mcp.bicep' = {
  scope: rg
  params: {
    name: mcpAppName
    location: location
    tags: tags
    environmentResourceId: mcpEnvironment.outputs.resourceId
    dockerImage: 'mcr.microsoft.com/azure-sdk/azure-mcp:${mcpImageTag}'
    args: [
      '--transport'
      'http'
      '--outgoing-auth-strategy'
      'UseOnBehalfOf'
      '--mode'
      'all'
      '--read-only'
    ]
    azureAdTenantId: tenant().tenantId
    azureAdClientId: mcpServerApp.appId
    azureAdInstance: environment().authentication.loginEndpoint
    userAssignedManagedIdentityId: mcpManagedIdentity.outputs.resourceId
    userAssignedManagedIdentityClientId: mcpManagedIdentity.outputs.clientId
    tokenExchangeAudience: tokenExchangeAudience
    appInsightsConnectionString: monitoring.outputs.connectionString
  }
}

// App outputs
output APPLICATIONINSIGHTS_CONNECTION_STRING string = monitoring.outputs.connectionString
output AZURE_LOCATION string = location
output SERVICE_API_NAME string = api.outputs.name
output SERVICE_API_URI string = 'https://${api.outputs.defaultHostname}'
output AZURE_FUNCTION_APP_NAME string = api.outputs.name
output STATIC_WEB_APP_NAME string = webapp.outputs.name
output STATIC_WEB_APP_URI string = 'https://${webapp.outputs.defaultHostname}'
output PRE_STATIC_WEB_APP_URI string = webAppName
output RESOURCE_GROUP string = rg.name
output STORAGE_CONNECTION__queueServiceUri string = 'https://${storage.outputs.name}.queue.${environment().suffixes.storage}'
output AZURE_OPENAI_ENDPOINT string = foundry.outputs.endpoint
output AZURE_OPENAI_DEPLOYMENT_NAME string = modelName
output FOUNDRY_PROJECT_ENDPOINT string = foundry.outputs.projectEndpoint
output MCP_SERVER_URI string = 'https://${mcpApp.outputs.fqdn}'
output MCP_SERVER_NAME string = mcpApp.outputs.name
output MCP_SERVER_CLIENT_ID string = mcpServerApp.appId
output MCP_CLIENT_CLIENT_ID string = mcpClientApp.appId
