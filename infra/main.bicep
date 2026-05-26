/*
  Prototype: main.bicep for Copilot SDK + stdio MCP strategy.

  What changed vs the current main.bicep:
  ─────────────────────────────────────────
  REMOVED:
    - Function App + App Service Plan (Flex Consumption)
    - Durable Task Scheduler + Task Hub + DTS roles
    - All MCP Container App / App Registration / OBO resources
    - Deployment storage container (Azure Functions packaging)

  REPLACED WITH:
    - Container Apps Environment + Container App (FastAPI + Copilot SDK)
    - Azure Container Registry (for Docker image)

  KEPT:
    - User-assigned Managed Identity (for ARM, AI Services, storage)
    - Storage Account (for app data if needed)
    - AI Services / Azure OpenAI (optional — can use GitHub Copilot backend instead)
    - Log Analytics + Application Insights
    - VNet + Private Endpoint (optional)
    - Client App Registration (SPA MSAL, ARM user_impersonation scope)

  The infra/app/mcp.bicep and infra/app/dts.bicep are NO LONGER NEEDED.
*/
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

@description('Name of the API service (Container App)')
param apiServiceName string = ''

@description('Name of the user assigned identity')
param apiUserAssignedIdentityName string = ''

@description('Name of the application insights resource')
param applicationInsightsName string = ''

@description('Name of the container registry')
param containerRegistryName string = ''

@description('Name of the Container Apps Environment')
param containerAppsEnvironmentName string = ''

@description('Container image name for the API service. Set by azd after build+push.')
param apiImageName string = ''

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

@description('Model deployment SKU name')
param modelSkuName string = 'S0'

@description('Model deployment capacity')
param modelCapacity int = 10

@description('Model deployment location. If you want to deploy an Azure AI resource/model in different location than the rest of the resources created.')
param modelLocation string = location

@description('The AI Service Account full ARM Resource ID. This is an optional field, and if not provided, the resource will be created.')
param aiServiceAccountResourceId string = ''

// NOTE: MCP Server Parameters section REMOVED entirely.
// MCP servers now run as stdio subprocesses inside the app container.
// No mcpImageTag, mcpServiceName, or freshdeskMcpUri params needed.

// Variables
var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, rg.id, environmentName, location))
var aiResourceToken = toLower(uniqueString(subscription().id, rg.id, environmentName, modelLocation))
var tags = { 'azd-env-name': environmentName }
var apiAppName = !empty(apiServiceName) ? apiServiceName : '${abbrs.appContainerApps}api-${resourceToken}'

// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2025-04-01' = {
  name: !empty(resourceGroupName) ? resourceGroupName : '${abbrs.resourcesResourceGroups}${environmentName}'
  location: location
  tags: tags
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

// Container Registry using AVM
module acr 'br/public:avm/res/container-registry/registry:0.9.1' = {
  name: 'acr-${resourceToken}'
  scope: rg
  params: {
    name: !empty(containerRegistryName) ? containerRegistryName : '${abbrs.containerRegistryRegistries}${resourceToken}'
    location: location
    tags: tags
    acrSku: 'Basic'
    acrAdminUserEnabled: false
    roleAssignments: [
      {
        principalId: apiUserAssignedIdentity.outputs.principalId
        roleDefinitionIdOrName: '7f951dda-4ed3-4680-a7ca-43fe172d538d' // AcrPull
        principalType: 'ServicePrincipal'
      }
    ]
  }
}

// Container Apps Environment using AVM
module containerAppsEnv 'br/public:avm/res/app/managed-environment:0.8.1' = {
  name: 'cae-${resourceToken}'
  scope: rg
  params: {
    name: !empty(containerAppsEnvironmentName) ? containerAppsEnvironmentName : '${abbrs.appManagedEnvironments}${resourceToken}'
    location: location
    tags: tags
    logAnalyticsWorkspaceResourceId: logAnalytics.outputs.resourceId
    zoneRedundant: false
  }
}

// Container App — FastAPI + Copilot SDK + azure-mcp (stdio)
module api 'br/public:avm/res/app/container-app:0.12.0' = {
  name: 'api-${resourceToken}'
  scope: rg
  params: {
    name: apiAppName
    location: location
    tags: union(tags, { 'azd-service-name': 'api' })
    environmentResourceId: containerAppsEnv.outputs.resourceId
    managedIdentities: {
      userAssignedResourceIds: [apiUserAssignedIdentity.outputs.resourceId]
    }
    registries: [
      {
        server: acr.outputs.loginServer
        identity: apiUserAssignedIdentity.outputs.resourceId
      }
    ]
    containers: [
      {
        name: 'api'
        image: !empty(apiImageName) ? apiImageName : 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
        resources: {
          cpu: '1.0'
          memory: '2Gi'
        }
        env: [
          { name: 'AZURE_CLIENT_ID', value: apiUserAssignedIdentity.outputs.clientId }
          { name: 'COPILOT_MODEL', value: 'gpt-5-mini' }
          { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: monitoring.outputs.connectionString }
          // GITHUB_TOKEN should be set via Key Vault reference or manual config
          // AZURE_OPENAI_ENDPOINT is optional — only if using Azure OpenAI instead of GitHub Copilot
        ]
      }
    ]
    ingressTargetPort: 8000
    ingressExternal: true
    ingressTransport: 'auto'
    scaleMinReplicas: 1
    scaleMaxReplicas: 5
  }
}

// AI Services configuration
var aiServiceExists = aiServiceAccountResourceId != ''
var aiServiceName = '${aiServicesName}${aiResourceToken}'

// AI Services (Cognitive Services) using AVM with model deployment
module aiServices 'br/public:avm/res/cognitive-services/account:0.9.2' = if (!aiServiceExists) {
  scope: rg
  name: 'aiServices-${aiResourceToken}'
  params: {
    name: aiServiceName
    location: modelLocation
    tags: tags
    kind: 'AIServices'
    customSubDomainName: toLower(aiServiceName)
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: true
    sku: modelSkuName
    deployments: [
      {
        name: modelName
        model: {
          format: modelFormat
          name: modelName
          version: modelVersion
        }
        sku: {
          name: 'GlobalStandard'
          capacity: modelCapacity
        }
      }
    ]
    roleAssignments: [
      {
        principalId: apiUserAssignedIdentity.outputs.principalId
        roleDefinitionIdOrName: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd' // Cognitive Services OpenAI User
        principalType: 'ServicePrincipal'
      }
      {
        principalId: principalId
        roleDefinitionIdOrName: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd' // Cognitive Services OpenAI User
        principalType: 'User'
      }
    ]
  }
}

// Storage role assignments using AVM pattern
var storageQueueDataContributorRole = '974c5e8b-45b9-4653-ba55-5f855dd0fb88'

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

// ============= Client App Registration (simplified) =============
// Only need a single app registration for user sign-in (SPA + MSAL).
// No server app reg, no custom scopes, no FIC, no OBO.
// The client app requests ARM user_impersonation so users can get ARM tokens.

var clientAppUniqueName = '${environmentName}-client'

resource clientApp 'Microsoft.Graph/applications@v1.0' = {
  uniqueName: clientAppUniqueName
  displayName: '${environmentName} Client'
  signInAudience: 'AzureADMyOrg'
  spa: {
    redirectUris: [
      'http://localhost:3000'
      'https://${api.outputs.fqdn}'
    ]
  }
  publicClient: {
    redirectUris: [
      'http://localhost'
    ]
  }
  isFallbackPublicClient: true
  requiredResourceAccess: [
    {
      // Azure Resource Manager — user_impersonation
      // This lets the frontend acquire ARM tokens directly for the logged-in user.
      // No custom MCP audience needed.
      resourceAppId: '797f4846-ba00-4fd7-ba43-dac1f8f63013'
      resourceAccess: [
        {
          id: '41094075-9dad-400e-a0bd-54e686782033' // user_impersonation
          type: 'Scope'
        }
      ]
    }
  ]
}

resource clientSp 'Microsoft.Graph/servicePrincipals@v1.0' = {
  appId: clientApp.appId
}

// App outputs
output APPLICATIONINSIGHTS_CONNECTION_STRING string = monitoring.outputs.connectionString
output AZURE_LOCATION string = location
output SERVICE_API_NAME string = api.outputs.name
output SERVICE_API_URI string = 'https://${api.outputs.fqdn}'
output CONTAINER_REGISTRY_NAME string = acr.outputs.name
output CONTAINER_REGISTRY_LOGIN_SERVER string = acr.outputs.loginServer
output RESOURCE_GROUP string = rg.name
output AZURE_OPENAI_ENDPOINT string = aiServiceExists ? reference(aiServiceAccountResourceId, '2023-05-01').endpoint : aiServices!.outputs.endpoint
output AZURE_OPENAI_DEPLOYMENT_NAME string = modelName
output CLIENT_APP_CLIENT_ID string = clientApp.appId  // For MSAL in the frontend
output AZURE_TENANT_ID string = tenant().tenantId
