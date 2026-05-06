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

@description('Name of the application insights resource')
param applicationInsightsName string = ''

@description('Name of the log analytics workspace')
param logAnalyticsName string = ''

@description('Name of the resource group')
param resourceGroupName string = ''

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
var mcpAppName = !empty(mcpServiceName) ? mcpServiceName : '${abbrs.appContainerApps}mcp-${resourceToken}'

// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2025-04-01' = {
  name: !empty(resourceGroupName) ? resourceGroupName : '${abbrs.resourcesResourceGroups}${environmentName}'
  location: location
  tags: tags
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

// Role assignment for the deploying user on the Foundry resource
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
var mcpAppRoleId = guid(subscription().id, environmentName, 'Mcp.Tools.ReadWrite.All')
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
  // App role for managed identity auth (e.g. Foundry project MI)
  appRoles: [
    {
      id: mcpAppRoleId
      allowedMemberTypes: ['Application']
      displayName: 'MCP Tools ReadWrite All'
      description: 'Allow the application to access Azure MCP tools without a signed-in user.'
      isEnabled: true
      value: 'Mcp.Tools.ReadWrite.All'
    }
  ]
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

// Grant Foundry project MI the app role on the MCP server SP
resource foundryMcpAppRoleAssignment 'Microsoft.Graph/appRoleAssignedTo@v1.0' = {
  appRoleId: mcpAppRoleId
  principalId: foundry.outputs.projectPrincipalId
  resourceId: mcpServerSp.id
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

// Client App Registration — used by Foundry Agent Service for OAuth identity passthrough
resource mcpClientApp 'Microsoft.Graph/applications@v1.0' = {
  uniqueName: mcpClientUniqueName
  displayName: '${environmentName} MCP Client'
  signInAudience: 'AzureADMyOrg'
  web: {
    redirectUris: [
      'http://localhost'
    ]
    implicitGrantSettings: {
      enableIdTokenIssuance: false
      enableAccessTokenIssuance: false
    }
  }
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

// Outputs
output APPLICATIONINSIGHTS_CONNECTION_STRING string = monitoring.outputs.connectionString
output AZURE_LOCATION string = location
output RESOURCE_GROUP string = rg.name
output AZURE_OPENAI_ENDPOINT string = foundry.outputs.endpoint
output AZURE_OPENAI_DEPLOYMENT_NAME string = modelName
output FOUNDRY_PROJECT_ENDPOINT string = foundry.outputs.projectEndpoint
output MCP_SERVER_URI string = 'https://${mcpApp.outputs.fqdn}'
output MCP_SERVER_NAME string = mcpApp.outputs.name
output MCP_SERVER_CLIENT_ID string = mcpServerApp.appId
output MCP_CLIENT_CLIENT_ID string = mcpClientApp.appId
output AZURE_TENANT_ID string = tenant().tenantId
