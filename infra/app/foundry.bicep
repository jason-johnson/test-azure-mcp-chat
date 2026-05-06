@description('Name of the AI Foundry resource')
param name string

@description('Name of the project')
param projectName string

@description('Location for resources')
param location string

@description('Tags for resources')
param tags object = {}

@description('Model name to deploy')
param modelName string

@description('Model format')
param modelFormat string = 'OpenAI'

@description('Model version')
param modelVersion string

@description('Model deployment capacity')
param modelCapacity int = 10

// AI Foundry resource (CognitiveServices account with project management enabled)
resource aiFoundry 'Microsoft.CognitiveServices/accounts@2025-06-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'S0'
  }
  kind: 'AIServices'
  properties: {
    allowProjectManagement: true
    customSubDomainName: toLower(name)
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: true
  }
}

// Foundry project — groups agents, threads, files, evaluations
resource project 'Microsoft.CognitiveServices/accounts/projects@2025-06-01' = {
  name: projectName
  parent: aiFoundry
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {}
}

// Model deployment for agent use
resource modelDeployment 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: aiFoundry
  name: modelName
  sku: {
    name: 'GlobalStandard'
    capacity: modelCapacity
  }
  properties: {
    model: {
      name: modelName
      format: modelFormat
      version: modelVersion
    }
  }
}

output accountName string = aiFoundry.name
output endpoint string = aiFoundry.properties.endpoint
output projectName string = project.name
output projectEndpoint string = '${aiFoundry.properties.endpoint}api/projects/${project.name}'
output resourceId string = aiFoundry.id
output projectPrincipalId string = project.identity.principalId
