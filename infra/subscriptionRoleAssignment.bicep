targetScope = 'subscription'

@description('The principal object ID to assign the role to.')
param principalId string

@description('The role definition GUID at subscription scope.')
param roleDefinitionId string

@description('The Azure principal type for the assignment.')
param principalType string = 'ServicePrincipal'

resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(subscription().subscriptionId, principalId, roleDefinitionId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleDefinitionId)
    principalId: principalId
    principalType: principalType
  }
}
