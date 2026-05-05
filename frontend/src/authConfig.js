import { PublicClientApplication } from '@azure/msal-browser';

// These come from azd environment outputs, injected at build time
const msalConfig = {
  auth: {
    clientId: process.env.REACT_APP_MCP_CLIENT_ID || '',
    authority: `https://login.microsoftonline.com/${process.env.REACT_APP_TENANT_ID || 'common'}`,
    redirectUri: window.location.origin,
  },
  cache: {
    cacheLocation: 'sessionStorage', // cleared when tab closes
    storeAuthStateInCookie: false,
  },
};

export const msalInstance = new PublicClientApplication(msalConfig);

// Scope for Azure MCP server — the user's token will carry this audience
const mcpServerClientId = process.env.REACT_APP_MCP_SERVER_CLIENT_ID || '';
export const mcpScopes = mcpServerClientId
  ? [`api://${mcpServerClientId}/Mcp.Tools.ReadWrite`]
  : [];

export const loginRequest = {
  scopes: mcpScopes,
};
