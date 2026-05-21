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

// ARM scope — the user's token grants access to Azure resources directly.
// No custom MCP audience needed; azure-mcp runs as a stdio subprocess.
export const armScopes = ['https://management.azure.com/user_impersonation'];

export const loginRequest = {
  scopes: armScopes,
};
