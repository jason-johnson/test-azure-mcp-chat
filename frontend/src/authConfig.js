import { PublicClientApplication } from '@azure/msal-browser';

const runtimeConfig = window.__APP_CONFIG__ || {};

// Config resolves in this order: runtime injection -> build env vars -> safe defaults.
const msalConfig = {
  auth: {
    clientId: runtimeConfig.msalClientId || process.env.REACT_APP_MCP_CLIENT_ID || '',
    authority: `https://login.microsoftonline.com/${runtimeConfig.tenantId || process.env.REACT_APP_TENANT_ID || 'common'}`,
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
