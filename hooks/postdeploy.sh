#!/usr/bin/env sh
set -eu

# azd injects environment values from .azure/<env>/.env into hooks.
required_vars="AZURE_RESOURCE_GROUP SERVICE_API_NAME CLIENT_APP_CLIENT_ID AZURE_TENANT_ID"
for var in $required_vars; do
  eval value="\${$var:-}"
  if [ -z "$value" ]; then
    echo "[postdeploy] Missing required environment variable: $var"
    echo "[postdeploy] Skipping Container App env update."
    exit 0
  fi
done

echo "[postdeploy] Updating Container App env vars on ${SERVICE_API_NAME}..."
az containerapp update \
  --name "$SERVICE_API_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --set-env-vars \
    "MSAL_CLIENT_ID=$CLIENT_APP_CLIENT_ID" \
    "TENANT_ID=$AZURE_TENANT_ID"

echo "[postdeploy] Container App env vars updated successfully."
