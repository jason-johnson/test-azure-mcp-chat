#!/usr/bin/env sh
set -eu

# azd injects environment values from .azure/<env>/.env into hooks.
resource_group="${AZURE_RESOURCE_GROUP:-${RESOURCE_GROUP:-}}"

required_vars="SERVICE_API_NAME CLIENT_APP_CLIENT_ID AZURE_TENANT_ID"
for var in $required_vars; do
  eval value="\${$var:-}"
  if [ -z "$value" ]; then
    echo "[postdeploy] Missing required environment variable: $var"
    echo "[postdeploy] Skipping Container App env update."
    exit 0
  fi
done

if [ -z "$resource_group" ]; then
  echo "[postdeploy] Missing resource group environment variable (AZURE_RESOURCE_GROUP or RESOURCE_GROUP)."
  echo "[postdeploy] Skipping Container App env update."
  exit 0
fi

echo "[postdeploy] Updating Container App env vars on ${SERVICE_API_NAME} in ${resource_group}..."
az containerapp update \
  --name "$SERVICE_API_NAME" \
  --resource-group "$resource_group" \
  --set-env-vars \
    "MSAL_CLIENT_ID=$CLIENT_APP_CLIENT_ID" \
    "TENANT_ID=$AZURE_TENANT_ID"

echo "[postdeploy] Container App env vars updated successfully."

"$(dirname "$0")/postdeploy-verify.sh"
