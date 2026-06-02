#!/usr/bin/env sh
set -eu

resource_group="${AZURE_RESOURCE_GROUP:-${RESOURCE_GROUP:-}}"

required_vars="SERVICE_API_NAME CLIENT_APP_CLIENT_ID AZURE_TENANT_ID"
for var in $required_vars; do
  eval value="\${$var:-}"
  if [ -z "$value" ]; then
    echo "[postdeploy-verify] Missing required environment variable: $var"
    echo "[postdeploy-verify] Skipping verification."
    exit 0
  fi
done

if [ -z "$resource_group" ]; then
  echo "[postdeploy-verify] Missing resource group environment variable (AZURE_RESOURCE_GROUP or RESOURCE_GROUP)."
  echo "[postdeploy-verify] Skipping verification."
  exit 0
fi

msal_actual=$(az containerapp show \
  --name "$SERVICE_API_NAME" \
  --resource-group "$resource_group" \
  --query "properties.template.containers[0].env[?name=='MSAL_CLIENT_ID'] | [0].value" \
  -o tsv)

tenant_actual=$(az containerapp show \
  --name "$SERVICE_API_NAME" \
  --resource-group "$resource_group" \
  --query "properties.template.containers[0].env[?name=='TENANT_ID'] | [0].value" \
  -o tsv)

echo "[postdeploy-verify] Expected MSAL_CLIENT_ID: $CLIENT_APP_CLIENT_ID"
echo "[postdeploy-verify] Actual   MSAL_CLIENT_ID: ${msal_actual:-<empty>}"
echo "[postdeploy-verify] Expected TENANT_ID: $AZURE_TENANT_ID"
echo "[postdeploy-verify] Actual   TENANT_ID: ${tenant_actual:-<empty>}"

if [ "${msal_actual:-}" != "$CLIENT_APP_CLIENT_ID" ] || [ "${tenant_actual:-}" != "$AZURE_TENANT_ID" ]; then
  echo "[postdeploy-verify] Verification failed: Container App env vars do not match expected values."
  exit 1
fi

echo "[postdeploy-verify] Verification passed."
