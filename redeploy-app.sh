#!/bin/bash

# Script to redeploy the app to Azure App Service
# This script redeploys the application from the connected source repository

# Check if required variables are set
if [ -z "$APP_SERVICE_NAME" ]; then
    echo "Error: APP_SERVICE_NAME environment variable is not set"
    echo "Please set it with: export APP_SERVICE_NAME='your-app-service-name'"
    exit 1
fi

if [ -z "$RESOURCE_GROUP_NAME" ]; then
    echo "Error: RESOURCE_GROUP_NAME environment variable is not set"
    echo "Please set it with: export RESOURCE_GROUP_NAME='your-resource-group-name'"
    exit 1
fi

echo "Starting redeployment of app: $APP_SERVICE_NAME"
echo "Resource Group: $RESOURCE_GROUP_NAME"
echo ""

# Check if app service exists
echo "Verifying app service exists..."
if ! az webapp show --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP_NAME >/dev/null 2>&1; then
    echo "Error: App Service '$APP_SERVICE_NAME' not found in resource group '$RESOURCE_GROUP_NAME'"
    echo "Please verify the names are correct."
    exit 1
fi

# Deploy from current directory using ZIP deployment
echo "Deploying from current directory using ZIP deployment..."

# Create a temporary zip file excluding common non-deployment files
TEMP_ZIP="/tmp/app-deployment-$(date +%s).zip"

# Create zip excluding common development files
zip -r "$TEMP_ZIP" . \
    -x "*.git*" \
    -x "*__pycache__*" \
    -x "*.pyc" \
    -x "*venv*" \
    -x "*env*" \
    -x "*.env" \
    -x "*node_modules*" \
    -x "*.DS_Store" \
    -x "*Thumbs.db" \
    -x "*.log" \
    -x "*temp*" \
    -x "*tmp*"

if [ ! -f "$TEMP_ZIP" ]; then
    echo "❌ Failed to create deployment package"
    exit 1
fi

echo "Created deployment package: $TEMP_ZIP"

# Deploy using ZIP
az webapp deployment source config-zip \
    --name $APP_SERVICE_NAME \
    --resource-group $RESOURCE_GROUP_NAME \
    --src "$TEMP_ZIP"

if [ $? -eq 0 ]; then
    echo "✅ ZIP deployment completed successfully!"
    
    # Clean up temp file
    rm -f "$TEMP_ZIP"
    
    echo ""
    echo "You can monitor the application logs with:"
    echo "az webapp log tail --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP_NAME"
    echo ""
    echo "Or check deployment history with:"
    echo "az webapp deployment list --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP_NAME"
else
    echo "❌ ZIP deployment failed"
    rm -f "$TEMP_ZIP"
    exit 1
fi

echo ""
echo "App URL: https://$APP_SERVICE_NAME.azurewebsites.net"
echo ""
echo "To restart the app service:"
echo "az webapp restart --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP_NAME"
