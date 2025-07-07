#!/bin/bash

# Script to set environment variables for Azure App Service from .env file
# This script reads environment variables from .env file and sets them on the Azure App Service

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found in current directory"
    echo "Please create a .env file with the required environment variables:"
    echo "  AZURE_OPENAI_API_KEY=your_api_key"
    echo "  AZURE_OPENAI_ENDPOINT=your_endpoint"
    echo "  AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name"
    exit 1
fi

# Check if required variables are set
if [ -z "$APP_SERVICE_NAME" ]; then
    echo "Error: APP_SERVICE_NAME environment variable is not set"
    exit 1
fi

if [ -z "$RESOURCE_GROUP_NAME" ]; then
    echo "Error: RESOURCE_GROUP_NAME environment variable is not set"
    exit 1
fi

echo "Reading environment variables from .env file..."

# Read .env file and extract the required variables
AZURE_OPENAI_API_KEY=$(grep "^AZURE_OPENAI_API_KEY=" .env | cut -d '=' -f2- | sed 's/^["'"'"']\|["'"'"']$//g')
AZURE_OPENAI_ENDPOINT=$(grep "^AZURE_OPENAI_ENDPOINT=" .env | cut -d '=' -f2- | sed 's/^["'"'"']\|["'"'"']$//g')
AZURE_OPENAI_DEPLOYMENT_NAME=$(grep "^AZURE_OPENAI_DEPLOYMENT_NAME=" .env | cut -d '=' -f2- | sed 's/^["'"'"']\|["'"'"']$//g')

# Validate that required variables were found
if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo "Warning: AZURE_OPENAI_API_KEY not found in .env file"
fi

if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
    echo "Warning: AZURE_OPENAI_ENDPOINT not found in .env file"
fi

if [ -z "$AZURE_OPENAI_DEPLOYMENT_NAME" ]; then
    echo "Warning: AZURE_OPENAI_DEPLOYMENT_NAME not found in .env file"
fi

echo "Setting environment variables for App Service: $APP_SERVICE_NAME"

# Set environment variables on Azure App Service
if [ -n "$AZURE_OPENAI_API_KEY" ]; then
    echo "Setting AZURE_OPENAI_API_KEY..."
    az webapp config appsettings set \
        --name $APP_SERVICE_NAME \
        --resource-group $RESOURCE_GROUP_NAME \
        --settings AZURE_OPENAI_API_KEY="$AZURE_OPENAI_API_KEY"
fi

if [ -n "$AZURE_OPENAI_ENDPOINT" ]; then
    echo "Setting AZURE_OPENAI_ENDPOINT..."
    az webapp config appsettings set \
        --name $APP_SERVICE_NAME \
        --resource-group $RESOURCE_GROUP_NAME \
        --settings AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT"
fi

if [ -n "$AZURE_OPENAI_DEPLOYMENT_NAME" ]; then
    echo "Setting AZURE_OPENAI_DEPLOYMENT_NAME..."
    az webapp config appsettings set \
        --name $APP_SERVICE_NAME \
        --resource-group $RESOURCE_GROUP_NAME \
        --settings AZURE_OPENAI_DEPLOYMENT_NAME="$AZURE_OPENAI_DEPLOYMENT_NAME"
fi

if [ -n "$MCP_URL" ]; then
    echo "Setting MCP_URL..."
    az webapp config.appsettings set \
        --name $APP_SERVICE_NAME \
        --resource-group $RESOURCE_GROUP_NAME \
        --settings MCP_URL="$MCP_URL"
fi

echo "Environment variables have been set successfully!"
echo ""
echo "You can verify the settings with:"
echo "az webapp config appsettings list --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP_NAME"
