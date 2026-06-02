# Build the React frontend.
FROM node:20-slim AS frontend-build

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Backend runtime image (FastAPI + stdio Azure MCP subprocess + served frontend).
FROM python:3.12-slim

# Azure MCP server is a .NET binary and needs ICU globalization libs.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libicu-dev curl ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

# Install Node.js runtime so npx can launch @azure/mcp in stdio mode.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
	&& apt-get install -y --no-install-recommends nodejs \
	&& rm -rf /var/lib/apt/lists/*

# Install Azure CLI so azure-mcp can use AzureCliCredential in managed identity environments.
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Install Python dependencies
WORKDIR /app
COPY api/prototype_copilot_sdk/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/prototype_copilot_sdk/app.py api/prototype_copilot_sdk/azure_agent.py api/prototype_copilot_sdk/ticket_agent.py ./

# Copy compiled frontend and serve it from FastAPI.
COPY --from=frontend-build /frontend/build ./frontend_build

EXPOSE 8000

CMD ["/bin/sh", "-c", "az login --identity --username \"$AZURE_CLIENT_ID\" >/dev/null 2>&1 || true; uvicorn app:app --host 0.0.0.0 --port 8000"]
