# Dockerfile for Copilot SDK + stdio MCP architecture.
#
# Bundles:
#   1. React frontend (built as static files, served by FastAPI)
#   2. Python app (FastAPI + github-copilot-sdk)
#   3. Copilot CLI (bundled with github-copilot-sdk pip package)
#   4. Azure MCP server binary (stdio mode)
#
# Single container/pod: FastAPI serves both the API (/api/*) and the React SPA (/).
# The Copilot SDK spawns azure-mcp as a stdio subprocess.
# Auth to ARM uses the container's managed identity (AZURE_CLIENT_ID).

# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --quiet

# Injected by azd at build time via buildArgs
ARG REACT_APP_MCP_CLIENT_ID
ARG REACT_APP_TENANT_ID
ENV REACT_APP_API_URL=/api
ENV REACT_APP_MCP_CLIENT_ID=$REACT_APP_MCP_CLIENT_ID
ENV REACT_APP_TENANT_ID=$REACT_APP_TENANT_ID
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.12-slim

# Azure MCP server is a .NET binary and needs ICU globalization libs.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libicu-dev curl \
	&& rm -rf /var/lib/apt/lists/*

# Install Azure CLI so azure-mcp can use AzureCliCredential in managed identity environments.
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Copy entire Node.js installation from the frontend build stage
COPY --from=frontend-build /usr/local /usr/local

# Install Python dependencies
WORKDIR /app
COPY api/prototype_copilot_sdk/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/prototype_copilot_sdk/app.py api/prototype_copilot_sdk/azure_agent.py api/prototype_copilot_sdk/ticket_agent.py ./

# Copy built React app so FastAPI can serve it as static files
COPY --from=frontend-build /frontend/build ./static

EXPOSE 8000

CMD ["/bin/sh", "-c", "az login --identity --username \"$AZURE_CLIENT_ID\" >/dev/null 2>&1 || true; uvicorn app:app --host 0.0.0.0 --port 8000"]
