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

# API calls are relative (/api/...) since frontend is served from the same origin
ENV REACT_APP_API_URL=/api
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.12-slim

# Copy azure-mcp binary from MCR (standalone .NET binary, no Node.js needed)
COPY --from=mcr.microsoft.com/azure-sdk/azure-mcp:latest /app /opt/azure-mcp

# Point the agent at the binary instead of npx
ENV AZURE_MCP_COMMAND=/opt/azure-mcp/azure-mcp
ENV AZURE_MCP_ARGS="server start --mode all --read-only"

# Install Python dependencies
WORKDIR /app
COPY api/prototype_copilot_sdk/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/prototype_copilot_sdk/app.py api/prototype_copilot_sdk/azure_agent.py api/prototype_copilot_sdk/ticket_agent.py ./

# Copy built React app so FastAPI can serve it as static files
COPY --from=frontend-build /frontend/build ./static

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
