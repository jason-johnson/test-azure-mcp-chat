# Prototype Dockerfile for Copilot SDK + stdio MCP architecture.
#
# Bundles:
#   1. Python app (FastAPI + github-copilot-sdk)
#   2. GitHub Copilot CLI (agent runtime)
#   3. Azure MCP server binary (stdio mode)
#
# The Copilot CLI spawns azure-mcp as a stdio subprocess.
# Auth to ARM uses the container's managed identity (AZURE_CLIENT_ID).
# No separate MCP Container App, no OBO, no app registrations for MCP.

FROM python:3.11-slim

# Install azure-mcp binary
# The Azure MCP server is a single .NET binary distributed via MCR
# In production, pin to a specific tag instead of 'latest'
COPY --from=mcr.microsoft.com/azure-sdk/azure-mcp:latest /app /opt/azure-mcp
ENV AZURE_MCP_PATH=/opt/azure-mcp/azure-mcp

# Install Copilot CLI
# NOTE: Actual installation method TBD — may be npm package, binary download, etc.
# RUN npm install -g @github/copilot-cli
# For now, assume it's available at this path:
ENV COPILOT_CLI_PATH=/usr/local/bin/copilot-cli

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copilot CLI MCP server configuration
# This tells the CLI to spawn azure-mcp in stdio mode
COPY mcp-config.json /root/.copilot/mcp.json

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
