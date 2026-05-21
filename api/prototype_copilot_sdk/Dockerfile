# Dockerfile for Copilot SDK + stdio MCP architecture.
#
# Bundles:
#   1. Python app (FastAPI + github-copilot-sdk)
#   2. Copilot CLI (bundled with github-copilot-sdk pip package)
#   3. Azure MCP server binary (stdio mode)
#
# The Copilot SDK spawns azure-mcp as a stdio subprocess.
# Auth to ARM uses the container's managed identity (AZURE_CLIENT_ID).
# No separate MCP Container App, no OBO, no app registrations for MCP.

FROM python:3.12-slim

# Copy azure-mcp binary from MCR (standalone .NET binary, no Node.js needed)
COPY --from=mcr.microsoft.com/azure-sdk/azure-mcp:latest /app /opt/azure-mcp

# Point the agent at the binary instead of npx
ENV AZURE_MCP_COMMAND=/opt/azure-mcp/azure-mcp
ENV AZURE_MCP_ARGS="server start --mode all --read-only"

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py azure_agent.py ticket_agent.py ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
