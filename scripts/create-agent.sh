#!/bin/bash
set -euo pipefail

echo "=== Post-provision: Creating/updating Azure AI Agent ==="

# Install required Python packages
pip install -q azure-ai-agents azure-ai-projects azure-identity requests

# Run the agent creation script
python "$(dirname "$0")/create-agent.py"

echo "=== Agent provisioning complete ==="
