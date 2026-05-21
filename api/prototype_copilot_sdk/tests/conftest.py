"""
Shared fixtures and skip markers for prototype_copilot_sdk integration tests.

Credentials required (set as environment variables):
  - For GitHub Copilot backend:  GITHUB_TOKEN
  - For Azure OpenAI provider:   AZURE_OPENAI_ENDPOINT (+ AZURE_OPENAI_API_KEY or MI)
  - For Azure MCP agent tests:   azure-mcp binary on disk (AZURE_MCP_PATH)

Tests that need an LLM skip when no credentials are available.
"""
import os
import sys

import pytest

# Ensure the prototype package is importable (must be before api/ on sys.path
# to avoid importing the old agent_framework-based modules)
_proto_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _proto_dir in sys.path:
    sys.path.remove(_proto_dir)
sys.path.insert(0, _proto_dir)


# --------------- credential helpers ---------------

def _has_github_token() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN"))


def _has_azure_openai() -> bool:
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))


def _has_llm_credentials() -> bool:
    return _has_github_token() or _has_azure_openai()


def _has_azure_mcp_binary() -> bool:
    path = os.environ.get("AZURE_MCP_PATH", "/usr/local/bin/azure-mcp")
    return os.path.isfile(path)


# --------------- skip markers (importable via conftest auto-loading) ---------------

requires_llm = pytest.mark.skipif(
    not _has_llm_credentials(),
    reason=(
        "No LLM credentials. Set GITHUB_TOKEN (GitHub Copilot) "
        "or AZURE_OPENAI_ENDPOINT (Azure OpenAI) to run this test."
    ),
)

requires_azure_mcp = pytest.mark.skipif(
    not _has_azure_mcp_binary(),
    reason=(
        f"azure-mcp binary not found at "
        f"{os.environ.get('AZURE_MCP_PATH', '/usr/local/bin/azure-mcp')}. "
        f"Set AZURE_MCP_PATH to the correct location."
    ),
)


# --------------- fixtures ---------------

@pytest.fixture(scope="session")
def copilot_config():
    """Build a SubprocessConfig that works in the current environment."""
    from copilot import SubprocessConfig

    kwargs = {}

    if _has_github_token():
        kwargs["github_token"] = os.environ["GITHUB_TOKEN"]

    return SubprocessConfig(**kwargs)


@pytest.fixture(scope="session")
def provider_config():
    """Build a provider config dict, or None if using default Copilot backend."""
    if not _has_azure_openai():
        return None  # Use default GitHub Copilot backend

    from azure_agent import _get_azure_provider_config
    return _get_azure_provider_config()
