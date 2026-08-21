"""Unit tests for the external tool registry and A2A invoker."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.integrations.a2a import (
    A2AInvocationError,
    invoke_external_tool,
    validate_arguments,
)
from backend.integrations.registry import (
    ExternalToolDef,
    ExternalToolRegistry,
    _substitute_env,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SAMPLE_YAML = """
tools:
  - name: call_test_agent
    provider: test
    capability: test_research
    visibility: always
    description: Test agent for unit tests.
    keywords:
      - test
      - unit
    llm_parameters:
      type: object
      properties:
        prompt:
          type: string
          description: A test prompt.
      required:
        - prompt
      additionalProperties: false
    agent_owner: test-owner
    protocol: A2A_JSONRPC
    base_url: https://test.example.com
    path: /agents/test-agent/
  - name: hidden_tool
    provider: test
    capability: hidden
    visibility: hidden
    description: A hidden tool.
    llm_parameters:
      type: object
      properties:
        query:
          type: string
      required:
        - query
    agent_owner: test-owner
    protocol: A2A_JSONRPC
    base_url: https://hidden.example.com
    path: /agents/hidden/
"""


@pytest.fixture
def tmp_registry(tmp_path):
    """Create a temp registry YAML and return a loaded ExternalToolRegistry."""
    registry_path = tmp_path / "tool_registry.yaml"
    registry_path.write_text(SAMPLE_YAML)
    return ExternalToolRegistry(registry_path)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestExternalToolRegistry:
    def test_loads_tools(self, tmp_registry):
        assert len(tmp_registry) == 2
        assert "call_test_agent" in tmp_registry
        assert "hidden_tool" in tmp_registry

    def test_get_tool(self, tmp_registry):
        tool = tmp_registry.get("call_test_agent")
        assert tool is not None
        assert tool.name == "call_test_agent"
        assert tool.provider == "test"
        assert tool.protocol == "A2A_JSONRPC"
        assert tool.base_url == "https://test.example.com"
        assert tool.path == "/agents/test-agent/"

    def test_get_unknown_returns_none(self, tmp_registry):
        assert tmp_registry.get("nonexistent") is None

    def test_resolved_url(self, tmp_registry):
        tool = tmp_registry.get("call_test_agent")
        assert tool.resolved_url == "https://test.example.com/agents/test-agent/"

    def test_mcp_visible(self, tmp_registry):
        mcp_tools = tmp_registry.list_mcp_visible()
        assert len(mcp_tools) == 1
        assert mcp_tools[0].name == "call_test_agent"

    def test_rest_visible(self, tmp_registry):
        rest_tools = tmp_registry.list_rest_visible()
        assert len(rest_tools) == 1
        assert rest_tools[0].name == "call_test_agent"

    def test_hidden_tool_not_visible(self, tmp_registry):
        hidden = tmp_registry.get("hidden_tool")
        assert not hidden.is_mcp_visible
        assert not hidden.is_rest_visible

    def test_empty_registry(self, tmp_path):
        empty_path = tmp_path / "empty.yaml"
        empty_path.write_text("tools: []\n")
        reg = ExternalToolRegistry(empty_path)
        assert len(reg) == 0

    def test_missing_file(self, tmp_path):
        reg = ExternalToolRegistry(tmp_path / "nonexistent.yaml")
        assert len(reg) == 0

    def test_reload(self, tmp_registry, tmp_path):
        assert len(tmp_registry) == 2
        # Overwrite with empty.
        tmp_registry.registry_path.write_text("tools: []\n")
        tmp_registry.reload()
        assert len(tmp_registry) == 0


# ---------------------------------------------------------------------------
# Environment variable substitution tests
# ---------------------------------------------------------------------------

class TestEnvSubstitution:
    def test_substitute_env(self, monkeypatch):
        monkeypatch.setenv("TEST_HOST", "myhost.example.com")
        result = _substitute_env("https://${TEST_HOST}/path")
        assert result == "https://myhost.example.com/path"

    def test_unresolved_env(self):
        result = _substitute_env("https://${UNDEFINED_VAR}/path")
        assert result == "https://${UNDEFINED_VAR}/path"

    def test_registry_substitutes_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_HOST", "resolved.example.com")
        yaml_content = """
tools:
  - name: env_test
    provider: test
    capability: test
    visibility: always
    description: test
    llm_parameters:
      type: object
      properties:
        prompt:
          type: string
      required:
        - prompt
    agent_owner: test
    protocol: A2A_JSONRPC
    base_url: https://${TEST_HOST}
    path: /agents/test/
"""
        p = tmp_path / "registry.yaml"
        p.write_text(yaml_content)
        reg = ExternalToolRegistry(p)
        tool = reg.get("env_test")
        assert tool.base_url == "https://resolved.example.com"
        assert tool.resolved_url == "https://resolved.example.com/agents/test/"


# ---------------------------------------------------------------------------
# Argument validation tests
# ---------------------------------------------------------------------------

class TestValidateArguments:
    def test_valid_args(self, tmp_registry):
        tool = tmp_registry.get("call_test_agent")
        errors = validate_arguments(tool, {"prompt": "hello"})
        assert errors == []

    def test_missing_required(self, tmp_registry):
        tool = tmp_registry.get("call_test_agent")
        errors = validate_arguments(tool, {})
        assert "missing required parameter: prompt" in errors

    def test_unknown_param(self, tmp_registry):
        tool = tmp_registry.get("call_test_agent")
        errors = validate_arguments(tool, {"prompt": "x", "extra": 1})
        assert any("unknown parameter" in e for e in errors)

    def test_wrong_type(self, tmp_registry):
        tool = tmp_registry.get("call_test_agent")
        errors = validate_arguments(tool, {"prompt": 123})
        assert any("must be string" in e for e in errors)


# ---------------------------------------------------------------------------
# A2A invoker tests (with mocked HTTP)
# ---------------------------------------------------------------------------

class TestA2AInvoker:
    def test_invoke_unknown_tool(self, tmp_registry, monkeypatch):
        # Patch the registry singleton.
        from backend.integrations import a2a as a2a_mod
        monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: tmp_registry)
        with pytest.raises(A2AInvocationError, match="not found"):
            invoke_external_tool("nonexistent", {"prompt": "test"})

    def test_invoke_unresolved_url(self, tmp_registry, monkeypatch):
        from backend.integrations import a2a as a2a_mod
        monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: tmp_registry)
        # Override base_url to have an unresolved env var.
        tool = tmp_registry.get("call_test_agent")
        tool.base_url = "${UNRESOLVED_HOST}"
        with pytest.raises(A2AInvocationError, match="unresolved base_url"):
            invoke_external_tool("call_test_agent", {"prompt": "test"})

    def test_invoke_missing_prompt(self, tmp_registry, monkeypatch):
        from backend.integrations import a2a as a2a_mod
        monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: tmp_registry)
        with pytest.raises(A2AInvocationError, match="requires a 'prompt'"):
            invoke_external_tool("call_test_agent", {})

    def test_invoke_success(self, tmp_registry, monkeypatch):
        from backend.integrations import a2a as a2a_mod
        monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: tmp_registry)

        # Mock httpx.Client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "id": "task-123",
                "state": "completed",
                "artifacts": [{"type": "text", "text": "Research result."}],
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.close = MagicMock()

        result = invoke_external_tool(
            "call_test_agent", {"prompt": "Compare DDR5 power management."},
            client=mock_client,
        )
        assert result["state"] == "completed"
        assert result["id"] == "task-123"
        assert mock_client.post.called

        # Verify the request body was JSON-RPC.
        call_args = mock_client.post.call_args
        body = call_args.kwargs["json"]
        assert body["jsonrpc"] == "2.0"
        assert body["method"] == "tasks/send"
        assert body["params"]["message"]["parts"][0]["text"] == "Compare DDR5 power management."

    def test_invoke_jsonrpc_error(self, tmp_registry, monkeypatch):
        from backend.integrations import a2a as a2a_mod
        monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: tmp_registry)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "error": {"code": -32600, "message": "Invalid request"},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with pytest.raises(A2AInvocationError, match="Invalid request"):
            invoke_external_tool(
                "call_test_agent", {"prompt": "test"},
                client=mock_client,
            )

    def test_invoke_http_error(self, tmp_registry, monkeypatch):
        import httpx as httpx_mod
        from backend.integrations import a2a as a2a_mod
        monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: tmp_registry)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=mock_response,
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with pytest.raises(A2AInvocationError, match="HTTP request failed"):
            invoke_external_tool(
                "call_test_agent", {"prompt": "test"},
                client=mock_client,
            )
