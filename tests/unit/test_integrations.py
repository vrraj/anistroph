"""Unit tests — external tool registry and A2A invoker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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


SAMPLE_YAML = """
tools:
  - name: call_test_agent
    provider: test
    capability: test_research
    visibility: always
    description: Test agent for unit tests.
    keywords:
      - test
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
    registry_path = tmp_path / "tool_registry.yaml"
    registry_path.write_text(SAMPLE_YAML)
    return ExternalToolRegistry(registry_path)


class TestExternalToolRegistry:
    def test_loads_and_filters_tools(self, tmp_registry):
        assert len(tmp_registry) == 2
        assert tmp_registry.get("call_test_agent") is not None
        assert tmp_registry.get("nonexistent") is None
        assert len(tmp_registry.list_mcp_visible()) == 1
        assert len(tmp_registry.list_rest_visible()) == 1
        # Hidden tool not visible
        assert not tmp_registry.get("hidden_tool").is_mcp_visible

    def test_resolved_url(self, tmp_registry):
        tool = tmp_registry.get("call_test_agent")
        assert tool.resolved_url == "https://test.example.com/agents/test-agent/"

    def test_env_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_HOST", "resolved.example.com")
        p = tmp_path / "registry.yaml"
        p.write_text("""
tools:
  - name: env_test
    provider: test
    capability: test
    visibility: always
    description: test
    llm_parameters:
      type: object
      properties: {prompt: {type: string}}
      required: [prompt]
    agent_owner: test
    protocol: A2A_JSONRPC
    base_url: https://${TEST_HOST}
    path: /agents/test/
""")
        reg = ExternalToolRegistry(p)
        assert reg.get("env_test").base_url == "https://resolved.example.com"


class TestValidateArguments:
    def test_valid_and_invalid_args(self, tmp_registry):
        tool = tmp_registry.get("call_test_agent")
        assert validate_arguments(tool, {"prompt": "hello"}) == []
        assert "missing required parameter" in validate_arguments(tool, {})[0]
        assert "unknown parameter" in validate_arguments(tool, {"prompt": "x", "extra": 1})[0]


class TestA2AInvoker:
    def test_invoke_success(self, tmp_registry, monkeypatch):
        from backend.integrations import a2a as a2a_mod
        monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: tmp_registry)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"id": "task-123", "state": "completed",
                       "artifacts": [{"type": "text", "text": "Result."}]},
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        result = invoke_external_tool("call_test_agent", {"prompt": "test"}, client=mock_client)
        assert result["state"] == "completed"

        # Verify A2A v1.0 protocol format.
        body = mock_client.post.call_args.kwargs["json"]
        assert body["method"] == "SendMessage"
        assert body["params"]["message"]["role"] == "ROLE_USER"
        assert "messageId" in body["params"]["message"]
        headers = mock_client.post.call_args.kwargs["headers"]
        assert headers["A2A-Version"] == "1.0"

    def test_invoke_errors(self, tmp_registry, monkeypatch):
        from backend.integrations import a2a as a2a_mod
        monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: tmp_registry)

        # Unknown tool
        with pytest.raises(A2AInvocationError, match="not found"):
            invoke_external_tool("nonexistent", {"prompt": "test"})

        # Missing prompt
        with pytest.raises(A2AInvocationError, match="requires a 'prompt'"):
            invoke_external_tool("call_test_agent", {})

        # Unresolved URL
        tool = tmp_registry.get("call_test_agent")
        tool.base_url = "${UNRESOLVED_HOST}"
        with pytest.raises(A2AInvocationError, match="unresolved base_url"):
            invoke_external_tool("call_test_agent", {"prompt": "test"})

        # JSON-RPC error
        tool.base_url = "https://test.example.com"
        mock_response = MagicMock()
        mock_response.json.return_value = {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Bad"}}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        with pytest.raises(A2AInvocationError, match="Bad"):
            invoke_external_tool("call_test_agent", {"prompt": "test"}, client=mock_client)
