"""Shared external-tool invoker — A2A JSON-RPC client.

Both MCP and REST call this same invoker to reach external A2A agents
such as Aina-Veris. No Aina-Veris-specific logic lives here — the invoker
is generic and driven by the external tool registry.

A2A protocol reference: https://github.com/google-a2a/a2a
The invoker sends a JSON-RPC 2.0 ``tasks/send`` request to the agent's
endpoint and returns the resulting task state / artifacts.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

import httpx

from backend.integrations.registry import ExternalToolDef, get_external_tool_registry

logger = logging.getLogger(__name__)

# A2A JSON-RPC method for sending a task to an agent.
A2A_TASK_SEND_METHOD = "tasks/send"

# Default timeout for A2A requests (seconds).
DEFAULT_TIMEOUT = 120


class A2AInvocationError(Exception):
    """Raised when an A2A invocation fails."""


def _build_task_params(prompt: str, **extra: Any) -> dict[str, Any]:
    """Build the A2A task/send parameters for a text prompt."""
    return {
        "id": str(uuid.uuid4()),
        "message": {
            "role": "user",
            "parts": [
                {"type": "text", "text": prompt},
            ],
        },
        **extra,
    }


def _build_jsonrpc_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request envelope."""
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }


def invoke_external_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    client: Optional[httpx.Client] = None,
) -> dict[str, Any]:
    """Invoke a registered external tool via A2A JSON-RPC.

    Args:
        tool_name: the registered tool name (e.g. call_veris_semiconductor_research_agent).
        arguments: the tool arguments, validated against the tool's llm_parameters schema.
        timeout: HTTP timeout in seconds.
        client: optional pre-configured httpx.Client (for testing).

    Returns:
        The A2A task response as a dict. Typically contains:
        - ``id``: the task ID
        - ``state``: task state (e.g. "completed", "working", "failed")
        - ``artifacts``: list of result artifacts
        - ``messages``: list of messages from the agent

    Raises:
        A2AInvocationError: if the tool is not found, the request fails,
            or the response indicates an error.
    """
    registry = get_external_tool_registry()
    tool = registry.get(tool_name)
    if tool is None:
        raise A2AInvocationError(f"external tool {tool_name!r} not found in registry")

    if tool.protocol != "A2A_JSONRPC":
        raise A2AInvocationError(
            f"tool {tool_name!r} uses protocol {tool.protocol!r}; "
            "only A2A_JSONRPC is supported"
        )

    # The prompt is the primary argument for A2A text-based agents.
    prompt = arguments.get("prompt", "")
    if not prompt:
        raise A2AInvocationError(
            f"tool {tool_name!r} requires a 'prompt' argument"
        )

    # Build the A2A JSON-RPC request.
    task_params = _build_task_params(prompt)
    request_body = _build_jsonrpc_request(A2A_TASK_SEND_METHOD, task_params)

    url = tool.resolved_url
    if not tool.base_url or "${" in tool.base_url:
        raise A2AInvocationError(
            f"tool {tool_name!r} has an unresolved base_url "
            f"({tool.base_url!r}); set the VERIS_BASE_URL environment variable"
        )

    logger.info("A2A invoke: tool=%s url=%s", tool_name, url)

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=timeout)

    try:
        response = client.post(
            url,
            json=request_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
        result = response.json()

        # Check for JSON-RPC error.
        if "error" in result:
            err = result["error"]
            raise A2AInvocationError(
                f"A2A agent returned error: code={err.get('code')} "
                f"message={err.get('message')}"
            )

        # Return the result field (the A2A task object).
        return result.get("result", result)
    except httpx.HTTPError as e:
        raise A2AInvocationError(f"A2A HTTP request failed: {e}") from e
    finally:
        if own_client:
            client.close()


def validate_arguments(tool: ExternalToolDef, arguments: dict[str, Any]) -> list[str]:
    """Validate arguments against the tool's llm_parameters schema.

    Returns a list of validation error messages (empty if valid).
    """
    errors: list[str] = []
    schema = tool.llm_parameters
    if not schema:
        return errors

    required = schema.get("required", [])
    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)

    # Check required fields.
    for req in required:
        if req not in arguments:
            errors.append(f"missing required parameter: {req}")

    # Check for unknown properties.
    if additional is False:
        for key in arguments:
            if key not in properties:
                errors.append(f"unknown parameter: {key}")

    # Check types (basic).
    type_map = {"string": str, "number": (int, float), "integer": int,
                "boolean": bool, "array": list, "object": dict}
    for key, value in arguments.items():
        if key in properties:
            expected_type = properties[key].get("type")
            if expected_type and expected_type in type_map:
                py_type = type_map[expected_type]
                if not isinstance(value, py_type):
                    errors.append(
                        f"parameter {key!r} must be {expected_type}, "
                        f"got {type(value).__name__}"
                    )

    return errors
