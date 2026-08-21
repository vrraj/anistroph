"""Integrations API routes — external tool invocation.

Exposes the same registered external capabilities as MCP, through a shared
external-tool invoker. REST and MCP resolve the same tool definition and
invoke the same underlying A2A service.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.integrations.a2a import (
    AGENT_UNAVAILABLE_MESSAGE,
    A2AInvocationError,
    invoke_external_tool,
    validate_arguments,
)
from backend.integrations.registry import get_external_tool_registry

router = APIRouter(prefix="/integrations", tags=["integrations"])


class InvokeRequest(BaseModel):
    """Request body for POST /integrations/tools/{tool_name}/invoke."""
    arguments: dict = {}


@router.get("/tools")
async def list_external_tools():
    """List all externally-registered tools visible to REST."""
    registry = get_external_tool_registry()
    return [
        {
            "name": t.name,
            "provider": t.provider,
            "capability": t.capability,
            "description": t.description,
            "protocol": t.protocol,
            "agent_owner": t.agent_owner,
            "llm_parameters": t.llm_parameters,
        }
        for t in registry.list_rest_visible()
    ]


@router.post("/tools/{tool_name}/invoke")
async def invoke_tool(tool_name: str, req: InvokeRequest):
    """Invoke a registered external tool via the shared A2A invoker.

    Request parameters are validated against the tool's ``llm_parameters``
    schema before invocation.
    """
    registry = get_external_tool_registry()
    tool = registry.get(tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"external tool {tool_name!r} not found")
    if not tool.is_rest_visible:
        raise HTTPException(status_code=403, detail=f"tool {tool_name!r} is not REST-visible")

    # Validate arguments.
    errors = validate_arguments(tool, req.arguments)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})

    try:
        result = invoke_external_tool(tool_name, req.arguments)
        return result
    except A2AInvocationError as e:
        if e.connection_error:
            # Soft-fail: return a 200 with an unavailable message so the
            # calling agent can proceed without the RAG response.
            return {
                "state": "unavailable",
                "message": AGENT_UNAVAILABLE_MESSAGE,
            }
        raise HTTPException(status_code=502, detail=str(e))
