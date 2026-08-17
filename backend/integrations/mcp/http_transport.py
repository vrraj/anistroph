"""MCP Streamable HTTP transport — exposes the same MCP tools over HTTP.

This adds an /mcp endpoint to the FastAPI app so that remote MCP clients
can discover and execute tools over HTTP (Streamable HTTP transport)
instead of launching a local stdio subprocess.

Both transports (stdio and HTTP) share the same tool handlers and call
the same AnistrophServices layer.
"""

from __future__ import annotations

import contextlib
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsRequest, ListToolsResult, TextContent, Tool as MCPTool

from backend.integrations.mcp.tools import TOOL_DEFS, call_tool

# Build a single MCP server instance with tool handlers.
_server = Server("anistroph")


async def _handle_list_tools(ctx: Any, params: Any) -> Any:
    """Return all Anistroph tools as MCP Tool objects."""
    tools = []
    for name, desc, schema in TOOL_DEFS:
        tools.append(MCPTool(
            name=name,
            description=desc,
            inputSchema=schema,
        ))
    return ListToolsResult(tools=tools)


async def _handle_call_tool(ctx: Any, params: Any) -> Any:
    """Execute a tool call and return the result."""
    result = await call_tool(params.name, dict(params.arguments or {}))
    content = [TextContent(type="text", text=item.text) for item in result]
    return CallToolResult(content=content)


_server.add_request_handler("tools/list", ListToolsRequest, _handle_list_tools)
_server.add_request_handler("tools/call", CallToolRequestParams, _handle_call_tool)


# Session manager handles multiple concurrent MCP sessions over HTTP.
_session_manager = StreamableHTTPSessionManager(
    app=_server,
    stateless=False,
)


async def handle_mcp_request(scope, receive, send):
    """ASGI handler for /mcp — delegates to the session manager."""
    await _session_manager.handle_request(scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan(app):
    """Manage MCP session manager lifecycle alongside the FastAPI app."""
    async with _session_manager.run():
        yield


def create_mcp_http_app():
    """Return an ASGI app for the MCP HTTP endpoint.

    This is mounted at /mcp on the FastAPI app. It delegates to the
    StreamableHTTPSessionManager which handles JSON-RPC 2.0 requests
    over HTTP (POST for requests, GET for SSE streams, DELETE to close).
    """
    async def asgi_app(scope, receive, send):
        await _session_manager.handle_request(scope, receive, send)

    return asgi_app
