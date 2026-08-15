"""MCP server — exposes Anistroph capabilities through the Model Context Protocol.

Uses the low-level mcp.server.Server API with stdio transport.
All tools call the same core services as REST — no separate analytical logic.
"""

from __future__ import annotations

import asyncio
from typing import Any

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server

from backend.integrations.mcp.tools import call_tool, get_tool_list

server = Server("anistroph")


def _handle_list_tools(params: types.ListToolsRequest) -> types.ListToolsResult:
    return types.ListToolsResult(tools=get_tool_list())


async def _handle_call_tool(params: types.CallToolRequest) -> types.CallToolResult:
    result = await call_tool(params.name, dict(params.arguments or {}))
    return types.CallToolResult(content=result)


server.add_request_handler("tools/list", types.ListToolsRequest, _handle_list_tools)
server.add_request_handler("tools/call", types.CallToolRequest, _handle_call_tool)


async def run() -> None:
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(NotificationOptions()),
        )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
