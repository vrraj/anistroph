"""Anistroph FastAPI application — REST APIs, MCP HTTP, and static UI.

All routes invoke the same core Anistroph services (backend.services).
The /mcp endpoint exposes MCP tools over Streamable HTTP transport so
remote MCP clients can discover and execute tools without a stdio subprocess.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.analysis import router as analysis_router
from backend.api.datasets import router as datasets_router
from backend.api.evaluations import router as evaluations_router
from backend.api.integrations import router as integrations_router
from backend.api.models import router as models_router
from backend.api.predictions import router as predictions_router
from backend.api.search import router as search_router
from backend.integrations.mcp.http_transport import create_mcp_http_app, lifespan as mcp_lifespan
from backend.schemas.api import HealthResponse
from backend.services import get_services

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIR = _REPO_ROOT / "frontend"

# Paths excluded from the GPT Action OpenAPI spec (runtime-only, no admin ops).
# Matches MCP scope: training and dataset registration are not exposed.
_GPT_EXCLUDED_PATHS = {"/models/train", "/datasets"}


def create_app() -> FastAPI:
    app = FastAPI(
        title="Anistroph",
        description="Extensible, domain-agnostic predictive analytics platform",
        version="1.0.0",
        lifespan=mcp_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health.
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health():
        return HealthResponse()

    # Routers.
    app.include_router(datasets_router)
    app.include_router(analysis_router)
    app.include_router(models_router)
    app.include_router(predictions_router)
    app.include_router(evaluations_router)
    app.include_router(search_router)
    app.include_router(integrations_router)

    # MCP Streamable HTTP transport — exposes all MCP tools (native +
    # external) over HTTP at /mcp so remote clients can discover
    # (tools/list) and execute (tools/call) without a stdio subprocess.
    # Uses JSON-RPC 2.0, same as the stdio transport. Handled via a raw
    # ASGI middleware (not FastAPI routes) because MCP manages its own
    # request/response cycle, including SSE streaming. Both /mcp and /mcp/
    # are handled to avoid 307 redirects that break MCP clients.
    mcp_handler = create_mcp_http_app()

    class MCPASGIMiddleware:
        """Raw ASGI middleware that intercepts /mcp and /mcp/ and delegates
        to the MCP Streamable HTTP handler. All other paths pass through."""
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                path = scope.get("path", "")
                if path == "/mcp" or path == "/mcp/":
                    # Normalize path for the MCP session manager.
                    scope = dict(scope)
                    scope["path"] = "/mcp"
                    scope["raw_path"] = b"/mcp"
                    await mcp_handler(scope, receive, send)
                    return
            await self.app(scope, receive, send)

    # Add the middleware so it runs before routing.
    app.add_middleware(MCPASGIMiddleware)

    # Filtered OpenAPI spec for ChatGPT GPT Actions — runtime endpoints only.
    # Excludes training and dataset registration (admin operations), matching
    # the MCP tool scope.
    @app.get("/openapi-gpt.json", include_in_schema=False)
    async def openapi_gpt():
        spec = app.openapi()
        filtered_paths = {}
        for path, methods in spec.get("paths", {}).items():
            if path in _GPT_EXCLUDED_PATHS:
                # Keep GET but drop POST (e.g. GET /datasets is fine, POST /datasets is admin).
                filtered_methods = {
                    m: v for m, v in methods.items() if m != "post"
                }
                if filtered_methods:
                    filtered_paths[path] = filtered_methods
            else:
                filtered_paths[path] = methods
        spec["paths"] = filtered_paths
        spec["info"]["title"] = "Anistroph (Runtime — GPT Action)"
        spec["info"]["description"] = (
            "Runtime prediction, explanation, and analysis endpoints for "
            "ChatGPT GPT Actions. Training and dataset registration are "
            "not exposed."
        )
        return JSONResponse(spec)

    # Static UI.
    if _FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def index():
            index_path = _FRONTEND_DIR / "index.html"
            if index_path.exists():
                return index_path.read_text()
            return "<h1>Anistroph</h1><p>UI not yet built.</p>"

    return app


app = create_app()
