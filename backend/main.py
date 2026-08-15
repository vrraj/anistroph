"""Anistroph FastAPI application — REST APIs and static UI.

All routes invoke the same core Anistroph services (backend.services).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.analysis import router as analysis_router
from backend.api.datasets import router as datasets_router
from backend.api.models import router as models_router
from backend.api.predictions import router as predictions_router
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
        version="0.1.0",
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
