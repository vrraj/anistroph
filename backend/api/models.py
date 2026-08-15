"""Model API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import TrainRequest
from backend.services import get_services

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/train")
async def train(req: TrainRequest):
    """Train a model."""
    svc = get_services()
    try:
        return svc.train(
            req.dataset_id, req.target_name, req.model_type,
            req.model_parameters, req.model_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
async def list_models():
    """List all registered models."""
    svc = get_services()
    return [m.model_dump() for m in svc.list_models()]


@router.get("/types")
async def list_model_types():
    """List available model types."""
    svc = get_services()
    return {"model_types": svc.list_model_types()}


@router.get("/{model_id}")
async def get_model(model_id: str):
    """Get a model's metadata."""
    svc = get_services()
    meta = svc.get_model(model_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")
    return meta.model_dump()


@router.get("/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get a model's evaluation metrics."""
    svc = get_services()
    try:
        return svc.get_model_metrics(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
