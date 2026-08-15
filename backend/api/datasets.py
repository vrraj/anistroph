"""Dataset API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import RegisterDatasetRequest
from backend.services import get_services

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("")
async def register_dataset(req: RegisterDatasetRequest):
    """Register a dataset from a config YAML + source data file."""
    svc = get_services()
    try:
        meta = svc.register_dataset_from_config(
            req.config_path, req.source_path, req.parquet_path
        )
        return meta.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
async def list_datasets():
    """List all registered datasets."""
    svc = get_services()
    return [m.model_dump() for m in svc.list_datasets()]


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get a registered dataset's metadata."""
    svc = get_services()
    meta = svc.get_dataset(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id!r} not found")
    return meta.model_dump()


@router.get("/{dataset_id}/profile")
async def profile_dataset(dataset_id: str):
    """Profile a dataset."""
    svc = get_services()
    try:
        return svc.profile(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
