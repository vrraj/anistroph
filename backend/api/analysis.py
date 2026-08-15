"""Analysis API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import CompareRequest, SliceRequest
from backend.services import get_services

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/slice")
async def slice(req: SliceRequest):
    """Slice a dataset by dimensions with an aggregation."""
    svc = get_services()
    try:
        return svc.slice(
            req.dataset_id, req.dimensions, req.metric,
            req.aggregation, req.filters, req.limit
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/compare")
async def compare(req: CompareRequest):
    """Compare a metric across values of a dimension."""
    svc = get_services()
    try:
        return svc.compare(
            req.dataset_id, req.dimension, req.metric,
            req.aggregation, req.filters
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
