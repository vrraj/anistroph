"""Prediction API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import ExplainRequest, PredictRequest
from backend.services import get_services

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("")
async def predict(req: PredictRequest):
    """Make a prediction."""
    svc = get_services()
    try:
        return svc.predict(req.model_id, req.entity_id, req.timestamp, req.records)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/batch")
async def predict_batch(reqs: list[PredictRequest]):
    """Make batch predictions."""
    svc = get_services()
    results = []
    for req in reqs:
        try:
            results.append(svc.predict(req.model_id, req.entity_id, req.timestamp, req.records))
        except ValueError as e:
            results.append({"error": str(e), "model_id": req.model_id})
    return results


@router.post("/explain")
async def explain(req: ExplainRequest):
    """Explain a prediction."""
    svc = get_services()
    try:
        return svc.explain(req.model_id, req.entity_id, req.timestamp, req.records, req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
