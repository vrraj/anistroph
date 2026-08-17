"""Evaluation API routes — evaluate a trained model against the held-out
evaluation partition."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import EvaluateRequest
from backend.services import get_services

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("/{model_id}")
async def evaluate_model(model_id: str, req: EvaluateRequest):
    """Evaluate a trained model against the dataset's held-out evaluation set.

    Runs inference on evaluation.parquet and compares predictions against
    known actual target values. Returns aggregate metrics and a sample of
    prediction-vs-actual rows.
    """
    svc = get_services()
    try:
        return svc.evaluate_model(model_id, req.sample_size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
