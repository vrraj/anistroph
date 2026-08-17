"""Evaluation API routes — evaluate a trained model against the held-out
evaluation partition."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import EvaluateRequest, EvaluationSlicesRequest
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
        return svc.evaluate_model(model_id, req.sample_size, filters=req.filters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{model_id}/slices")
async def find_eval_slices(model_id: str, req: EvaluationSlicesRequest):
    """Find slices where model error deviates most from the overall baseline.

    Runs inference on the held-out evaluation partition and searches
    categorical dimension combinations for populations where the prediction
    error differs materially from the overall average.
    """
    svc = get_services()
    try:
        return svc.find_evaluation_slices(
            model_id,
            metric=req.metric,
            dimensions=req.dimensions,
            min_sample_size=req.min_sample_size,
            max_dimensions=req.max_dimensions,
            top_k=req.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
