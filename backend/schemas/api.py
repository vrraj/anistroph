"""API request/response schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class RegisterDatasetRequest(BaseModel):
    config_path: str
    source_path: str
    parquet_path: Optional[str] = None


class SliceRequest(BaseModel):
    dataset_id: str
    dimensions: list[str]
    metric: str
    aggregation: str = "mean"
    filters: Optional[dict[str, Any]] = None
    limit: Optional[int] = None


class CompareRequest(BaseModel):
    dataset_id: str
    dimension: str
    metric: str
    aggregation: str = "mean"
    filters: Optional[dict[str, Any]] = None


class TrainRequest(BaseModel):
    dataset_id: str
    target_name: str
    model_type: str
    model_parameters: Optional[dict[str, Any]] = None
    model_id: Optional[str] = None


class PredictRequest(BaseModel):
    model_id: str
    entity_id: Optional[str] = None
    timestamp: Optional[str] = None
    records: Optional[list[dict[str, Any]]] = None


class ExplainRequest(BaseModel):
    model_id: str
    entity_id: Optional[str] = None
    timestamp: Optional[str] = None
    records: Optional[list[dict[str, Any]]] = None
    top_k: int = 10


class SampleRowsRequest(BaseModel):
    n: int = 25
    filters: Optional[dict[str, Any]] = None
    columns: Optional[list[str]] = None
    sort_by: Optional[str] = None
    descending: bool = False


class EvaluateRequest(BaseModel):
    sample_size: int = 50
