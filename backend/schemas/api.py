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


class InterestingSlicesRequest(BaseModel):
    dataset_id: str
    metric: str
    dimensions: Optional[list[str]] = None
    min_sample_size: int = 100
    max_dimensions: int = 3
    aggregation: str = "mean"
    filters: Optional[dict[str, Any]] = None
    top_k: int = 20


class TrainRequest(BaseModel):
    dataset_id: str
    target_name: str
    model_type: Optional[str] = None  # auto-selected from task_type if omitted
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


class FilterExpressionRequest(BaseModel):
    """A single structured filter for parametric search."""
    field: str
    op: str  # eq, in, gte, lte, between, contains_range
    value: Optional[Any] = None
    min_field: Optional[str] = None  # for contains_range
    max_field: Optional[str] = None  # for contains_range
    low: Optional[float] = None  # for between
    high: Optional[float] = None  # for between


class SortExpressionRequest(BaseModel):
    """A single sort directive for parametric search."""
    field: str
    descending: bool = False


class SearchRequest(BaseModel):
    """Request body for POST /datasets/{dataset_id}/search."""
    filters: list[FilterExpressionRequest] = Field(default_factory=list)
    sort: Optional[list[SortExpressionRequest]] = None
    limit: int = 50
    columns: Optional[list[str]] = None


class PredictOnSearchRequest(BaseModel):
    """Request body for POST /datasets/{dataset_id}/predict-on-search."""
    model_id: str
    filters: list[FilterExpressionRequest] = Field(default_factory=list)
    sort: Optional[list[SortExpressionRequest]] = None
    limit: int = 50
    columns: Optional[list[str]] = None
    timestamp: Optional[str] = None


class EvaluateRequest(BaseModel):
    sample_size: int = 50
    filters: Optional[dict[str, Any]] = None


class EvaluationSlicesRequest(BaseModel):
    metric: str = "abs_error"
    dimensions: Optional[list[str]] = None
    min_sample_size: int = 50
    max_dimensions: int = 3
    top_k: int = 20
