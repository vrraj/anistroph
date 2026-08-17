"""MCP tool definitions — deterministic Anistroph capabilities exposed via MCP.

Each tool calls the same core services as REST. No separate analytical logic
lives inside MCP. No arbitrary Python execution is exposed.
"""

from __future__ import annotations

import json
from typing import Any

import mcp.types as types

from backend.services import get_services

# Each tool definition: (name, description, input_schema, handler)
TOOL_DEFS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "anistroph_list_datasets",
        "List all registered datasets in Anistroph.",
        {"type": "object", "properties": {}, "required": []},
    ),
    (
        "anistroph_profile_dataset",
        "Profile a registered dataset (row count, column types, distributions, time range, event distribution).",
        {
            "type": "object",
            "properties": {"dataset_id": {"type": "string"}},
            "required": ["dataset_id"],
        },
    ),
    (
        "anistroph_slice_data",
        "Slice a dataset by dimensions with an aggregation over a metric.",
        {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "dimensions": {"type": "array", "items": {"type": "string"}},
                "metric": {"type": "string"},
                "aggregation": {"type": "string", "default": "mean"},
                "filters": {"type": "object"},
                "limit": {"type": "integer"},
            },
            "required": ["dataset_id", "dimensions", "metric"],
        },
    ),
    (
        "anistroph_compare_data",
        "Compare a metric across values of a dimension.",
        {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "dimension": {"type": "string"},
                "metric": {"type": "string"},
                "aggregation": {"type": "string", "default": "mean"},
                "filters": {"type": "object"},
            },
            "required": ["dataset_id", "dimension", "metric"],
        },
    ),
    (
        "anistroph_find_interesting_slices",
        "Find slices with the largest deviation from the overall metric baseline. Searches 1, 2, and 3-dimensional combinations of categorical columns.",
        {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "metric": {"type": "string"},
                "dimensions": {"type": "array", "items": {"type": "string"}},
                "min_sample_size": {"type": "integer", "default": 100},
                "max_dimensions": {"type": "integer", "default": 3},
                "aggregation": {"type": "string", "default": "mean"},
                "filters": {"type": "object"},
                "top_k": {"type": "integer", "default": 20},
            },
            "required": ["dataset_id", "metric"],
        },
    ),
    (
        "anistroph_list_models",
        "List all registered trained models in Anistroph.",
        {"type": "object", "properties": {}, "required": []},
    ),
    (
        "anistroph_get_model_metrics",
        "Get the evaluation metrics for a trained model.",
        {
            "type": "object",
            "properties": {"model_id": {"type": "string"}},
            "required": ["model_id"],
        },
    ),
    (
        "anistroph_get_model_inputs",
        "Get the prediction input schema for a trained model — what the caller must supply to predict. Returns the prediction mode (entity_id lookup vs records), the entity_key, whether a timestamp is required, and (for records-based prediction) the list of required source columns with their types and transforms. Use this before calling anistroph_predict to discover what inputs a model expects.",
        {
            "type": "object",
            "properties": {"model_id": {"type": "string"}},
            "required": ["model_id"],
        },
    ),
    (
        "anistroph_predict",
        "Make a prediction using a trained model. For temporal datasets, provide entity_id and timestamp. For non-temporal datasets, provide records.",
        {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "entity_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "records": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["model_id"],
        },
    ),
    (
        "anistroph_explain_prediction",
        "Explain a prediction using SHAP TreeExplainer (for XGBoost models) or importance-weighted contributions. Returns top_positive (features that increase the prediction) and top_negative (features that decrease it), plus a combined top_drivers list. For temporal datasets, provide entity_id and timestamp. For non-temporal datasets, provide entity_id.",
        {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "entity_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "records": {"type": "array", "items": {"type": "object"}},
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["model_id"],
        },
    ),
    (
        "anistroph_sample_rows",
        "Return up to n raw rows from a registered dataset, optionally filtered by column values, with an optional column subset and sort. Use this to inspect individual records (e.g. a specific wafer_id) rather than aggregations. n is capped at 1000.",
        {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "n": {"type": "integer", "default": 10},
                "filters": {"type": "object", "description": "Equality filters, e.g. {\"wafer_id\": \"WAFER_015000\"} or {\"etch_tool\": [\"ETCH_02\", \"ETCH_03\"]} for IN-style."},
                "columns": {"type": "array", "items": {"type": "string"}},
                "sort_by": {"type": "string"},
                "descending": {"type": "boolean", "default": False},
            },
            "required": ["dataset_id"],
        },
    ),
    (
        "anistroph_evaluate_model",
        "Evaluate a trained model against the dataset's held-out evaluation partition. Loads evaluation.parquet, runs inference using the persisted model, and compares predictions against known actual target values. Returns aggregate metrics (MAE/MSE/RMSE/R2/MAPE/max_error for regression, AUC/precision/recall/F1 for classification) and a sample of prediction-vs-actual rows. The evaluation set is never used during training. Optional filters allow slice-level evaluation (e.g. metrics for a single city, lot, or zip code) — when filters are provided, the response includes both overall metrics and filtered_metrics for comparison.",
        {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "sample_size": {"type": "integer", "default": 50, "description": "Number of prediction-vs-actual rows to return (capped at 1000). Aggregate metrics are always over the full evaluation set (or the filtered subset when filters are provided)."},
                "filters": {"type": "object", "description": "Optional equality filters for slice-level evaluation, e.g. {\"city\": \"Saratoga\"} or {\"lot_id\": [\"LOT_001\", \"LOT_002\"]}. When provided, the response includes both overall metrics and filtered_metrics for the matching rows.", "additionalProperties": True},
            },
            "required": ["model_id"],
        },
    ),
    (
        "anistroph_find_evaluation_slices",
        "Find populations where model prediction error deviates most from the overall average. Runs inference on the held-out evaluation partition, computes per-row error, and searches 1/2/3-dimensional combinations of categorical columns (e.g. city, etch_tool, product_id) for slices where the error metric differs materially from the overall baseline. This is the model-evaluation analogue of find_interesting_slices: instead of finding slices where the target deviates, it finds slices where the prediction error deviates — identifying populations where the model performs better or worse than average. Returns ranked slices with dimension values, row count, error metric value, overall baseline, and difference.",
        {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "metric": {"type": "string", "default": "abs_error", "description": "Error metric to aggregate. Regression: 'abs_error' (absolute error), 'error' (signed error — shows bias direction), 'pct_error' (percentage error — relative to actual value). Classification: 'log_loss' (per-row log loss)."},
                "min_sample_size": {"type": "integer", "default": 50, "description": "Minimum rows per slice to be considered. Slices with fewer rows are excluded."},
                "max_dimensions": {"type": "integer", "default": 3, "description": "Maximum number of dimensions to combine (1-3). Higher values search more combinations but take longer."},
                "top_k": {"type": "integer", "default": 20, "description": "Number of top slices to return, ranked by absolute difference from the overall error baseline."},
            },
            "required": ["model_id"],
        },
    ),
]


def get_tool_list() -> list[types.Tool]:
    """Return the list of MCP Tool objects."""
    tools = []
    for name, desc, schema in TOOL_DEFS:
        tools.append(
            types.Tool(
                name=name,
                description=desc,
                input_schema=schema,
            )
        )
    return tools


async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Dispatch a tool call to the appropriate core service."""
    svc = get_services()

    try:
        if name == "anistroph_list_datasets":
            result = [m.model_dump() for m in svc.list_datasets()]
        elif name == "anistroph_profile_dataset":
            result = svc.profile(arguments["dataset_id"])
        elif name == "anistroph_slice_data":
            result = svc.slice(
                arguments["dataset_id"],
                arguments["dimensions"],
                arguments["metric"],
                arguments.get("aggregation", "mean"),
                arguments.get("filters"),
                arguments.get("limit"),
            )
        elif name == "anistroph_compare_data":
            result = svc.compare(
                arguments["dataset_id"],
                arguments["dimension"],
                arguments["metric"],
                arguments.get("aggregation", "mean"),
                arguments.get("filters"),
            )
        elif name == "anistroph_find_interesting_slices":
            result = svc.find_interesting_slices(
                arguments["dataset_id"],
                arguments["metric"],
                arguments.get("dimensions"),
                arguments.get("min_sample_size", 100),
                arguments.get("max_dimensions", 3),
                arguments.get("aggregation", "mean"),
                arguments.get("filters"),
                arguments.get("top_k", 20),
            )
        elif name == "anistroph_list_models":
            # Return a compact summary — full model_dump() can be multi-MB
            # for classification models with large PR curves. Use
            # anistroph_get_model_metrics for full metrics.
            result = [
                {
                    "model_id": m.model_id,
                    "model_type": m.model_type,
                    "dataset_id": m.dataset_id,
                    "target_name": m.target_name,
                    "target_type": m.target_type,
                    "created_at": m.created_at,
                }
                for m in svc.list_models()
            ]
        elif name == "anistroph_get_model_metrics":
            result = svc.get_model_metrics(arguments["model_id"])
        elif name == "anistroph_get_model_inputs":
            result = svc.get_model_inputs(arguments["model_id"])
        elif name == "anistroph_predict":
            result = svc.predict(
                arguments["model_id"],
                arguments.get("entity_id"),
                arguments.get("timestamp"),
                arguments.get("records"),
            )
        elif name == "anistroph_explain_prediction":
            result = svc.explain(
                arguments["model_id"],
                arguments.get("entity_id"),
                arguments.get("timestamp"),
                arguments.get("records"),
                arguments.get("top_k", 10),
            )
        elif name == "anistroph_sample_rows":
            result = svc.sample_rows(
                arguments["dataset_id"],
                arguments.get("n", 10),
                arguments.get("filters"),
                arguments.get("columns"),
                arguments.get("sort_by"),
                arguments.get("descending", False),
            )
        elif name == "anistroph_evaluate_model":
            result = svc.evaluate_model(
                arguments["model_id"],
                arguments.get("sample_size", 50),
                filters=arguments.get("filters"),
            )
        elif name == "anistroph_find_evaluation_slices":
            result = svc.find_evaluation_slices(
                arguments["model_id"],
                metric=arguments.get("metric", "abs_error"),
                min_sample_size=arguments.get("min_sample_size", 50),
                max_dimensions=arguments.get("max_dimensions", 3),
                top_k=arguments.get("top_k", 20),
            )
        else:
            return [types.TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]

        return [types.TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]
