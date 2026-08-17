"""Evaluation runner — evaluate a persisted model against a held-out
evaluation partition.

This module loads ``evaluation.parquet`` (produced at dataset registration
time), builds features using the persisted FeatureMetadata (no refit), runs
inference with the persisted model, and compares predictions against the
known actual target values.

The evaluation file is never used during model fitting — training loads only
``train.parquet``. This module is the runtime / MCP path for post-training
evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import polars as pl

from backend.datasets.config import DatasetConfig
from backend.features.engine import FeatureEngine
from backend.ml.base import Predictor
from backend.ml.evaluation import evaluate_binary, evaluate_regression
from backend.ml.inference import _load_predictor
from backend.ml.registry import ModelRegistry
from backend.targets.engine import TargetEngine
from backend.targets.spec import TargetType


def _apply_filters(df: pl.DataFrame, filters: dict[str, Any]) -> pl.DataFrame:
    """Apply equality / IN-style filters to a DataFrame.

    Mirrors the filter logic in ``services.sample_rows``.
    """
    for col, val in filters.items():
        if col not in df.columns:
            raise ValueError(f"unknown filter column {col!r}")
        if isinstance(val, list):
            df = df.filter(pl.col(col).is_in(val))
        else:
            df = df.filter(pl.col(col) == val)
    return df


def _run_inference_on_eval_set(
    model_id: str,
    model_registry: ModelRegistry,
    config: DatasetConfig,
    eval_parquet_path: str,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, bool, float]:
    """Load the eval partition, build features, and run inference.

    Returns:
        df: the original eval DataFrame (with target built) — has original
            categorical columns like ``city``, ``etch_tool``, etc.
        y_actual: ground-truth target values.
        y_pred: model predictions (regression: predicted values;
            classification: predicted probabilities of positive class).
        is_regression: whether the target is a regression target.
        threshold: decision threshold (for classification).
    """
    meta = model_registry.get(model_id)
    if meta is None:
        raise ValueError(f"model {model_id!r} not found")

    spec = config.dataset_spec
    feature_spec = model_registry.load_feature_spec(model_id)
    feature_metadata = model_registry.load_feature_metadata(model_id)
    target_spec = model_registry.load_target_spec(model_id)

    predictor = _load_predictor(meta.model_type, meta.artifact_path)
    feature_cols = feature_metadata.feature_names

    # Load the imputer persisted during training.
    imputer_path = Path(meta.artifact_path) / "imputer.joblib"
    imputer = None
    if imputer_path.exists():
        imputer = joblib.load(str(imputer_path))

    # --- Load evaluation data ---
    df = pl.read_parquet(eval_parquet_path)

    # --- Build target ---
    target_engine = TargetEngine()
    df = target_engine.build_target(df, spec, target_spec)

    # --- Build features (no refit — use persisted metadata) ---
    feature_engine = FeatureEngine()
    feat_df, _ = feature_engine.build_features(
        df, spec, feature_spec, metadata=feature_metadata, fit=False,
    )

    # Join features with target.
    target_col = target_spec.name
    keys = [spec.entity_key] + ([spec.time_key] if spec.time_key else [])
    full = feat_df.join(
        df.select(keys + [target_col]), on=keys, how="left",
    )

    X = full.select(feature_cols).to_numpy()
    y_actual = full[target_col].to_numpy()

    if imputer is not None:
        X = imputer.transform(X)

    is_regression = target_spec.type in (TargetType.REGRESSION,)
    threshold = meta.decision_threshold

    # --- Run inference ---
    if is_regression:
        y_pred = predictor.predict(X)
    else:
        y_pred = predictor.predict_proba(X)[:, 1]

    return df, y_actual, y_pred, is_regression, threshold


def evaluate_on_eval_set(
    model_id: str,
    model_registry: ModelRegistry,
    config: DatasetConfig,
    eval_parquet_path: str,
    sample_size: int = 50,
    filters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate a model against the held-out evaluation partition.

    Args:
        model_id: registered model to evaluate.
        model_registry: model registry (for loading model artifacts).
        config: DatasetConfig for the model's dataset.
        eval_parquet_path: path to the evaluation Parquet file.
        sample_size: number of prediction-vs-actual rows to include in the
            response (capped at 1000). Aggregate metrics are over the full set
            (or the filtered set when ``filters`` is provided).
        filters: optional equality / IN-style filters, e.g.
            ``{"city": "Saratoga"}`` or ``{"lot_id": ["LOT_001", "LOT_002"]}``.
            When provided, the response includes both ``metrics`` (overall,
            all eval rows) and ``filtered_metrics`` (filtered subset only),
            plus ``filtered_row_count`` and a prediction-vs-actual sample drawn
            from the filtered subset.

    Returns a dict with:
        - model_id, dataset_id, target_name, target_type
        - eval_row_count: number of rows in the full evaluation set
        - metrics: aggregate evaluation metrics over the full eval set
        - filtered_metrics: aggregate metrics over the filtered subset
            (only present when ``filters`` is provided)
        - filtered_row_count: number of rows matching the filters
            (only present when ``filters`` is provided)
        - filters: the filters applied (only present when provided)
        - predictions_sample: list of {entity_id, actual, predicted, ...} rows
          (from the filtered subset when filters are applied, otherwise from
          the full set)
    """
    meta = model_registry.get(model_id)
    if meta is None:
        raise ValueError(f"model {model_id!r} not found")

    spec = config.dataset_spec
    target_spec = model_registry.load_target_spec(model_id)
    target_col = target_spec.name

    # --- Run inference on the eval set (reusable helper) ---
    df, y_actual, y_pred, is_regression, threshold = _run_inference_on_eval_set(
        model_id, model_registry, config, eval_parquet_path,
    )

    # --- Overall metrics (full eval set) ---
    if is_regression:
        baseline = float(np.mean(y_actual))
        metrics = evaluate_regression(y_actual, y_pred, baseline_pred=baseline)
    else:
        metrics = evaluate_binary(y_actual, y_pred, threshold=threshold)

    # --- Determine which rows to use for the sample + filtered metrics ---
    if filters:
        # Filters apply to the ORIGINAL columns (e.g. "city", "lot_id"),
        # which are in `df` but not in the feature-engineered `full`.
        # Strategy: filter `df` to get matching row indices.
        df_indexed = df.with_row_index("_row_idx")
        df_filtered = _apply_filters(df_indexed, filters)
        filtered_pos = df_filtered["_row_idx"].to_list()

        if not filtered_pos:
            raise ValueError(
                f"no evaluation rows match filters {filters!r}"
            )

        y_actual_f = y_actual[filtered_pos]
        if is_regression:
            y_pred_f = y_pred[filtered_pos]
            baseline_f = float(np.mean(y_actual_f))
            filtered_metrics = evaluate_regression(
                y_actual_f, y_pred_f, baseline_pred=baseline_f,
            )
            pred_values = y_pred_f.tolist()
            actual_values = y_actual_f.tolist()
        else:
            y_proba_f = y_pred[filtered_pos]
            filtered_metrics = evaluate_binary(
                y_actual_f, y_proba_f, threshold=threshold,
            )
            pred_values = y_proba_f.tolist()
            actual_values = y_actual_f.tolist()

        # Sample rows come from the filtered subset.
        entity_col = spec.entity_key
        time_col = spec.time_key
        keys = [entity_col] + ([time_col] if time_col else [])
        sample_source = df_filtered.select(keys)
        filtered_row_count = len(filtered_pos)
    else:
        filtered_metrics = None
        filtered_row_count = None
        pred_values = y_pred.tolist()
        actual_values = y_actual.tolist()
        entity_col = spec.entity_key
        time_col = spec.time_key
        keys = [entity_col] + ([time_col] if time_col else [])
        sample_source = df.select(keys)

    # --- Build prediction-vs-actual sample ---
    sample_size = max(1, min(int(sample_size), 1000))
    entity_col = spec.entity_key
    time_col = spec.time_key

    # Build a sample of rows with entity, (timestamp), actual, predicted.
    sample_rows: list[dict[str, Any]] = []
    n = len(pred_values)
    for i in range(min(sample_size, n)):
        row: dict[str, Any] = {
            "entity_id": str(sample_source[entity_col][i]) if entity_col in sample_source.columns else None,
            "actual": float(actual_values[i]),
            "predicted": float(pred_values[i]),
        }
        if is_regression:
            row["error"] = float(pred_values[i] - actual_values[i])
            row["abs_error"] = abs(row["error"])
        else:
            row["predicted_label"] = int(pred_values[i] >= threshold)
        if time_col and time_col in sample_source.columns:
            row["timestamp"] = str(sample_source[time_col][i])
        sample_rows.append(row)

    result: dict[str, Any] = {
        "model_id": model_id,
        "dataset_id": meta.dataset_id,
        "target_name": target_spec.name,
        "target_type": target_spec.type.value,
        "eval_row_count": df.height,
        "metrics": metrics,
        "predictions_sample": sample_rows,
    }

    if filters is not None:
        result["filtered_metrics"] = filtered_metrics
        result["filtered_row_count"] = filtered_row_count
        result["filters"] = filters

    return result


def find_evaluation_slices(
    model_id: str,
    model_registry: ModelRegistry,
    config: DatasetConfig,
    eval_parquet_path: str,
    metric: str = "abs_error",
    dimensions: Optional[list[str]] = None,
    min_sample_size: int = 50,
    max_dimensions: int = 3,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Find slices where model error deviates most from the overall baseline.

    Runs inference on the held-out evaluation partition, computes per-row
    error, joins it with the original categorical columns, and searches
    1/2/3-dimensional combinations for slices where the error metric differs
    materially from the overall average.

    This is the evaluation analogue of ``find_interesting_slices``: instead
    of finding slices where the *target* deviates, it finds slices where the
    *prediction error* deviates — identifying populations where the model
    performs better or worse than average.

    Args:
        model_id: registered model to evaluate.
        model_registry: model registry (for loading model artifacts).
        config: DatasetConfig for the model's dataset.
        eval_parquet_path: path to the evaluation Parquet file.
        metric: error metric to aggregate. One of:
            - ``"abs_error"``: absolute error (default, regression)
            - ``"error"``: signed error (regression — shows bias direction)
            - ``"pct_error"``: percentage error (regression — relative)
            - ``"log_loss"``: per-row log loss (classification)
        dimensions: categorical columns to combine. If None, auto-detect
            categorical columns from the dataset spec.
        min_sample_size: minimum rows per slice to be considered.
        max_dimensions: max number of dimensions to combine (1-3).
        top_k: number of top slices to return.

    Returns a list of slices sorted by absolute difference from the overall
    error baseline, each with: dimensions, values, row_count, metric_value,
    overall_baseline, difference, abs_difference.
    """
    from itertools import combinations

    meta = model_registry.get(model_id)
    if meta is None:
        raise ValueError(f"model {model_id!r} not found")

    spec = config.dataset_spec
    target_spec = model_registry.load_target_spec(model_id)
    target_col = target_spec.name

    # --- Run inference on the eval set ---
    df, y_actual, y_pred, is_regression, threshold = _run_inference_on_eval_set(
        model_id, model_registry, config, eval_parquet_path,
    )

    # --- Compute per-row error metric ---
    if is_regression:
        abs_errors = np.abs(y_pred - y_actual)
        signed_errors = y_pred - y_actual
        # Percentage error (guard against zero actuals).
        non_zero = y_actual != 0
        pct_errors = np.zeros_like(y_actual, dtype=float)
        if non_zero.any():
            pct_errors[non_zero] = (
                np.abs(y_pred[non_zero] - y_actual[non_zero])
                / np.abs(y_actual[non_zero])
            ) * 100.0

        error_map = {
            "abs_error": abs_errors,
            "error": signed_errors,
            "pct_error": pct_errors,
        }
    else:
        # Classification: per-row log loss.
        eps = 1e-15
        y_proba_clipped = np.clip(y_pred, eps, 1 - eps)
        log_losses = -(y_actual * np.log(y_proba_clipped)
                       + (1 - y_actual) * np.log(1 - y_proba_clipped))
        error_map = {"log_loss": log_losses}

    if metric not in error_map:
        raise ValueError(
            f"unsupported metric {metric!r}; supported: {list(error_map.keys())}"
        )

    error_values = error_map[metric]

    # --- Auto-detect categorical columns if not specified ---
    if dimensions is None:
        # Use columns declared as categorical features in the spec.
        dimensions = [
            c.name for c in spec.columns.values()
            if c.type.value == "categorical" and c.role.value == "feature"
        ]

    if not dimensions:
        return []

    # Filter dimensions to those actually present in the eval DataFrame.
    dimensions = [d for d in dimensions if d in df.columns]
    if not dimensions:
        return []

    # --- Build a DataFrame with original categorical columns + error ---
    # `df` has the original columns (city, etch_tool, etc.) and is in the
    # same row order as y_actual / y_pred.
    keys = [spec.entity_key] + ([spec.time_key] if spec.time_key else [])
    error_df = df.select(keys + dimensions).with_columns(
        pl.Series("_error", error_values),
    )

    # --- Compute overall error baseline ---
    overall_error = float(np.mean(error_values))

    # --- Search 1, 2, and 3-dimensional combinations ---
    results: list[dict[str, Any]] = []
    for n_dims in range(1, min(max_dimensions, len(dimensions)) + 1):
        for combo in combinations(dimensions, n_dims):
            grouped = error_df.group_by(list(combo)).agg(
                pl.col("_error").mean().alias("metric_value"),
                pl.col("_error").median().alias("metric_median"),
                pl.col("_error").std().alias("metric_std"),
                pl.len().alias("row_count"),
            )

            # Filter by min sample size.
            grouped = grouped.filter(pl.col("row_count") >= min_sample_size)

            for row in grouped.to_dicts():
                diff = row["metric_value"] - overall_error
                results.append({
                    "dimensions": list(combo),
                    "values": {d: row[d] for d in combo},
                    "row_count": row["row_count"],
                    "metric_value": float(row["metric_value"]),
                    "metric_median": float(row["metric_median"]) if row["metric_median"] is not None else None,
                    "metric_std": float(row["metric_std"]) if row["metric_std"] is not None else None,
                    "overall_baseline": overall_error,
                    "difference": float(diff),
                    "abs_difference": abs(float(diff)),
                })

    # Sort by absolute difference, descending.
    results.sort(key=lambda x: x["abs_difference"], reverse=True)
    return results[:top_k]
