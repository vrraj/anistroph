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


def evaluate_on_eval_set(
    model_id: str,
    model_registry: ModelRegistry,
    config: DatasetConfig,
    eval_parquet_path: str,
    sample_size: int = 50,
) -> dict[str, Any]:
    """Evaluate a model against the held-out evaluation partition.

    Args:
        model_id: registered model to evaluate.
        model_registry: model registry (for loading model artifacts).
        config: DatasetConfig for the model's dataset.
        eval_parquet_path: path to the evaluation Parquet file.
        sample_size: number of prediction-vs-actual rows to include in the
            response (capped at 1000). Aggregate metrics are over the full set.

    Returns a dict with:
        - model_id, dataset_id, target_name, target_type
        - eval_row_count: number of rows in the evaluation set
        - metrics: aggregate evaluation metrics
        - predictions_sample: list of {entity_id, actual, predicted, ...} rows
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

    # --- Run inference ---
    if is_regression:
        y_pred = predictor.predict(X)
        baseline = float(np.mean(y_actual))
        metrics = evaluate_regression(y_actual, y_pred, baseline_pred=baseline)
        pred_values = y_pred.tolist()
        actual_values = y_actual.tolist()
    else:
        y_proba = predictor.predict_proba(X)[:, 1]
        threshold = meta.decision_threshold
        metrics = evaluate_binary(y_actual, y_proba, threshold=threshold)
        pred_values = y_proba.tolist()
        actual_values = y_actual.tolist()

    # --- Build prediction-vs-actual sample ---
    sample_size = max(1, min(int(sample_size), 1000))
    entity_col = spec.entity_key
    time_col = spec.time_key

    # Build a sample of rows with entity, (timestamp), actual, predicted.
    sample_rows: list[dict[str, Any]] = []
    n = len(pred_values)
    for i in range(min(sample_size, n)):
        row: dict[str, Any] = {
            "entity_id": str(full[entity_col][i]) if entity_col in full.columns else None,
            "actual": float(actual_values[i]),
            "predicted": float(pred_values[i]),
        }
        if is_regression:
            row["error"] = float(pred_values[i] - actual_values[i])
            row["abs_error"] = abs(row["error"])
        else:
            row["predicted_label"] = int(pred_values[i] >= threshold)
        if time_col and time_col in full.columns:
            row["timestamp"] = str(full[time_col][i])
        sample_rows.append(row)

    return {
        "model_id": model_id,
        "dataset_id": meta.dataset_id,
        "target_name": target_spec.name,
        "target_type": target_spec.type.value,
        "eval_row_count": n,
        "metrics": metrics,
        "predictions_sample": sample_rows,
    }
