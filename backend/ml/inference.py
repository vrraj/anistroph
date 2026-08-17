"""Inference — generic prediction interface.

The external caller does NOT construct engineered features. For temporal
entity-based datasets, inference accepts (model_id, entity_id, timestamp)
and Anistroph retrieves the required historical observations, runs the same
Feature Engine, and predicts.

predict(model_id, entity_id=None, timestamp=None, records=None)

This allows future non-temporal datasets to provide records directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl
import joblib

from backend.datasets.config import DatasetConfig
from backend.datasets.registry import DatasetRegistry
from backend.features.engine import FeatureEngine
from backend.ml.base import Predictor
from backend.ml.registry import ModelRegistry
from backend.models.logistic import LogisticRegressionPredictor
from backend.models.xgboost import XGBoostPredictor
from backend.models.xgboost_regressor import XGBoostRegressorPredictor
from backend.models.linear_regression import LinearRegressionPredictor
from backend.targets.spec import TargetType


def _load_predictor(model_type: str, artifact_path: str) -> Predictor:
    if model_type == "logistic_regression":
        return LogisticRegressionPredictor.load(f"{artifact_path}/model.joblib")
    elif model_type == "xgboost":
        return XGBoostPredictor.load(f"{artifact_path}/model.joblib")
    elif model_type == "xgboost_regressor":
        return XGBoostRegressorPredictor.load(f"{artifact_path}/model.joblib")
    elif model_type == "linear_regression":
        return LinearRegressionPredictor.load(f"{artifact_path}/model.joblib")
    raise ValueError(f"unknown model type: {model_type!r}")


def predict(
    model_id: str,
    model_registry: ModelRegistry,
    dataset_registry: DatasetRegistry,
    config: DatasetConfig,
    entity_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    records: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Generic prediction.

    For temporal datasets: provide entity_id and timestamp. Anistroph loads
    the historical observations for that entity up to the timestamp, builds
    features using the same Feature Engine + persisted FeatureMetadata, and
    predicts.

    For non-temporal datasets: provide records directly.
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

    # Load the imputer persisted during training (handles NaNs from rolling windows).
    imputer_path = f"{meta.artifact_path}/imputer.joblib"
    imputer = None
    from pathlib import Path
    if Path(imputer_path).exists():
        imputer = joblib.load(imputer_path)

    is_regression = target_spec.type in (TargetType.REGRESSION,)

    # Determine if this is a temporal lookup (entity has multiple rows over time)
    # or a single-row entity lookup (e.g. wafer-level data with a timestamp
    # used only for chronological splitting).
    if spec.is_temporal() and entity_id is not None and timestamp is not None:
        # Temporal lookup: load history up to timestamp.
        df = pl.read_parquet(meta.parquet_path)
        ts_parsed = _parse_timestamp(timestamp)
        entity_df = df.filter(
            (pl.col(spec.entity_key) == entity_id)
            & (pl.col(spec.time_key) <= ts_parsed)
        ).sort(spec.time_key)

        if entity_df.height == 0:
            raise ValueError(
                f"no observations found for entity {entity_id!r} up to {timestamp}"
            )

        # Build features using the same engine + metadata (no refit).
        engine = FeatureEngine()
        feat_df, _ = engine.build_features(entity_df, spec, feature_spec, metadata=feature_metadata, fit=False)

        # Use the last row (the requested timestamp) for prediction.
        # If the exact timestamp isn't present, use the latest available.
        if ts_parsed in entity_df[spec.time_key].to_list():
            row = feat_df.filter(pl.col(spec.time_key) == ts_parsed)
        else:
            row = feat_df.tail(1)

        X = row.select(feature_cols).to_numpy()
        if imputer is not None:
            X = imputer.transform(X)

        if is_regression:
            pred_value = float(predictor.predict(X)[0])
            return {
                "model_id": model_id,
                "entity_id": entity_id,
                "timestamp": str(row[spec.time_key][0]),
                "predicted_yield": pred_value,
                "target_name": target_spec.name,
            }
        else:
            proba = predictor.predict_proba(X)[0, 1]
            pred_label = int(proba >= meta.decision_threshold)
            return {
                "model_id": model_id,
                "entity_id": entity_id,
                "timestamp": str(row[spec.time_key][0]),
                "prediction": pred_label,
                "probability": float(proba),
                "decision_threshold": meta.decision_threshold,
                "target_name": target_spec.name,
            }
    else:
        # Non-temporal: either entity_id lookup or records provided.
        if entity_id is not None:
            df = pl.read_parquet(meta.parquet_path)
            entity_df = df.filter(pl.col(spec.entity_key) == entity_id)
            if entity_df.height == 0:
                raise ValueError(f"no rows found for entity {entity_id!r}")
            engine = FeatureEngine()
            feat_df, _ = engine.build_features(entity_df, spec, feature_spec, metadata=feature_metadata, fit=False)
            X = feat_df.select(feature_cols).to_numpy()
            if imputer is not None:
                X = imputer.transform(X)

            if is_regression:
                pred_value = float(predictor.predict(X)[0])
                # Include actual yield if available.
                actual = None
                if target_spec.source_column in entity_df.columns:
                    actual = float(entity_df[target_spec.source_column][0])
                result = {
                    "model_id": model_id,
                    "entity_id": entity_id,
                    "predicted_yield": pred_value,
                    "target_name": target_spec.name,
                }
                if actual is not None:
                    result["actual_yield"] = actual
                return result
            else:
                proba = predictor.predict_proba(X)[0, 1]
                pred_label = int(proba >= meta.decision_threshold)
                return {
                    "model_id": model_id,
                    "entity_id": entity_id,
                    "prediction": pred_label,
                    "probability": float(proba),
                    "decision_threshold": meta.decision_threshold,
                    "target_name": target_spec.name,
                }
        elif records is not None:
            df = pl.DataFrame(records)
            # If the dataset has an entity_key / time_key (for chronological
            # splitting) but the caller's records don't include them, add
            # placeholder columns so build_features can sort without failing.
            # This is safe: only `current`/`categorical` transforms are used
            # with records-based prediction (rolling windows require the
            # entity_id+timestamp path above), so the placeholder values are
            # never read as features.
            placeholder_cols = {}
            if spec.entity_key and spec.entity_key not in df.columns:
                placeholder_cols[spec.entity_key] = ["__record__"] * df.height
            if spec.time_key and spec.time_key not in df.columns:
                placeholder_cols[spec.time_key] = [datetime(2000, 1, 1)] * df.height
            if placeholder_cols:
                df = df.with_columns([pl.Series(k, v) for k, v in placeholder_cols.items()])
            engine = FeatureEngine()
            feat_df, _ = engine.build_features(df, spec, feature_spec, metadata=feature_metadata, fit=False)
            X = feat_df.select(feature_cols).to_numpy()
            if imputer is not None:
                X = imputer.transform(X)

            if is_regression:
                preds = predictor.predict(X)
                return {
                    "model_id": model_id,
                    "predictions": preds.tolist(),
                    "target_name": target_spec.name,
                }
            else:
                probas = predictor.predict_proba(X)[:, 1]
                preds = (probas >= meta.decision_threshold).astype(int)
                return {
                    "model_id": model_id,
                    "predictions": preds.tolist(),
                    "probabilities": probas.tolist(),
                    "decision_threshold": meta.decision_threshold,
                    "target_name": target_spec.name,
                }
        else:
            raise ValueError("non-temporal dataset requires entity_id or records")


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO timestamp string."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
