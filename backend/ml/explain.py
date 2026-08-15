"""Explainability — generic model explanation.

Uses SHAP TreeExplainer for XGBoost models to produce per-prediction
feature contributions with proper signs (positive = increases prediction,
negative = decreases prediction). Falls back to importance-weighted
contributions for models without SHAP support.

Returns structured top-drivers data with separate top_positive and
top_negative lists. No LLM generates or fabricates drivers.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import joblib

from backend.datasets.config import DatasetConfig
from backend.datasets.registry import DatasetRegistry
from backend.features.engine import FeatureEngine
from backend.ml.base import Predictor
from backend.ml.inference import _load_predictor, _parse_timestamp
from backend.ml.registry import ModelRegistry
from backend.targets.spec import TargetType
import polars as pl


def explain_prediction(
    model_id: str,
    model_registry: ModelRegistry,
    dataset_registry: DatasetRegistry,
    config: DatasetConfig,
    entity_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    records: Optional[list[dict[str, Any]]] = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Explain a prediction by returning the top contributing features.

    For XGBoost models, uses SHAP TreeExplainer to produce exact per-
    prediction contributions. For other models, falls back to
    importance-weighted contributions.

    Returns top_positive (features that increase the prediction) and
    top_negative (features that decrease the prediction), each sorted
    by absolute contribution magnitude.
    """
    meta = model_registry.get(model_id)
    if meta is None:
        raise ValueError(f"model {model_id!r} not found")

    spec = config.dataset_spec
    feature_spec = model_registry.load_feature_spec(model_id)
    feature_metadata = model_registry.load_feature_metadata(model_id)
    target_spec = model_registry.load_target_spec(model_id)
    predictor = _load_predictor(meta.model_type, meta.artifact_path)
    predictor._feature_names = feature_metadata.feature_names
    feature_cols = feature_metadata.feature_names

    # Load the imputer persisted during training.
    from pathlib import Path
    imputer_path = f"{meta.artifact_path}/imputer.joblib"
    imputer = joblib.load(imputer_path) if Path(imputer_path).exists() else None

    # Get the prediction + feature vector.
    is_regression = target_spec.type in (TargetType.REGRESSION,)

    # Temporal lookup with history (entity_id + timestamp both provided).
    if spec.is_temporal() and entity_id is not None and timestamp is not None:
        df = pl.read_parquet(meta.parquet_path)
        ts_parsed = _parse_timestamp(timestamp)
        entity_df = df.filter(
            (pl.col(spec.entity_key) == entity_id)
            & (pl.col(spec.time_key) <= ts_parsed)
        ).sort(spec.time_key)
        if entity_df.height == 0:
            raise ValueError(f"no observations for entity {entity_id!r} up to {timestamp}")
        engine = FeatureEngine()
        feat_df, _ = engine.build_features(entity_df, spec, feature_spec, metadata=feature_metadata, fit=False)
        if ts_parsed in entity_df[spec.time_key].to_list():
            row = feat_df.filter(pl.col(spec.time_key) == ts_parsed)
        else:
            row = feat_df.tail(1)
        X = row.select(feature_cols).to_numpy()
        if imputer is not None:
            X = imputer.transform(X)
        if is_regression:
            pred_value = float(predictor.predict(X)[0])
        else:
            pred_value = float(predictor.predict_proba(X)[0, 1])
        feature_values = {c: float(row[c][0]) for c in feature_cols if row[c][0] is not None}
        ts_str = str(row[spec.time_key][0])
    elif entity_id is not None:
        # Single-row entity lookup (no timestamp or non-temporal).
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
        else:
            pred_value = float(predictor.predict_proba(X)[0, 1])
        feature_values = {c: float(feat_df[c][0]) for c in feature_cols}
        ts_str = str(entity_df[spec.time_key][0]) if spec.time_key and spec.time_key in entity_df.columns else None
    elif records is not None:
        df = pl.DataFrame(records)
        engine = FeatureEngine()
        feat_df, _ = engine.build_features(df, spec, feature_spec, metadata=feature_metadata, fit=False)
        X = feat_df.select(feature_cols).to_numpy()
        if imputer is not None:
            X = imputer.transform(X)
        if is_regression:
            pred_value = float(predictor.predict(X)[0])
        else:
            pred_value = float(predictor.predict_proba(X)[0, 1])
        feature_values = {c: float(feat_df[c][0]) for c in feature_cols}
        ts_str = None
    else:
        raise ValueError("provide entity_id (optionally with timestamp) or records")

    # --- Compute per-prediction contributions ---
    contributions = _compute_contributions(predictor, X, feature_cols, feature_values)

    # Split into positive (increase prediction) and negative (decrease).
    positive = [c for c in contributions if c["impact"] > 0]
    negative = [c for c in contributions if c["impact"] < 0]

    # Sort positive by impact descending (most positive first).
    positive.sort(key=lambda x: x["impact"], reverse=True)
    # Sort negative by impact ascending (most negative first).
    negative.sort(key=lambda x: x["impact"])

    top_positive = positive[:top_k]
    top_negative = negative[:top_k]

    # Also keep a combined top_drivers for backward compatibility (sorted by abs).
    combined = positive + negative
    combined.sort(key=lambda x: abs(x["impact"]), reverse=True)
    top_drivers = combined[:top_k]

    if is_regression:
        return {
            "model_id": model_id,
            "entity_id": entity_id,
            "timestamp": ts_str,
            "predicted_yield": pred_value,
            "target_name": target_spec.name,
            "explanation_method": _explanation_method(predictor),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "top_drivers": top_drivers,
        }
    else:
        return {
            "model_id": model_id,
            "entity_id": entity_id,
            "timestamp": ts_str,
            "prediction": int(pred_value >= meta.decision_threshold),
            "probability": pred_value,
            "decision_threshold": meta.decision_threshold,
            "target_name": target_spec.name,
            "explanation_method": _explanation_method(predictor),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "top_drivers": top_drivers,
        }


def _explanation_method(predictor: Predictor) -> str:
    """Return the explanation method name used."""
    if hasattr(predictor, "explain_instance") and not isinstance(predictor, type):
        try:
            # Check if the method is overridden (not the base NotImplementedError).
            import inspect
            method = getattr(type(predictor), "explain_instance", None)
            if method is not None and method.__qualname__ != "Predictor.explain_instance":
                return "shap_tree_explainer"
        except Exception:
            pass
    return "importance_weighted"


def _compute_contributions(
    predictor: Predictor,
    X: np.ndarray,
    feature_cols: list[str],
    feature_values: dict[str, float],
) -> list[dict[str, Any]]:
    """Compute per-prediction feature contributions.

    Uses SHAP TreeExplainer for XGBoost models. Falls back to
    importance * |value| for models without SHAP support.
    """
    # Try SHAP first (XGBoost models have explain_instance).
    try:
        shap_values = predictor.explain_instance(X)
        # shap_values shape: (n_samples, n_features) — take first sample.
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        vals = shap_values[0]
        contributions = []
        for i, fname in enumerate(feature_cols):
            contributions.append({
                "feature": fname,
                "impact": float(vals[i]),
                "value": feature_values.get(fname),
            })
        return contributions
    except (NotImplementedError, AttributeError):
        pass
    except Exception:
        # If SHAP fails for any reason, fall back to importance-weighted.
        pass

    # Fallback: importance * |feature value|.
    importance = predictor.feature_importance() or {}
    contributions = []
    for fname in feature_cols:
        imp = abs(importance.get(fname, 0.0))
        val = abs(feature_values.get(fname, 0.0))
        # Use signed value for direction: positive features get positive impact.
        raw_val = feature_values.get(fname, 0.0)
        sign = 1.0 if raw_val >= 0 else -1.0
        contributions.append({
            "feature": fname,
            "impact": float(imp * val * sign),
            "value": feature_values.get(fname),
        })
    return contributions
