"""Training pipeline — generic, explicit training.

Pipeline:
    Dataset Registry → DatasetSpec → load Parquet → Feature Engine →
    Target Engine → training matrix X/y → chronological split →
    preprocessing → model.fit() → evaluation → persist model → register model

Training is explicit. Inference never retrains automatically.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer

from backend.datasets.config import DatasetConfig
from backend.datasets.registry import DatasetRegistry
from backend.datasets.spec import DatasetSpec, SplitSpec
from backend.features.engine import FeatureEngine, FeatureMetadata
from backend.features.spec import FeatureSpec
from backend.ml.base import Predictor
from backend.ml.evaluation import best_threshold_by_f1, evaluate_binary
from backend.ml.registry import ModelRegistry
from backend.models.logistic import LogisticRegressionPredictor
from backend.models.xgboost import XGBoostPredictor
from backend.targets.engine import TargetEngine
from backend.targets.spec import TargetSpec, TargetType


MODEL_FACTORIES: dict[str, type[Predictor]] = {
    "logistic_regression": LogisticRegressionPredictor,
    "xgboost": XGBoostPredictor,
}


def available_model_types() -> list[str]:
    return list(MODEL_FACTORIES.keys())


def chronological_split(
    df: pl.DataFrame,
    time_key: str,
    split: SplitSpec,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split a temporal dataset chronologically (no random shuffling)."""
    df = df.sort(time_key)
    n = df.height
    train_end = int(n * split.train)
    val_end = int(n * (split.train + split.validation))
    train = df[:train_end]
    val = df[train_end:val_end]
    test = df[val_end:]
    return train, val, test


def random_split(
    df: pl.DataFrame,
    split: SplitSpec,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Random split for non-temporal datasets."""
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n = df.height
    train_end = int(n * split.train)
    val_end = int(n * (split.train + split.validation))
    return df[:train_end], df[train_end:val_end], df[val_end:]


def train_model(
    dataset_id: str,
    target_name: str,
    model_type: str,
    dataset_registry: DatasetRegistry,
    model_registry: ModelRegistry,
    config: DatasetConfig,
    feature_spec: Optional[FeatureSpec] = None,
    model_parameters: Optional[dict[str, Any]] = None,
    model_id: Optional[str] = None,
    parquet_path: Optional[str] = None,
) -> dict[str, Any]:
    """Train a model and persist it.

    Args:
        dataset_id: registered dataset to train on.
        target_name: name of the target (must match TargetSpec.name).
        model_type: one of available_model_types().
        dataset_registry: dataset registry.
        model_registry: model registry.
        config: DatasetConfig with specs.
        feature_spec: override feature spec (defaults to config's).
        model_parameters: hyperparameters for the model.
        model_id: explicit model ID (auto-generated if None).
        parquet_path: override parquet path (defaults to registry's).

    Returns a dict with model_id, metrics, and metadata.
    """
    # --- Resolve dataset ---
    dmeta = dataset_registry.get(dataset_id)
    if dmeta is None:
        raise ValueError(f"dataset {dataset_id!r} is not registered")
    spec = config.dataset_spec
    fs = feature_spec or config.feature_spec
    ts = config.target_spec
    if ts is None:
        raise ValueError("no TargetSpec in config")
    if ts.name != target_name:
        raise ValueError(f"target name mismatch: {target_name!r} != {ts.name!r}")
    pq = parquet_path or dmeta.parquet_path

    if model_type not in MODEL_FACTORIES:
        raise ValueError(f"unknown model type {model_type!r}; available: {available_model_types()}")

    # --- Load data ---
    df = pl.read_parquet(pq)

    # --- Build target ---
    target_engine = TargetEngine()
    df = target_engine.build_target(df, spec, ts)

    # --- Build features (fit on full data; categorical fit on train only to avoid leakage) ---
    feature_engine = FeatureEngine()

    # Split first, then fit features on train only (categorical categories).
    if spec.is_temporal():
        train_df, val_df, test_df = chronological_split(df, spec.time_key, spec.split)
    else:
        train_df, val_df, test_df = random_split(df, spec.split)

    # Fit feature metadata on training data only.
    train_feat, metadata = feature_engine.build_features(train_df, spec, fs, fit=True)
    # Transform val and test using the same metadata (no refit).
    val_feat, _ = feature_engine.build_features(val_df, spec, fs, metadata=metadata, fit=False)
    test_feat, _ = feature_engine.build_features(test_df, spec, fs, metadata=metadata, fit=False)

    # Join features with target.
    target_col = ts.name
    keys = [spec.entity_key] + ([spec.time_key] if spec.time_key else [])
    train_full = train_feat.join(train_df.select(keys + [target_col]), on=keys, how="left")
    val_full = val_feat.join(val_df.select(keys + [target_col]), on=keys, how="left")
    test_full = test_feat.join(test_df.select(keys + [target_col]), on=keys, how="left")

    feature_cols = metadata.feature_names
    X_train = train_full.select(feature_cols).to_numpy()
    y_train = train_full[target_col].to_numpy()
    X_val = val_full.select(feature_cols).to_numpy()
    y_val = val_full[target_col].to_numpy()
    X_test = test_full.select(feature_cols).to_numpy()
    y_test = test_full[target_col].to_numpy()

    # --- Impute NaN values (from rolling windows at series start) ---
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # --- Train ---
    params = model_parameters or {}
    predictor = MODEL_FACTORIES[model_type](**params)
    predictor._feature_names = feature_cols
    predictor.fit(X_train, y_train, X_val, y_val)

    # --- Evaluate ---
    y_test_proba = predictor.predict_proba(X_test)[:, 1]
    threshold = best_threshold_by_f1(y_val, predictor.predict_proba(X_val)[:, 1])
    metrics = evaluate_binary(y_test, y_test_proba, threshold=threshold)

    # --- Persist model ---
    if model_id is None:
        model_id = f"{dataset_id}-{model_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Use the model registry's artifact directory (absolute path).
    artifact_dir = model_registry.artifacts_dir / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(artifact_dir / "model.joblib")
    predictor.save(model_path)

    # Persist the imputer alongside the model.
    import joblib
    joblib.dump(imputer, str(artifact_dir / "imputer.joblib"))

    # Training/validation/test periods.
    def _period(d: pl.DataFrame) -> dict[str, str] | None:
        if spec.time_key and spec.time_key in d.columns:
            return {"start": str(d[spec.time_key].min()), "end": str(d[spec.time_key].max())}
        return None

    meta = model_registry.register(
        model_id=model_id,
        model_type=model_type,
        dataset_id=dataset_id,
        target_spec=ts,
        feature_spec=fs,
        feature_metadata=metadata,
        metrics=metrics,
        hyperparameters=params,
        decision_threshold=threshold,
        training_period=_period(train_df),
        validation_period=_period(val_df),
        test_period=_period(test_df),
        parquet_path=pq,
    )

    return {
        "model_id": model_id,
        "model_type": model_type,
        "dataset_id": dataset_id,
        "metrics": metrics,
        "decision_threshold": threshold,
        "feature_names": feature_cols,
        "metadata": meta.model_dump(),
    }
