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
from backend.ml.evaluation import best_threshold_by_f1, evaluate_binary, evaluate_regression
from backend.ml.registry import ModelRegistry
from backend.models.logistic import LogisticRegressionPredictor
from backend.models.xgboost import XGBoostPredictor
from backend.models.xgboost_regressor import XGBoostRegressorPredictor
from backend.models.linear_regression import LinearRegressionPredictor
from backend.targets.engine import TargetEngine
from backend.targets.spec import TargetSpec, TargetType


MODEL_FACTORIES: dict[str, type[Predictor]] = {
    "logistic_regression": LogisticRegressionPredictor,
    "xgboost": XGBoostPredictor,
    "xgboost_regressor": XGBoostRegressorPredictor,
    "linear_regression": LinearRegressionPredictor,
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
    is_train_partition: bool = False,
    validate_parquet_path: Optional[str] = None,
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
        is_train_partition: when True, parquet_path is a pre-partitioned
            train file. The function uses validate_parquet_path for validation
            (if provided) or carves a validation set from train. No test set
            is carved — the held-out evaluation partition is used separately
            via the evaluation endpoint.
        validate_parquet_path: path to a pre-partitioned validation file
            (used when is_train_partition is True).

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
    if is_train_partition and validate_parquet_path:
        # Pre-partitioned: use separate train and validate files.
        train_df = df
        val_df = target_engine.build_target(
            pl.read_parquet(validate_parquet_path), spec, ts,
        )
        # eval_df for training-time metrics = validation set.
        eval_df = val_df
    elif is_train_partition:
        # Pre-partitioned train file without a separate validate file:
        # carve a validation set from train (no test set).
        train_frac = spec.split.train
        val_frac = spec.split.validation
        total = train_frac + val_frac
        if total <= 0:
            total = 1.0
            train_frac = 1.0
            val_frac = 0.0
        train_only_split = SplitSpec(
            strategy=spec.split.strategy,
            train=train_frac / total,
            validation=val_frac / total,
            test=0.0,
        )
        if spec.is_temporal():
            train_df, val_df, _ = chronological_split(df, spec.time_key, train_only_split)
        else:
            train_df, val_df, _ = random_split(df, train_only_split)
        eval_df = val_df
    else:
        if spec.is_temporal():
            train_df, val_df, test_df = chronological_split(df, spec.time_key, spec.split)
        else:
            train_df, val_df, test_df = random_split(df, spec.split)
        eval_df = test_df

    # Fit feature metadata on training data only.
    train_feat, metadata = feature_engine.build_features(train_df, spec, fs, fit=True)
    # Transform val and eval using the same metadata (no refit).
    val_feat, _ = feature_engine.build_features(val_df, spec, fs, metadata=metadata, fit=False)
    eval_feat, _ = feature_engine.build_features(eval_df, spec, fs, metadata=metadata, fit=False)

    # Join features with target.
    target_col = ts.name
    keys = [spec.entity_key] + ([spec.time_key] if spec.time_key else [])
    train_full = train_feat.join(train_df.select(keys + [target_col]), on=keys, how="left")
    val_full = val_feat.join(val_df.select(keys + [target_col]), on=keys, how="left")
    eval_full = eval_feat.join(eval_df.select(keys + [target_col]), on=keys, how="left")

    feature_cols = metadata.feature_names
    X_train = train_full.select(feature_cols).to_numpy()
    y_train = train_full[target_col].to_numpy()
    X_val = val_full.select(feature_cols).to_numpy()
    y_val = val_full[target_col].to_numpy()
    X_eval = eval_full.select(feature_cols).to_numpy()
    y_eval = eval_full[target_col].to_numpy()

    # --- Impute NaN values (from rolling windows at series start) ---
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_eval = imputer.transform(X_eval)

    # --- Train ---
    params = model_parameters or {}
    predictor = MODEL_FACTORIES[model_type](**params)
    predictor._feature_names = feature_cols
    predictor.fit(X_train, y_train, X_val, y_val)

    # --- Evaluate (branch on target type) ---
    is_regression = ts.type in (TargetType.REGRESSION,)

    if is_regression:
        y_eval_pred = predictor.predict(X_eval)
        baseline_pred = float(np.mean(y_train))
        metrics = evaluate_regression(y_eval, y_eval_pred, baseline_pred=baseline_pred)
        threshold = 0.0  # not used for regression
    else:
        y_eval_proba = predictor.predict_proba(X_eval)[:, 1]
        threshold = best_threshold_by_f1(y_val, predictor.predict_proba(X_val)[:, 1])
        metrics = evaluate_binary(y_eval, y_eval_proba, threshold=threshold)

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
        test_period=_period(eval_df),
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
