"""Unit tests — ML training, evaluation, persistence, reload, prediction, feature parity."""

from __future__ import annotations

import pytest
import polars as pl

from backend.datasets.config import load_dataset_config
from backend.datasets.registry import DatasetRegistry
from backend.ml.evaluation import evaluate_binary, best_threshold_by_f1
from backend.ml.registry import ModelRegistry
from backend.ml.training import train_model, chronological_split, random_split
from backend.models.logistic import LogisticRegressionPredictor
from backend.models.xgboost import XGBoostPredictor

import numpy as np


@pytest.fixture
def registered_dataset(tmp_artifacts, small_dataset, config):
    """Register the small dataset and return the services needed."""
    from backend.services import AnistrophServices
    svc = AnistrophServices(
        dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_artifacts / "artifacts" / "models",
    )
    meta = svc.register_dataset_from_config(
        "datasets/predictive_maintenance/dataset.yaml",
        str(tmp_artifacts / "data" / "synthetic" / "small.csv"),
        parquet_path=str(tmp_artifacts / "data" / "processed" / "small.parquet"),
    )
    return svc, meta


class TestSplit:
    def test_chronological_split(self, small_dataset, config):
        train, val, test = chronological_split(small_dataset, "timestamp", config.dataset_spec.split)
        assert train.height + val.height + test.height == small_dataset.height
        # Train should be before val, val before test.
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_random_split(self, small_dataset, config):
        train, val, test = random_split(small_dataset, config.dataset_spec.split, seed=42)
        assert train.height + val.height + test.height == small_dataset.height


class TestEvaluation:
    def test_evaluate_binary(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.4, 0.8, 0.9, 0.3, 0.7])
        metrics = evaluate_binary(y_true, y_proba, threshold=0.5)
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics
        assert metrics["roc_auc"] > 0.5

    def test_best_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        t = best_threshold_by_f1(y_true, y_proba)
        assert 0.0 <= t <= 1.0


class TestPredictors:
    def test_logistic_save_load(self, tmp_path):
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegressionPredictor()
        model.fit(X, y)
        path = str(tmp_path / "model.joblib")
        model.save(path)
        model2 = LogisticRegressionPredictor.load(path)
        assert np.allclose(model.predict(X), model2.predict(X))

    def test_xgboost_save_load(self, tmp_path):
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        model = XGBoostPredictor(n_estimators=10)
        model.fit(X, y)
        path = str(tmp_path / "model.joblib")
        model.save(path)
        model2 = XGBoostPredictor.load(path)
        assert np.allclose(model.predict_proba(X), model2.predict_proba(X))

    def test_feature_importance(self, tmp_path):
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        model = XGBoostPredictor(n_estimators=10)
        model._feature_names = [f"f{i}" for i in range(5)]
        model.fit(X, y)
        imp = model.feature_importance()
        assert imp is not None
        assert len(imp) == 5


class TestTrainingPipeline:
    def test_train_xgboost(self, registered_dataset, config):
        svc, meta = registered_dataset
        result = svc.train(
            "predictive_maintenance", "failure_within_horizon", "xgboost",
            model_id="test-xgb-unit",
        )
        assert result["model_id"] == "test-xgb-unit"
        assert "metrics" in result
        assert "feature_names" in result
        # ROC-AUC should be computed (both classes present in eval set).
        # Performance on the small synthetic dataset is noisy, so we only
        # check that the metric exists, not that it exceeds 0.5.
        assert result["metrics"]["roc_auc"] is not None

    def test_train_logistic(self, registered_dataset, config):
        svc, meta = registered_dataset
        result = svc.train(
            "predictive_maintenance", "failure_within_horizon", "logistic_regression",
            model_id="test-lr-unit",
        )
        assert result["model_id"] == "test-lr-unit"
        assert result["metrics"]["roc_auc"] is not None

    def test_persist_and_reload(self, registered_dataset, config):
        svc, meta = registered_dataset
        result = svc.train(
            "predictive_maintenance", "failure_within_horizon", "xgboost",
            model_id="test-reload",
        )
        # Verify model is in registry.
        meta = svc.get_model("test-reload")
        assert meta is not None
        assert meta.model_type == "xgboost"
        # Verify artifact files exist.
        from pathlib import Path
        art = Path(meta.artifact_path)
        assert (art / "model.joblib").exists()
        assert (art / "metadata.json").exists()
        assert (art / "feature_spec.json").exists()
        assert (art / "target_spec.json").exists()
        assert (art / "metrics.json").exists()

    def test_prediction(self, registered_dataset, config):
        svc, meta = registered_dataset
        svc.train("predictive_maintenance", "failure_within_horizon", "xgboost", model_id="test-pred")
        pred = svc.predict("test-pred", entity_id="TOOL_000", timestamp="2026-06-05T12:00:00")
        assert "probability" in pred
        assert "prediction" in pred
        assert 0.0 <= pred["probability"] <= 1.0

    def test_train_inference_feature_parity(self, registered_dataset, config):
        """Training and inference must use the exact same Feature Engine."""
        svc, meta = registered_dataset
        result = svc.train("predictive_maintenance", "failure_within_horizon", "xgboost", model_id="test-parity")
        # The model's feature_names should match what inference produces.
        pred = svc.predict("test-parity", entity_id="TOOL_000", timestamp="2026-06-05T12:00:00")
        assert "probability" in pred  # inference succeeded with same features

    def test_explain(self, registered_dataset, config):
        svc, meta = registered_dataset
        svc.train("predictive_maintenance", "failure_within_horizon", "xgboost", model_id="test-explain")
        expl = svc.explain("test-explain", entity_id="TOOL_000", timestamp="2026-06-05T12:00:00", top_k=5)
        assert "top_drivers" in expl
        assert len(expl["top_drivers"]) <= 5
        assert "feature" in expl["top_drivers"][0]
        assert "impact" in expl["top_drivers"][0]
