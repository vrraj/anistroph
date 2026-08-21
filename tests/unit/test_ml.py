"""Unit tests — ML training, evaluation, persistence, prediction, feature parity."""

from __future__ import annotations

import pytest
import polars as pl
import numpy as np

from backend.datasets.config import load_dataset_config
from backend.ml.evaluation import evaluate_binary, best_threshold_by_f1
from backend.ml.training import train_model, chronological_split, random_split, resolve_model_type
from backend.models.xgboost import XGBoostPredictor
from backend.targets.spec import TargetSpec, TargetType


@pytest.fixture
def registered_dataset(tmp_artifacts, small_dataset, config):
    from backend.services import AnistrophServices
    svc = AnistrophServices(
        dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_artifacts / "artifacts" / "models",
    )
    svc.register_dataset_from_config(
        "datasets/predictive_maintenance/dataset.yaml",
        str(tmp_artifacts / "data" / "synthetic" / "small.csv"),
        parquet_path=str(tmp_artifacts / "data" / "processed" / "small.parquet"),
    )
    return svc, None


class TestSplit:
    def test_chronological_split(self, small_dataset, config):
        train, val, test = chronological_split(small_dataset, "timestamp", config.dataset_spec.split)
        assert train.height + val.height + test.height == small_dataset.height
        assert train["timestamp"].max() <= val["timestamp"].min()


class TestEvaluation:
    def test_evaluate_binary(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.4, 0.8, 0.9, 0.3, 0.7])
        metrics = evaluate_binary(y_true, y_proba, threshold=0.5)
        assert metrics["roc_auc"] > 0.5
        assert "f1" in metrics


class TestTrainingPipeline:
    def test_train_and_predict(self, registered_dataset):
        svc, _ = registered_dataset
        result = svc.train("predictive_maintenance", "failure_within_horizon", "xgboost", model_id="test-xgb")
        assert result["model_id"] == "test-xgb"
        assert result["metrics"]["roc_auc"] is not None
        pred = svc.predict("test-xgb", entity_id="TOOL_000", timestamp="2026-06-05T12:00:00")
        assert 0.0 <= pred["probability"] <= 1.0

    def test_persist_and_reload(self, registered_dataset):
        svc, _ = registered_dataset
        svc.train("predictive_maintenance", "failure_within_horizon", "xgboost", model_id="test-reload")
        meta = svc.get_model("test-reload")
        from pathlib import Path
        art = Path(meta.artifact_path)
        assert (art / "model.joblib").exists()
        assert (art / "metadata.json").exists()

    def test_explain(self, registered_dataset):
        svc, _ = registered_dataset
        svc.train("predictive_maintenance", "failure_within_horizon", "xgboost", model_id="test-explain")
        expl = svc.explain("test-explain", entity_id="TOOL_000", timestamp="2026-06-05T12:00:00", top_k=5)
        assert "top_drivers" in expl
        assert len(expl["top_drivers"]) <= 5


class TestTaskTypeAutoSelection:
    def test_resolve_model_type(self):
        assert resolve_model_type(None, TargetSpec(name="y", type=TargetType.REGRESSION, source_column="y")) == "xgboost_regressor"
        assert resolve_model_type(None, TargetSpec(name="y", type=TargetType.CLASSIFICATION, source_column="y")) == "xgboost"
        assert resolve_model_type("linear_regression", TargetSpec(name="y", type=TargetType.REGRESSION, source_column="y")) == "linear_regression"
        with pytest.raises(ValueError, match="unknown model type"):
            resolve_model_type("nonexistent", TargetSpec(name="y", type=TargetType.REGRESSION, source_column="y"))

    def test_train_auto_selects_for_classification(self, registered_dataset):
        svc, _ = registered_dataset
        result = svc.train("predictive_maintenance", "failure_within_horizon", model_id="test-auto")
        assert result["model_type"] == "xgboost"
