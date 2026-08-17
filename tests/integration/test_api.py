"""Integration tests — REST API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.services import AnistrophServices, _services
import backend.services as svc_mod


@pytest.fixture
def client(tmp_artifacts, small_dataset):
    """Create a test client with isolated services."""
    services = AnistrophServices(
        dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_artifacts / "artifacts" / "models",
    )
    # Register dataset.
    services.register_dataset_from_config(
        "datasets/predictive_maintenance/dataset.yaml",
        str(tmp_artifacts / "data" / "synthetic" / "small.csv"),
        parquet_path=str(tmp_artifacts / "data" / "processed" / "small.parquet"),
    )
    # Override the module-level singleton.
    svc_mod._services = services
    app = create_app()
    yield TestClient(app)
    svc_mod._services = None


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestDatasetAPI:
    def test_list_datasets(self, client):
        r = client.get("/datasets")
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 1
        assert data[0]["dataset_id"] == "predictive_maintenance"

    def test_get_dataset(self, client):
        r = client.get("/datasets/predictive_maintenance")
        assert r.status_code == 200
        assert r.json()["dataset_id"] == "predictive_maintenance"

    def test_get_dataset_not_found(self, client):
        r = client.get("/datasets/nonexistent")
        assert r.status_code == 404

    def test_profile(self, client):
        r = client.get("/datasets/predictive_maintenance/profile")
        assert r.status_code == 200
        prof = r.json()
        assert "row_count" in prof
        assert "entity_count" in prof


class TestAnalysisAPI:
    def test_slice(self, client):
        r = client.post("/analysis/slice", json={
            "dataset_id": "predictive_maintenance",
            "dimensions": ["machine_type"],
            "metric": "failure",
            "aggregation": "mean",
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 0
        assert "machine_type" in data[0]

    def test_compare(self, client):
        r = client.post("/analysis/compare", json={
            "dataset_id": "predictive_maintenance",
            "dimension": "machine_type",
            "metric": "failure",
            "aggregation": "mean",
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 0


class TestModelAPI:
    def test_list_model_types(self, client):
        r = client.get("/models/types")
        assert r.status_code == 200
        assert "xgboost" in r.json()["model_types"]

    def test_train_and_get(self, client):
        r = client.post("/models/train", json={
            "dataset_id": "predictive_maintenance",
            "target_name": "failure_within_horizon",
            "model_type": "xgboost",
            "model_id": "api-test-xgb",
        })
        assert r.status_code == 200
        result = r.json()
        assert result["model_id"] == "api-test-xgb"

        r2 = client.get("/models/api-test-xgb")
        assert r2.status_code == 200

        r3 = client.get("/models/api-test-xgb/metrics")
        assert r3.status_code == 200
        assert "roc_auc" in r3.json()


class TestPredictionAPI:
    def test_predict(self, client):
        # Train first.
        client.post("/models/train", json={
            "dataset_id": "predictive_maintenance",
            "target_name": "failure_within_horizon",
            "model_type": "xgboost",
            "model_id": "api-pred-test",
        })
        r = client.post("/predictions", json={
            "model_id": "api-pred-test",
            "entity_id": "TOOL_000",
            "timestamp": "2026-06-05T12:00:00",
        })
        assert r.status_code == 200
        assert "probability" in r.json()

    def test_explain(self, client):
        client.post("/models/train", json={
            "dataset_id": "predictive_maintenance",
            "target_name": "failure_within_horizon",
            "model_type": "xgboost",
            "model_id": "api-explain-test",
        })
        r = client.post("/predictions/explain", json={
            "model_id": "api-explain-test",
            "entity_id": "TOOL_000",
            "timestamp": "2026-06-05T12:00:00",
            "top_k": 5,
        })
        assert r.status_code == 200
        assert "top_drivers" in r.json()


class TestEvaluationAPI:
    def test_evaluate_model(self, client):
        # Train a model first.
        client.post("/models/train", json={
            "dataset_id": "predictive_maintenance",
            "target_name": "failure_within_horizon",
            "model_type": "xgboost",
            "model_id": "api-eval-test",
        })
        r = client.post("/evaluations/api-eval-test", json={"sample_size": 10})
        assert r.status_code == 200
        data = r.json()
        assert data["model_id"] == "api-eval-test"
        assert data["dataset_id"] == "predictive_maintenance"
        assert data["eval_row_count"] > 0
        assert "metrics" in data
        assert "roc_auc" in data["metrics"]
        assert isinstance(data["predictions_sample"], list)
        assert len(data["predictions_sample"]) <= 10
        # Each sample row has actual + predicted.
        row = data["predictions_sample"][0]
        assert "entity_id" in row
        assert "actual" in row
        assert "predicted" in row

    def test_evaluate_model_not_found(self, client):
        r = client.post("/evaluations/nonexistent", json={"sample_size": 10})
        assert r.status_code == 400

    def test_evaluate_regression_model(self, client):
        # Train a regression model on the semiconductor dataset.
        client.post("/datasets", json={
            "config_path": "datasets/semiconductor_yield/dataset.yaml",
            "source_path": "data/semiconductor_yield/data.parquet",
        })
        client.post("/models/train", json={
            "dataset_id": "semiconductor_yield",
            "target_name": "wafer_yield",
            "model_type": "xgboost_regressor",
            "model_id": "api-eval-reg",
        })
        r = client.post("/evaluations/api-eval-reg", json={"sample_size": 5})
        assert r.status_code == 200
        data = r.json()
        assert data["target_type"] == "regression"
        assert "mae" in data["metrics"]
        assert "rmse" in data["metrics"]
        assert "r2" in data["metrics"]
        row = data["predictions_sample"][0]
        assert "error" in row
        assert "abs_error" in row
