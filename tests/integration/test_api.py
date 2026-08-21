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

    def test_train_auto_selects_model_type(self, client):
        """Omitting model_type auto-selects from the dataset's task type."""
        r = client.post("/models/train", json={
            "dataset_id": "predictive_maintenance",
            "target_name": "failure_within_horizon",
            "model_id": "api-auto-classification",
        })
        assert r.status_code == 200
        result = r.json()
        # predictive_maintenance is a classification dataset → xgboost.
        assert result["model_type"] == "xgboost"
        assert result["metrics"]["roc_auc"] is not None

    def test_delete_model(self, client):
        """Delete a model via REST and verify it's gone."""
        # Train a model to delete.
        client.post("/models/train", json={
            "dataset_id": "predictive_maintenance",
            "target_name": "failure_within_horizon",
            "model_id": "api-delete-test",
        })
        # Verify it exists.
        r = client.get("/models/api-delete-test")
        assert r.status_code == 200
        # Delete it.
        r = client.delete("/models/api-delete-test")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        # Verify it's gone.
        r = client.get("/models/api-delete-test")
        assert r.status_code == 404

    def test_delete_model_not_found(self, client):
        """Deleting a nonexistent model returns 404."""
        r = client.delete("/models/nonexistent-model")
        assert r.status_code == 404


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
        assert "mse" in data["metrics"]
        assert "rmse" in data["metrics"]
        assert "r2" in data["metrics"]
        assert "mape" in data["metrics"]
        assert "max_error" in data["metrics"]
        row = data["predictions_sample"][0]
        assert "error" in row
        assert "abs_error" in row

    def test_evaluate_regression_with_filters(self, client):
        """Filtered evaluation returns both overall and filtered metrics."""
        # Reuse the model from the previous test.
        client.post("/datasets", json={
            "config_path": "datasets/semiconductor_yield/dataset.yaml",
            "source_path": "data/semiconductor_yield/data.parquet",
        })
        client.post("/models/train", json={
            "dataset_id": "semiconductor_yield",
            "target_name": "wafer_yield",
            "model_type": "xgboost_regressor",
            "model_id": "api-eval-filtered",
        })
        r = client.post("/evaluations/api-eval-filtered", json={
            "sample_size": 5,
            "filters": {"etch_tool": "ETCH_01"},
        })
        assert r.status_code == 200
        data = r.json()
        # Overall metrics present
        assert "mae" in data["metrics"]
        assert "r2" in data["metrics"]
        # Filtered metrics present
        assert "filtered_metrics" in data
        assert "filtered_row_count" in data
        assert "filters" in data
        assert data["filters"] == {"etch_tool": "ETCH_01"}
        assert data["filtered_row_count"] > 0
        assert data["filtered_row_count"] < data["eval_row_count"]
        fm = data["filtered_metrics"]
        assert "mae" in fm
        assert "r2" in fm
        assert "mape" in fm

    def test_evaluate_no_filters_has_no_filtered_metrics(self, client):
        """Without filters, filtered_metrics should be absent."""
        client.post("/datasets", json={
            "config_path": "datasets/semiconductor_yield/dataset.yaml",
            "source_path": "data/semiconductor_yield/data.parquet",
        })
        client.post("/models/train", json={
            "dataset_id": "semiconductor_yield",
            "target_name": "wafer_yield",
            "model_type": "xgboost_regressor",
            "model_id": "api-eval-nofilter",
        })
        r = client.post("/evaluations/api-eval-nofilter", json={"sample_size": 5})
        assert r.status_code == 200
        data = r.json()
        assert "filtered_metrics" not in data
        assert "filtered_row_count" not in data

    def test_evaluate_unknown_filter_column(self, client):
        """Unknown filter column returns 400."""
        client.post("/datasets", json={
            "config_path": "datasets/semiconductor_yield/dataset.yaml",
            "source_path": "data/semiconductor_yield/data.parquet",
        })
        client.post("/models/train", json={
            "dataset_id": "semiconductor_yield",
            "target_name": "wafer_yield",
            "model_type": "xgboost_regressor",
            "model_id": "api-eval-badfilter",
        })
        r = client.post("/evaluations/api-eval-badfilter", json={
            "sample_size": 5,
            "filters": {"nonexistent_column": "foo"},
        })
        assert r.status_code == 400

    def test_find_evaluation_slices(self, client):
        """find_evaluation_slices returns ranked error slices."""
        client.post("/datasets", json={
            "config_path": "datasets/semiconductor_yield/dataset.yaml",
            "source_path": "data/semiconductor_yield/data.parquet",
        })
        client.post("/models/train", json={
            "dataset_id": "semiconductor_yield",
            "target_name": "wafer_yield",
            "model_type": "xgboost_regressor",
            "model_id": "api-eval-slices",
        })
        r = client.post("/evaluations/api-eval-slices/slices", json={
            "metric": "abs_error",
            "min_sample_size": 50,
            "top_k": 10,
        })
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Each slice should have required fields.
        s = data[0]
        assert "dimensions" in s
        assert "values" in s
        assert "row_count" in s
        assert "metric_value" in s
        assert "overall_baseline" in s
        assert "difference" in s
        assert "abs_difference" in s
        # Sorted by abs_difference descending.
        assert data[0]["abs_difference"] >= data[-1]["abs_difference"]
        # Row counts respect min_sample_size.
        for s in data:
            assert s["row_count"] >= 50

    def test_find_evaluation_slices_pct_error(self, client):
        """find_evaluation_slices supports pct_error metric."""
        client.post("/datasets", json={
            "config_path": "datasets/semiconductor_yield/dataset.yaml",
            "source_path": "data/semiconductor_yield/data.parquet",
        })
        client.post("/models/train", json={
            "dataset_id": "semiconductor_yield",
            "target_name": "wafer_yield",
            "model_type": "xgboost_regressor",
            "model_id": "api-eval-pct",
        })
        r = client.post("/evaluations/api-eval-pct/slices", json={
            "metric": "pct_error",
            "min_sample_size": 50,
            "top_k": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 0
        # pct_error values should be percentages (small for yield 0-1).
        assert all(s["metric_value"] >= 0 for s in data)


class TestSearchAPI:
    """Tests for the parametric search REST endpoints."""

    @pytest.fixture
    def mem_client(self, tmp_artifacts):
        """Client with semiconductor_memory dataset registered."""
        services = AnistrophServices(
            dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
            model_registry_dir=tmp_artifacts / "artifacts" / "models",
        )
        services.register_dataset_from_config(
            "datasets/semiconductor_memory/dataset.yaml",
            "data/semiconductor_memory/data.csv",
            parquet_path=str(tmp_artifacts / "data" / "processed" / "semiconductor_memory.parquet"),
        )
        svc_mod._services = services
        app = create_app()
        yield TestClient(app)
        svc_mod._services = None

    def test_get_search_contract(self, mem_client):
        r = mem_client.get("/datasets/semiconductor_memory/search-contract")
        assert r.status_code == 200
        contract = r.json()
        assert contract["dataset_id"] == "semiconductor_memory"
        assert "eq" in contract["supported_operators"]
        assert "semantic" in contract["supported_operators"]
        assert len(contract["searchable_fields"]) > 0
        assert len(contract["semantic_filters"]) >= 2  # operating_temperature, industrial_temperature

    def test_search_contract_no_config_returns_404(self, client):
        """predictive_maintenance has no search config → 404."""
        r = client.get("/datasets/predictive_maintenance/search-contract")
        assert r.status_code == 404

    def test_search_acceptance_query_1(self, mem_client):
        """DDR5_COMPONENT + x8 + >=6400 MT/s."""
        r = mem_client.post("/datasets/semiconductor_memory/search", json={
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
                {"field": "bus_width_bits", "op": "eq", "value": 8},
                {"field": "data_rate_mt_s", "op": "gte", "value": 6400},
            ],
            "limit": 10,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] > 0
        assert data["returned"] <= 10
        for row in data["rows"]:
            assert row["product_family"] == "DDR5_COMPONENT"
            assert row["bus_width_bits"] == 8
            assert row["data_rate_mt_s"] >= 6400

    def test_search_acceptance_query_2_semantic_temp(self, mem_client):
        """supports 55C (semantic operating_temperature filter)."""
        r = mem_client.post("/datasets/semiconductor_memory/search", json={
            "filters": [
                {"field": "operating_temperature", "op": "semantic", "value": 55},
            ],
            "limit": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] > 0
        # All rows should support 55C (min <= 55 AND max >= 55)
        for row in data["rows"]:
            assert row["operating_temp_min_c"] <= 55
            assert row["operating_temp_max_c"] >= 55
        # applied_filters should show the expanded contains_range
        assert len(data["applied_filters"]) == 1
        assert data["applied_filters"][0]["op"] == "contains_range"

    def test_search_acceptance_query_3(self, mem_client):
        """Production + >=24Gb + x8 + >=6400 MT/s."""
        r = mem_client.post("/datasets/semiconductor_memory/search", json={
            "filters": [
                {"field": "part_status", "op": "eq", "value": "Production"},
                {"field": "component_density_gb", "op": "gte", "value": 24},
                {"field": "bus_width_bits", "op": "eq", "value": 8},
                {"field": "data_rate_mt_s", "op": "gte", "value": 6400},
            ],
            "limit": 10,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] > 0
        for row in data["rows"]:
            assert row["part_status"] == "Production"
            assert row["component_density_gb"] >= 24
            assert row["bus_width_bits"] == 8
            assert row["data_rate_mt_s"] >= 6400

    def test_search_with_sort(self, mem_client):
        r = mem_client.post("/datasets/semiconductor_memory/search", json={
            "filters": [],
            "sort": [{"field": "data_rate_mt_s", "descending": True}],
            "limit": 5,
        })
        assert r.status_code == 200
        data = r.json()
        rates = [row["data_rate_mt_s"] for row in data["rows"]]
        assert rates == sorted(rates, reverse=True)

    def test_search_with_columns_subset(self, mem_client):
        r = mem_client.post("/datasets/semiconductor_memory/search", json={
            "filters": [],
            "columns": ["product_id", "product_family", "data_rate_mt_s"],
            "limit": 3,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["columns"] == ["product_id", "product_family", "data_rate_mt_s"]

    def test_search_unknown_field_returns_400(self, mem_client):
        r = mem_client.post("/datasets/semiconductor_memory/search", json={
            "filters": [{"field": "nonexistent_col", "op": "eq", "value": 1}],
            "limit": 5,
        })
        assert r.status_code == 400

    def test_search_unknown_dataset_returns_400(self, mem_client):
        r = mem_client.post("/datasets/nonexistent/search", json={
            "filters": [], "limit": 5,
        })
        assert r.status_code == 400


class TestPredictOnSearchAPI:
    """Tests for the predict-on-search REST endpoint."""

    @pytest.fixture
    def supply_client(self, tmp_artifacts):
        """Client with semiconductor_memory catalog + supply models registered."""
        services = AnistrophServices(
            dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
            model_registry_dir=tmp_artifacts / "artifacts" / "models",
        )
        # Register catalog
        services.register_dataset_from_config(
            "datasets/semiconductor_memory/dataset.yaml",
            "data/semiconductor_memory/data.csv",
            parquet_path=str(tmp_artifacts / "data" / "processed" / "semiconductor_memory.parquet"),
        )
        # Register supply datasets
        services.register_dataset_from_config(
            "datasets/semiconductor_memory_supply_risk/dataset.yaml",
            "data/semiconductor_memory_supply/data.parquet",
            parquet_path=str(tmp_artifacts / "data" / "processed" / "supply_risk.parquet"),
        )
        services.register_dataset_from_config(
            "datasets/semiconductor_memory_supply_lead_time/dataset.yaml",
            "data/semiconductor_memory_supply/data.parquet",
            parquet_path=str(tmp_artifacts / "data" / "processed" / "supply_lt.parquet"),
        )
        # Train both models
        services.train("semiconductor_memory_supply_risk", "supply_risk_next_4w",
                       "xgboost", model_id="test-mem-risk")
        services.train("semiconductor_memory_supply_lead_time", "lead_time_next_4w_days",
                       "xgboost_regressor", model_id="test-mem-lt")
        svc_mod._services = services
        app = create_app()
        yield TestClient(app)
        svc_mod._services = None

    def test_predict_on_search_classification(self, supply_client):
        """Search DDR5 components and rank by supply risk."""
        r = supply_client.post("/datasets/semiconductor_memory/predict-on-search", json={
            "model_id": "test-mem-risk",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
            ],
            "limit": 5,
            "columns": ["product_id", "product_family", "data_rate_mt_s"],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["model_type"] == "classification"
        assert data["matched"] > 0
        assert data["returned"] <= 5
        # Each row should have a prediction (probability)
        for row in data["rows"]:
            assert "prediction" in row
            assert row["prediction"] is not None
            assert "prediction_label" in row
        # Rows should be sorted by prediction descending
        preds = [row["prediction"] for row in data["rows"] if row["prediction"] is not None]
        assert preds == sorted(preds, reverse=True)

    def test_predict_on_search_regression(self, supply_client):
        """Search DDR5 components and rank by lead time."""
        r = supply_client.post("/datasets/semiconductor_memory/predict-on-search", json={
            "model_id": "test-mem-lt",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
            ],
            "limit": 5,
            "columns": ["product_id", "data_rate_mt_s"],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["model_type"] == "regression"
        assert data["matched"] > 0
        for row in data["rows"]:
            assert "prediction" in row
            assert row["prediction"] is not None
        # Rows should be sorted by prediction descending
        preds = [row["prediction"] for row in data["rows"] if row["prediction"] is not None]
        assert preds == sorted(preds, reverse=True)

    def test_predict_on_search_with_semantic_filter(self, supply_client):
        """Search with semantic temperature filter + predict."""
        r = supply_client.post("/datasets/semiconductor_memory/predict-on-search", json={
            "model_id": "test-mem-risk",
            "filters": [
                {"field": "operating_temperature", "op": "semantic", "value": 55},
            ],
            "limit": 3,
            "columns": ["product_id"],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] > 0
        assert len(data["applied_filters"]) == 1
        assert data["applied_filters"][0]["op"] == "contains_range"

    def test_predict_on_search_unknown_model_returns_400(self, supply_client):
        r = supply_client.post("/datasets/semiconductor_memory/predict-on-search", json={
            "model_id": "nonexistent",
            "filters": [],
            "limit": 5,
        })
        assert r.status_code == 400

    def test_predict_on_search_no_matches(self, supply_client):
        """Search with filters that match nothing → empty result."""
        r = supply_client.post("/datasets/semiconductor_memory/predict-on-search", json={
            "model_id": "test-mem-risk",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "NONEXISTENT_FAMILY"},
            ],
            "limit": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] == 0
        assert data["returned"] == 0
        assert data["rows"] == []
