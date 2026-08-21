"""Integration tests — MCP tools."""

from __future__ import annotations

import json

import pytest

from backend.integrations.mcp.tools import call_tool, get_tool_list
from backend.services import AnistrophServices
import backend.services as svc_mod


@pytest.fixture
def services(tmp_artifacts, small_dataset):
    svc = AnistrophServices(
        dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_artifacts / "artifacts" / "models",
    )
    svc.register_dataset_from_config(
        "datasets/predictive_maintenance/dataset.yaml",
        str(tmp_artifacts / "data" / "synthetic" / "small.csv"),
        parquet_path=str(tmp_artifacts / "data" / "processed" / "small.parquet"),
    )
    svc.train("predictive_maintenance", "failure_within_horizon", "xgboost", model_id="mcp-test-xgb")
    svc_mod._services = svc
    yield svc
    svc_mod._services = None


class TestToolDiscovery:
    def test_list_tools(self):
        tools = get_tool_list()
        names = [t.name for t in tools]
        assert "anistroph_list_datasets" in names
        assert "anistroph_profile_dataset" in names
        assert "anistroph_slice_data" in names
        assert "anistroph_compare_data" in names
        assert "anistroph_list_models" in names
        assert "anistroph_get_model_metrics" in names
        assert "anistroph_predict" in names
        assert "anistroph_explain_prediction" in names
        assert "anistroph_sample_rows" in names
        assert "anistroph_find_interesting_slices" in names
        assert "anistroph_evaluate_model" in names
        assert "anistroph_find_evaluation_slices" in names
        assert "anistroph_get_search_contract" in names
        assert "anistroph_search" in names
        assert "anistroph_predict_on_search" in names

    def test_tool_schemas(self):
        tools = get_tool_list()
        for t in tools:
            assert t.input_schema is not None
            assert "type" in t.input_schema


class TestToolCalls:
    async def test_list_datasets(self, services):
        result = await call_tool("anistroph_list_datasets", {})
        data = json.loads(result[0].text)
        assert len(data) >= 1
        assert data[0]["dataset_id"] == "predictive_maintenance"

    async def test_profile_dataset(self, services):
        result = await call_tool("anistroph_profile_dataset", {"dataset_id": "predictive_maintenance"})
        data = json.loads(result[0].text)
        assert "row_count" in data

    async def test_slice_data(self, services):
        result = await call_tool("anistroph_slice_data", {
            "dataset_id": "predictive_maintenance",
            "dimensions": ["machine_type"],
            "metric": "failure",
            "aggregation": "mean",
        })
        data = json.loads(result[0].text)
        assert len(data) > 0

    async def test_compare_data(self, services):
        result = await call_tool("anistroph_compare_data", {
            "dataset_id": "predictive_maintenance",
            "dimension": "machine_type",
            "metric": "failure",
            "aggregation": "mean",
        })
        data = json.loads(result[0].text)
        assert len(data) > 0

    async def test_list_models(self, services):
        result = await call_tool("anistroph_list_models", {})
        data = json.loads(result[0].text)
        assert any(m["model_id"] == "mcp-test-xgb" for m in data)

    async def test_get_model_metrics(self, services):
        result = await call_tool("anistroph_get_model_metrics", {"model_id": "mcp-test-xgb"})
        data = json.loads(result[0].text)
        assert "roc_auc" in data

    async def test_predict(self, services):
        result = await call_tool("anistroph_predict", {
            "model_id": "mcp-test-xgb",
            "entity_id": "TOOL_000",
            "timestamp": "2026-06-05T12:00:00",
        })
        data = json.loads(result[0].text)
        assert "probability" in data

    async def test_explain_prediction(self, services):
        result = await call_tool("anistroph_explain_prediction", {
            "model_id": "mcp-test-xgb",
            "entity_id": "TOOL_000",
            "timestamp": "2026-06-05T12:00:00",
            "top_k": 5,
        })
        data = json.loads(result[0].text)
        assert "top_drivers" in data

    async def test_sample_rows_default(self, services):
        result = await call_tool("anistroph_sample_rows", {"dataset_id": "predictive_maintenance"})
        data = json.loads(result[0].text)
        assert data["dataset_id"] == "predictive_maintenance"
        assert data["row_count"] > 0
        assert 0 < data["returned"] <= 10
        assert isinstance(data["rows"], list)
        assert len(data["rows"]) == data["returned"]
        assert data["columns"]

    async def test_sample_rows_n_and_columns(self, services):
        result = await call_tool("anistroph_sample_rows", {
            "dataset_id": "predictive_maintenance",
            "n": 3,
            "columns": ["machine_id", "machine_type"],
        })
        data = json.loads(result[0].text)
        assert data["returned"] <= 3
        assert data["columns"] == ["machine_id", "machine_type"]
        for row in data["rows"]:
            assert set(row.keys()) == {"machine_id", "machine_type"}

    async def test_sample_rows_filter_equality(self, services):
        result = await call_tool("anistroph_sample_rows", {
            "dataset_id": "predictive_maintenance",
            "filters": {"machine_type": "TYPE_A"},
            "n": 5,
        })
        data = json.loads(result[0].text)
        assert data["row_count"] > 0
        for row in data["rows"]:
            assert row["machine_type"] == "TYPE_A"

    async def test_sample_rows_filter_in_list(self, services):
        result = await call_tool("anistroph_sample_rows", {
            "dataset_id": "predictive_maintenance",
            "filters": {"machine_type": ["TYPE_A", "TYPE_B"]},
            "n": 5,
        })
        data = json.loads(result[0].text)
        for row in data["rows"]:
            assert row["machine_type"] in ("TYPE_A", "TYPE_B")

    async def test_sample_rows_sort(self, services):
        result = await call_tool("anistroph_sample_rows", {
            "dataset_id": "predictive_maintenance",
            "sort_by": "temperature",
            "descending": True,
            "n": 5,
        })
        data = json.loads(result[0].text)
        temps = [r["temperature"] for r in data["rows"]]
        assert temps == sorted(temps, reverse=True)

    async def test_sample_rows_n_capped(self, services):
        result = await call_tool("anistroph_sample_rows", {
            "dataset_id": "predictive_maintenance",
            "n": 100000,
        })
        data = json.loads(result[0].text)
        assert data["returned"] <= 1000

    async def test_sample_rows_unknown_filter_column(self, services):
        result = await call_tool("anistroph_sample_rows", {
            "dataset_id": "predictive_maintenance",
            "filters": {"not_a_column": "x"},
        })
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_sample_rows_unknown_dataset(self, services):
        result = await call_tool("anistroph_sample_rows", {"dataset_id": "nonexistent"})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_invalid_tool(self, services):
        result = await call_tool("nonexistent_tool", {})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_invalid_input(self, services):
        result = await call_tool("anistroph_profile_dataset", {"dataset_id": "nonexistent"})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_evaluate_model(self, services):
        result = await call_tool("anistroph_evaluate_model", {
            "model_id": "mcp-test-xgb",
            "sample_size": 10,
        })
        data = json.loads(result[0].text)
        assert data["model_id"] == "mcp-test-xgb"
        assert data["dataset_id"] == "predictive_maintenance"
        assert data["eval_row_count"] > 0
        assert "metrics" in data
        assert "roc_auc" in data["metrics"]
        assert isinstance(data["predictions_sample"], list)
        assert len(data["predictions_sample"]) <= 10

    async def test_evaluate_model_not_found(self, services):
        result = await call_tool("anistroph_evaluate_model", {"model_id": "nonexistent"})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_find_evaluation_slices(self, services):
        result = await call_tool("anistroph_find_evaluation_slices", {
            "model_id": "mcp-test-xgb",
            "metric": "log_loss",
            "min_sample_size": 50,
            "top_k": 5,
        })
        data = json.loads(result[0].text)
        assert isinstance(data, list)
        # Classification model — log_loss slices should be returned.
        if len(data) > 0:
            s = data[0]
            assert "dimensions" in s
            assert "values" in s
            assert "row_count" in s
            assert "metric_value" in s
            assert "overall_baseline" in s


# ---------------------------------------------------------------------------
# Parametric search tools (semiconductor_memory)
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_services(tmp_artifacts):
    """Services with semiconductor_memory dataset registered for search tests."""
    svc = AnistrophServices(
        dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_artifacts / "artifacts" / "models",
    )
    svc.register_dataset_from_config(
        "datasets/semiconductor_memory/dataset.yaml",
        "data/semiconductor_memory/data.csv",
        parquet_path=str(tmp_artifacts / "data" / "processed" / "semiconductor_memory.parquet"),
    )
    svc_mod._services = svc
    yield svc
    svc_mod._services = None


class TestSearchTools:
    async def test_get_search_contract(self, mem_services):
        result = await call_tool("anistroph_get_search_contract", {
            "dataset_id": "semiconductor_memory",
        })
        data = json.loads(result[0].text)
        assert data["dataset_id"] == "semiconductor_memory"
        assert "semantic" in data["supported_operators"]
        assert len(data["searchable_fields"]) > 0
        semantic_names = [s["name"] for s in data["semantic_filters"]]
        assert "operating_temperature" in semantic_names
        assert "industrial_temperature" in semantic_names

    async def test_search_acceptance_query_1(self, mem_services):
        """DDR5_COMPONENT + x8 + >=6400 MT/s."""
        result = await call_tool("anistroph_search", {
            "dataset_id": "semiconductor_memory",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
                {"field": "bus_width_bits", "op": "eq", "value": 8},
                {"field": "data_rate_mt_s", "op": "gte", "value": 6400},
            ],
            "limit": 10,
        })
        data = json.loads(result[0].text)
        assert data["matched"] > 0
        for row in data["rows"]:
            assert row["product_family"] == "DDR5_COMPONENT"
            assert row["bus_width_bits"] == 8
            assert row["data_rate_mt_s"] >= 6400

    async def test_search_semantic_temperature(self, mem_services):
        """supports 55C via semantic operating_temperature filter."""
        result = await call_tool("anistroph_search", {
            "dataset_id": "semiconductor_memory",
            "filters": [
                {"field": "operating_temperature", "op": "semantic", "value": 55},
            ],
            "limit": 5,
        })
        data = json.loads(result[0].text)
        assert data["matched"] > 0
        for row in data["rows"]:
            assert row["operating_temp_min_c"] <= 55
            assert row["operating_temp_max_c"] >= 55
        # applied_filters should show the expanded contains_range
        assert len(data["applied_filters"]) == 1
        assert data["applied_filters"][0]["op"] == "contains_range"

    async def test_search_unknown_semantic_returns_error(self, mem_services):
        result = await call_tool("anistroph_search", {
            "dataset_id": "semiconductor_memory",
            "filters": [
                {"field": "nonexistent_semantic", "op": "semantic", "value": 1},
            ],
            "limit": 5,
        })
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_search_no_config_dataset(self, services):
        """predictive_maintenance has no search config → error."""
        result = await call_tool("anistroph_get_search_contract", {
            "dataset_id": "predictive_maintenance",
        })
        data = json.loads(result[0].text)
        assert "error" in data


# ---------------------------------------------------------------------------
# Predict-on-search tools (semiconductor_memory + supply models)
# ---------------------------------------------------------------------------

@pytest.fixture
def supply_services(tmp_artifacts):
    """Services with catalog + supply datasets and trained models."""
    svc = AnistrophServices(
        dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_artifacts / "artifacts" / "models",
    )
    svc.register_dataset_from_config(
        "datasets/semiconductor_memory/dataset.yaml",
        "data/semiconductor_memory/data.csv",
        parquet_path=str(tmp_artifacts / "data" / "processed" / "semiconductor_memory.parquet"),
    )
    svc.register_dataset_from_config(
        "datasets/semiconductor_memory_supply_risk/dataset.yaml",
        "data/semiconductor_memory_supply/data.parquet",
        parquet_path=str(tmp_artifacts / "data" / "processed" / "supply_risk.parquet"),
    )
    svc.register_dataset_from_config(
        "datasets/semiconductor_memory_supply_lead_time/dataset.yaml",
        "data/semiconductor_memory_supply/data.parquet",
        parquet_path=str(tmp_artifacts / "data" / "processed" / "supply_lt.parquet"),
    )
    svc.train("semiconductor_memory_supply_risk", "supply_risk_next_4w",
              "xgboost", model_id="mcp-mem-risk")
    svc.train("semiconductor_memory_supply_lead_time", "lead_time_next_4w_days",
              "xgboost_regressor", model_id="mcp-mem-lt")
    svc_mod._services = svc
    yield svc
    svc_mod._services = None


class TestPredictOnSearchTools:
    async def test_predict_on_search_discovery(self, supply_services):
        """The anistroph_predict_on_search tool is listed."""
        tools = get_tool_list()
        names = [t.name for t in tools]
        assert "anistroph_predict_on_search" in names

    async def test_predict_on_search_classification(self, supply_services):
        """Search DDR5 components and rank by supply risk via MCP."""
        result = await call_tool("anistroph_predict_on_search", {
            "dataset_id": "semiconductor_memory",
            "model_id": "mcp-mem-risk",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
            ],
            "limit": 5,
            "columns": ["product_id", "product_family", "data_rate_mt_s"],
        })
        data = json.loads(result[0].text)
        assert data["model_type"] == "classification"
        assert data["matched"] > 0
        assert data["returned"] <= 5
        for row in data["rows"]:
            assert "prediction" in row
            assert row["prediction"] is not None

    async def test_predict_on_search_regression(self, supply_services):
        """Search DDR5 components and rank by lead time via MCP."""
        result = await call_tool("anistroph_predict_on_search", {
            "dataset_id": "semiconductor_memory",
            "model_id": "mcp-mem-lt",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
            ],
            "limit": 3,
            "columns": ["product_id"],
        })
        data = json.loads(result[0].text)
        assert data["model_type"] == "regression"
        assert data["matched"] > 0
        for row in data["rows"]:
            assert row["prediction"] is not None

    async def test_predict_on_search_with_acceptance_query(self, supply_services):
        """Phase 2 acceptance: DDR5 x8 >=6400 ranked by supply risk."""
        result = await call_tool("anistroph_predict_on_search", {
            "dataset_id": "semiconductor_memory",
            "model_id": "mcp-mem-risk",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
                {"field": "bus_width_bits", "op": "eq", "value": 8},
                {"field": "data_rate_mt_s", "op": "gte", "value": 6400},
            ],
            "limit": 5,
            "columns": ["product_id", "data_rate_mt_s", "component_density_gb"],
        })
        data = json.loads(result[0].text)
        assert data["matched"] > 0
        # All results should match the search filters
        for row in data["rows"]:
            assert row["data_rate_mt_s"] >= 6400

    async def test_predict_on_search_unknown_model(self, supply_services):
        result = await call_tool("anistroph_predict_on_search", {
            "dataset_id": "semiconductor_memory",
            "model_id": "nonexistent",
            "filters": [],
            "limit": 5,
        })
        data = json.loads(result[0].text)
        assert "error" in data
