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
