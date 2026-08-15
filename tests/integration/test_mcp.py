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

    async def test_invalid_tool(self, services):
        result = await call_tool("nonexistent_tool", {})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_invalid_input(self, services):
        result = await call_tool("anistroph_profile_dataset", {"dataset_id": "nonexistent"})
        data = json.loads(result[0].text)
        assert "error" in data
