"""Integration tests — MCP tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

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


@pytest.fixture
def mem_services(tmp_artifacts):
    """Services with semiconductor_memory dataset registered."""
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


@pytest.fixture
def supply_services(tmp_artifacts):
    """Services with semiconductor_memory catalog + supply models registered."""
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
    svc.train("semiconductor_memory_supply_risk", "supply_risk_next_4w", "xgboost", model_id="mcp-mem-risk")
    svc.train("semiconductor_memory_supply_lead_time", "lead_time_next_4w_days", "xgboost_regressor", model_id="mcp-mem-lt")
    svc_mod._services = svc
    yield svc
    svc_mod._services = None


@pytest.fixture
def ext_registry(tmp_path, monkeypatch):
    """Temp external tool registry for MCP tests."""
    from backend.integrations import registry as reg_mod
    from backend.integrations import a2a as a2a_mod
    from backend.integrations.mcp import tools as mcp_tools_mod
    from backend.integrations.registry import ExternalToolRegistry

    registry_path = tmp_path / "tool_registry.yaml"
    registry_path.write_text("""
tools:
  - name: call_test_agent
    provider: test
    capability: test_research
    visibility: always
    description: Test agent.
    keywords: [test]
    llm_parameters:
      type: object
      properties: {prompt: {type: string}}
      required: [prompt]
    agent_owner: test
    protocol: A2A_JSONRPC
    base_url: https://test.example.com
    path: /agents/test/
""")
    temp_reg = ExternalToolRegistry(registry_path)
    monkeypatch.setattr(reg_mod, "get_external_tool_registry", lambda: temp_reg)
    monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: temp_reg)
    monkeypatch.setattr(mcp_tools_mod, "get_external_tool_registry", lambda: temp_reg, raising=False)
    yield temp_reg


class TestToolDiscovery:
    def test_list_tools_includes_native_and_search(self):
        names = [t.name for t in get_tool_list()]
        assert "anistroph_list_datasets" in names
        assert "anistroph_search" in names
        assert "anistroph_predict_on_search" in names


class TestNativeMCPTools:
    async def test_list_datasets_and_predict(self, services):
        result = await call_tool("anistroph_list_datasets", {})
        data = json.loads(result[0].text)
        assert len(data) >= 1

        result = await call_tool("anistroph_predict", {
            "model_id": "mcp-test-xgb", "entity_id": "TOOL_000",
            "timestamp": "2026-06-05T12:00:00",
        })
        data = json.loads(result[0].text)
        assert "probability" in data

    async def test_sample_rows(self, services):
        result = await call_tool("anistroph_sample_rows", {
            "dataset_id": "predictive_maintenance", "n": 5,
        })
        data = json.loads(result[0].text)
        assert len(data["rows"]) <= 5


class TestSearchMCPTools:
    async def test_search_contract(self, mem_services):
        result = await call_tool("anistroph_get_search_contract", {"dataset_id": "semiconductor_memory"})
        data = json.loads(result[0].text)
        assert "semantic" in data["supported_operators"]

    async def test_search_acceptance_query(self, mem_services):
        result = await call_tool("anistroph_search", {
            "dataset_id": "semiconductor_memory",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
                {"field": "data_rate_mt_s", "op": "gte", "value": 6400},
            ],
            "limit": 5,
        })
        data = json.loads(result[0].text)
        assert data["matched"] > 0


class TestPredictOnSearchMCP:
    async def test_predict_on_search(self, supply_services):
        result = await call_tool("anistroph_predict_on_search", {
            "dataset_id": "semiconductor_memory",
            "model_id": "mcp-mem-risk",
            "filters": [{"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"}],
            "limit": 5, "columns": ["product_id"],
        })
        data = json.loads(result[0].text)
        assert data["matched"] > 0
        assert data["model_type"] == "classification"


class TestExternalToolMCP:
    async def test_external_tool_discovered(self, ext_registry):
        names = [t.name for t in get_tool_list()]
        assert "call_test_agent" in names

    async def test_external_tool_validation_error(self, ext_registry, services):
        result = await call_tool("call_test_agent", {})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "validation" in data["error"]

    async def test_external_tool_unresolved_url(self, ext_registry, services):
        # Override the tool's base_url with an unresolved env placeholder.
        tool = ext_registry.get("call_test_agent")
        tool.base_url = "${UNRESOLVED_HOST}"
        result = await call_tool("call_test_agent", {"prompt": "test"})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "unresolved" in data["error"]

    async def test_external_tool_connection_error_soft_fails(self, ext_registry, services, monkeypatch):
        """When the A2A agent is unreachable, return a soft-fail message."""
        import httpx
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("DNS resolution failed")
        mock_client.close = MagicMock()
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)
        result = await call_tool("call_test_agent", {"prompt": "test"})
        data = json.loads(result[0].text)
        assert data["state"] == "unavailable"
        assert "not available" in data["message"]
