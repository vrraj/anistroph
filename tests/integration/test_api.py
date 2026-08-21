"""Integration tests — REST API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from backend.main import create_app
from backend.services import AnistrophServices
import backend.services as svc_mod


@pytest.fixture
def client(tmp_artifacts, small_dataset):
    services = AnistrophServices(
        dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_artifacts / "artifacts" / "models",
    )
    services.register_dataset_from_config(
        "datasets/predictive_maintenance/dataset.yaml",
        str(tmp_artifacts / "data" / "synthetic" / "small.csv"),
        parquet_path=str(tmp_artifacts / "data" / "processed" / "small.parquet"),
    )
    svc_mod._services = services
    app = create_app()
    yield TestClient(app)
    svc_mod._services = None


@pytest.fixture
def mem_client(tmp_artifacts):
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


@pytest.fixture
def supply_client(tmp_artifacts):
    """Client with semiconductor_memory catalog + supply models registered."""
    services = AnistrophServices(
        dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_artifacts / "artifacts" / "models",
    )
    services.register_dataset_from_config(
        "datasets/semiconductor_memory/dataset.yaml",
        "data/semiconductor_memory/data.csv",
        parquet_path=str(tmp_artifacts / "data" / "processed" / "semiconductor_memory.parquet"),
    )
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
    services.train("semiconductor_memory_supply_risk", "supply_risk_next_4w", "xgboost", model_id="test-mem-risk")
    services.train("semiconductor_memory_supply_lead_time", "lead_time_next_4w_days", "xgboost_regressor", model_id="test-mem-lt")
    svc_mod._services = services
    app = create_app()
    yield TestClient(app)
    svc_mod._services = None


@pytest.fixture
def integ_client(tmp_artifacts, monkeypatch):
    """Client with a temp external tool registry."""
    registry_path = tmp_artifacts / "integrations" / "tool_registry.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("""
tools:
  - name: call_test_agent
    provider: test
    capability: test_research
    visibility: always
    description: Test agent for integration tests.
    keywords: [test]
    llm_parameters:
      type: object
      properties:
        prompt: {type: string, description: A test prompt.}
      required: [prompt]
      additionalProperties: false
    agent_owner: test-owner
    protocol: A2A_JSONRPC
    base_url: https://test.example.com
    path: /agents/test-agent/
""")
    from backend.integrations import registry as reg_mod
    from backend.integrations import a2a as a2a_mod
    from backend.integrations.registry import ExternalToolRegistry
    from backend.api import integrations as api_integ_mod
    from backend.integrations.mcp import tools as mcp_tools_mod
    temp_reg = ExternalToolRegistry(registry_path)
    monkeypatch.setattr(reg_mod, "get_external_tool_registry", lambda: temp_reg)
    monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: temp_reg)
    monkeypatch.setattr(api_integ_mod, "get_external_tool_registry", lambda: temp_reg)
    monkeypatch.setattr(api_integ_mod, "invoke_external_tool", a2a_mod.invoke_external_tool)
    monkeypatch.setattr(mcp_tools_mod, "get_external_tool_registry", lambda: temp_reg, raising=False)
    services = AnistrophServices(
        dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_artifacts / "artifacts" / "models",
    )
    svc_mod._services = services
    app = create_app()
    yield TestClient(app)
    svc_mod._services = None


class TestCoreAPI:
    def test_health_and_list_datasets(self, client):
        assert client.get("/health").json()["status"] == "ok"
        data = client.get("/datasets").json()
        assert data[0]["dataset_id"] == "predictive_maintenance"

    def test_train_predict_explain(self, client):
        client.post("/models/train", json={
            "dataset_id": "predictive_maintenance", "target_name": "failure_within_horizon",
            "model_type": "xgboost", "model_id": "api-test",
        })
        r = client.post("/predictions", json={
            "model_id": "api-test", "entity_id": "TOOL_000", "timestamp": "2026-06-05T12:00:00",
        })
        assert "probability" in r.json()
        r = client.post("/predictions/explain", json={
            "model_id": "api-test", "entity_id": "TOOL_000", "timestamp": "2026-06-05T12:00:00", "top_k": 5,
        })
        assert "top_drivers" in r.json()

    def test_evaluate_model(self, client):
        client.post("/models/train", json={
            "dataset_id": "predictive_maintenance", "target_name": "failure_within_horizon",
            "model_type": "xgboost", "model_id": "api-eval",
        })
        r = client.post("/evaluations/api-eval", json={"sample_size": 10})
        assert r.status_code == 200
        assert "roc_auc" in r.json()["metrics"]


class TestSearchAPI:
    def test_search_contract(self, mem_client):
        r = mem_client.get("/datasets/semiconductor_memory/search-contract")
        assert r.status_code == 200
        assert "semantic" in r.json()["supported_operators"]

    def test_search_acceptance_query(self, mem_client):
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
        for row in data["rows"]:
            assert row["bus_width_bits"] == 8 and row["data_rate_mt_s"] >= 6400

    def test_search_semantic_temp(self, mem_client):
        r = mem_client.post("/datasets/semiconductor_memory/search", json={
            "filters": [{"field": "operating_temperature", "op": "semantic", "value": 55}],
            "limit": 5,
        })
        assert r.status_code == 200
        assert r.json()["applied_filters"][0]["op"] == "contains_range"


class TestPredictOnSearchAPI:
    def test_classification(self, supply_client):
        r = supply_client.post("/datasets/semiconductor_memory/predict-on-search", json={
            "model_id": "test-mem-risk",
            "filters": [{"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"}],
            "limit": 5, "columns": ["product_id"],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["model_type"] == "classification"
        assert data["matched"] > 0
        preds = [row["prediction"] for row in data["rows"]]
        assert preds == sorted(preds, reverse=True)

    def test_regression(self, supply_client):
        r = supply_client.post("/datasets/semiconductor_memory/predict-on-search", json={
            "model_id": "test-mem-lt",
            "filters": [{"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"}],
            "limit": 5, "columns": ["product_id"],
        })
        assert r.status_code == 200
        assert r.json()["model_type"] == "regression"


class TestIntegrationsAPI:
    def test_list_external_tools(self, integ_client):
        r = integ_client.get("/integrations/tools")
        assert r.status_code == 200
        assert r.json()[0]["name"] == "call_test_agent"

    def test_invoke_with_mocked_a2a(self, integ_client, monkeypatch):
        import httpx
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"id": "task-1", "state": "completed",
                       "artifacts": [{"type": "text", "text": "Result."}]},
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.close = MagicMock()
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)
        r = integ_client.post("/integrations/tools/call_test_agent/invoke", json={
            "arguments": {"prompt": "Compare DDR5 power management."},
        })
        assert r.status_code == 200
        assert r.json()["state"] == "completed"

    def test_invoke_errors(self, integ_client):
        assert integ_client.post("/integrations/tools/call_test_agent/invoke", json={}).status_code == 422
        assert integ_client.post("/integrations/tools/nonexistent/invoke", json={"arguments": {}}).status_code == 404
