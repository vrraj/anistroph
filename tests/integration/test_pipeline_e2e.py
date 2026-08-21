"""End-to-end integration tests for the full semiconductor memory pipeline:
search -> predict -> A2A (Aina-Veris).

Uses the three products with dedicated datasheet PDFs in
product-specifications/sample-data/:
  ANM-D5C-0001 (32Gb, 4800 MT/s, x4, high supply risk)
  ANM-D5C-0002 (32Gb, 7200 MT/s, x4, low supply risk)
  ANM-D5C-0003 (32Gb, 4800 MT/s, x16, high supply risk)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.services import AnistrophServices
import backend.services as svc_mod

DATASHEET_PRODUCTS = ["ANM-D5C-0001", "ANM-D5C-0002", "ANM-D5C-0003"]
DATASHEET_DIR = Path(__file__).resolve().parent.parent.parent / "product-specifications" / "sample-data"
EXPECTED_DATASHEETS = [
    "ANM-D5C-0001_datasheet.pdf", "ANM-D5C-0002_datasheet.pdf", "ANM-D5C-0003_datasheet.pdf",
    "DDR5_Component_Family_Guide.pdf", "DDR5_Module_Family_Guide.pdf",
]


@pytest.fixture
def pipeline_services(tmp_artifacts):
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
    svc.train("semiconductor_memory_supply_risk", "supply_risk_next_4w", "xgboost", model_id="e2e-mem-risk")
    svc.train("semiconductor_memory_supply_lead_time", "lead_time_next_4w_days", "xgboost_regressor", model_id="e2e-mem-lt")
    svc_mod._services = svc
    yield svc
    svc_mod._services = None


@pytest.fixture
def ext_registry(tmp_path, monkeypatch):
    from backend.integrations import registry as reg_mod
    from backend.integrations import a2a as a2a_mod
    from backend.integrations.mcp import tools as mcp_tools_mod
    from backend.api import integrations as api_integ_mod
    from backend.integrations.registry import ExternalToolRegistry

    registry_path = tmp_path / "tool_registry.yaml"
    registry_path.write_text("""
tools:
  - name: call_veris_semiconductor_research_agent
    provider: veris
    capability: semiconductor_memory_research
    visibility: always
    description: Query Aina-Veris for grounded semiconductor-memory datasheet analysis.
    keywords: [semiconductor memory, datasheet]
    llm_parameters:
      type: object
      properties: {prompt: {type: string, description: Technical research question.}}
      required: [prompt]
      additionalProperties: false
    agent_owner: aina-veris
    protocol: A2A_JSONRPC
    base_url: http://test-aina-veris:8100
    path: /agents/aina-veris/
""")
    temp_reg = ExternalToolRegistry(registry_path)
    monkeypatch.setattr(reg_mod, "get_external_tool_registry", lambda: temp_reg)
    monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: temp_reg)
    monkeypatch.setattr(mcp_tools_mod, "get_external_tool_registry", lambda: temp_reg, raising=False)
    monkeypatch.setattr(api_integ_mod, "get_external_tool_registry", lambda: temp_reg)
    yield temp_reg


class TestDatasheetReferenceData:
    def test_datasheets_exist_and_products_in_catalog(self, pipeline_services):
        """The 5 reference PDFs exist and the 3 datasheet products are in the catalog."""
        for filename in EXPECTED_DATASHEETS:
            assert (DATASHEET_DIR / filename).exists()
        from backend.search.filters import FilterExpression
        for pid in DATASHEET_PRODUCTS:
            result = pipeline_services.search(
                "semiconductor_memory",
                filters=[FilterExpression(field="product_id", op="eq", value=pid)],
                columns=["product_id", "datasheet_id", "product_family"],
                limit=1,
            )
            assert result["matched"] == 1
            assert result["rows"][0]["datasheet_id"] == "DS-" + pid


class TestSearchPredictPipeline:
    def test_datasheet_products_match_broad_query(self, pipeline_services):
        """All 3 datasheet products match: DDR5_COMPONENT + >=24Gb + Production + 55C."""
        from backend.search.filters import FilterExpression
        result = pipeline_services.predict_on_search(
            search_dataset_id="semiconductor_memory",
            model_id="e2e-mem-risk",
            filters=[
                FilterExpression(field="product_family", op="eq", value="DDR5_COMPONENT"),
                FilterExpression(field="component_density_gb", op="gte", value=24),
                FilterExpression(field="part_status", op="eq", value="Production"),
                FilterExpression(field="operating_temperature", op="semantic", value=55),
            ],
            limit=500,
            columns=["product_id", "datasheet_id"],
        )
        matched_ids = {r["product_id"] for r in result["rows"]}
        for pid in DATASHEET_PRODUCTS:
            assert pid in matched_ids

    def test_datasheet_products_have_distinct_risk_profiles(self, pipeline_services):
        """ANM-D5C-0002 is low risk; ANM-D5C-0001 and ANM-D5C-0003 are high risk."""
        svc = pipeline_services
        for pid, expected_low in [("ANM-D5C-0002", True), ("ANM-D5C-0001", False), ("ANM-D5C-0003", False)]:
            pred = svc.predict(model_id="e2e-mem-risk", entity_id=pid, timestamp="2025-06-23")
            prob = pred.get("probability", 0)
            if expected_low:
                assert prob < 0.5, f"{pid} should be low risk, got {prob}"
            else:
                assert prob > 0.5, f"{pid} should be high risk, got {prob}"


class TestFullPipelineSearchPredictRAG:
    def test_full_pipeline_via_service(self, pipeline_services, ext_registry, monkeypatch):
        """Full pipeline: search -> predict -> identify lowest-risk -> A2A invoke."""
        from backend.integrations.a2a import invoke_external_tool
        import httpx
        from backend.search.filters import FilterExpression

        svc = pipeline_services
        result = svc.predict_on_search(
            search_dataset_id="semiconductor_memory",
            model_id="e2e-mem-risk",
            filters=[
                FilterExpression(field="product_family", op="eq", value="DDR5_COMPONENT"),
                FilterExpression(field="component_density_gb", op="gte", value=24),
                FilterExpression(field="part_status", op="eq", value="Production"),
                FilterExpression(field="operating_temperature", op="semantic", value=55),
            ],
            limit=500,
            columns=["product_id", "datasheet_id"],
        )
        lowest_risk = result["rows"][-3:]
        prompt = "Compare power-management for: " + ", ".join(
            f"{r['product_id']} ({r['datasheet_id']})" for r in lowest_risk
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"id": "task-e2e-1", "state": "completed",
                       "artifacts": [{"type": "text", "text": "Done."}]},
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.close = MagicMock()
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        a2a_result = invoke_external_tool("call_veris_semiconductor_research_agent", {"prompt": prompt})
        assert a2a_result["state"] == "completed"

        body = mock_client.post.call_args.kwargs["json"]
        assert body["method"] == "SendMessage"
        sent_text = body["params"]["message"]["parts"][0]["text"]
        for r in lowest_risk:
            assert r["product_id"] in sent_text

    def test_full_pipeline_via_rest(self, pipeline_services, ext_registry, monkeypatch):
        """Full pipeline via REST: search -> predict -> A2A invoke."""
        from fastapi.testclient import TestClient
        from backend.main import create_app
        import httpx

        client = TestClient(create_app())
        r = client.post("/datasets/semiconductor_memory/predict-on-search", json={
            "model_id": "e2e-mem-risk",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
                {"field": "component_density_gb", "op": "gte", "value": 24},
                {"field": "part_status", "op": "eq", "value": "Production"},
            ],
            "limit": 500, "columns": ["product_id", "datasheet_id"],
        })
        assert r.status_code == 200
        lowest_risk = r.json()["rows"][-3:]
        prompt = "Compare: " + ", ".join(f"{r['product_id']}" for r in lowest_risk)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"id": "task-rest-1", "state": "completed",
                       "artifacts": [{"type": "text", "text": "Done."}]},
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.close = MagicMock()
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        r2 = client.post("/integrations/tools/call_veris_semiconductor_research_agent/invoke", json={
            "arguments": {"prompt": prompt},
        })
        assert r2.status_code == 200
        assert r2.json()["state"] == "completed"

    async def test_full_pipeline_via_mcp(self, pipeline_services, ext_registry, monkeypatch):
        """Full pipeline via MCP: search -> predict -> A2A invoke."""
        import httpx
        from backend.integrations.mcp.tools import call_tool

        result = await call_tool("anistroph_predict_on_search", {
            "dataset_id": "semiconductor_memory",
            "model_id": "e2e-mem-risk",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
                {"field": "component_density_gb", "op": "gte", "value": 24},
            ],
            "limit": 500, "columns": ["product_id", "datasheet_id"],
        })
        data = json.loads(result[0].text)
        assert data["matched"] > 0

        lowest_risk = data["rows"][-3:]
        prompt = "Compare: " + ", ".join(r["product_id"] for r in lowest_risk)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"id": "task-mcp-1", "state": "completed",
                       "artifacts": [{"type": "text", "text": "Done."}]},
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.close = MagicMock()
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        a2a_result = await call_tool("call_veris_semiconductor_research_agent", {"prompt": prompt})
        a2a_data = json.loads(a2a_result[0].text)
        assert a2a_data["state"] == "completed"
