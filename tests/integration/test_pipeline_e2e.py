"""End-to-end integration tests for the full semiconductor memory pipeline:
search → predict → A2A (Aina-Veris).

These tests use the actual reference datasheets in
product-specifications/sample-data/ and the three products that have
dedicated datasheet PDFs:
  - ANM-D5C-0001 (DDR5 component, 32Gb, 4800 MT/s, x4, -40..95°C, Production)
  - ANM-D5C-0002 (DDR5 component, 32Gb, 7200 MT/s, x4, -40..95°C, Production)
  - ANM-D5C-0003 (DDR5 component, 32Gb, 4800 MT/s, x16, 0..95°C, Production)

All three match the broader query: DDR5_COMPONENT + >=24Gb + Production +
supports 55°C. They have distinct supply-risk profiles, which makes them
suitable for testing the predict-on-search ranking and the A2A handoff
(where the three lowest-risk products would be sent to Aina-Veris for
datasheet analysis).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.services import AnistrophServices
import backend.services as svc_mod

# ---------------------------------------------------------------------------
# Constants — the three products with dedicated datasheet PDFs
# ---------------------------------------------------------------------------

DATASHEET_PRODUCTS = ["ANM-D5C-0001", "ANM-D5C-0002", "ANM-D5C-0003"]
DATASHEET_IDS = ["DS-ANM-D5C-0001", "DS-ANM-D5C-0002", "DS-ANM-D5C-0003"]
DATASHEET_DIR = Path(__file__).resolve().parent.parent.parent / "product-specifications" / "sample-data"
EXPECTED_DATASHEETS = [
    "ANM-D5C-0001_datasheet.pdf",
    "ANM-D5C-0002_datasheet.pdf",
    "ANM-D5C-0003_datasheet.pdf",
    "DDR5_Component_Family_Guide.pdf",
    "DDR5_Module_Family_Guide.pdf",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_services(tmp_artifacts):
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
              "xgboost", model_id="e2e-mem-risk")
    svc.train("semiconductor_memory_supply_lead_time", "lead_time_next_4w_days",
              "xgboost_regressor", model_id="e2e-mem-lt")
    svc_mod._services = svc
    yield svc
    svc_mod._services = None


@pytest.fixture
def ext_registry(tmp_path, monkeypatch):
    """Temp external tool registry for A2A tests."""
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
    keywords:
      - semiconductor memory
      - datasheet
    llm_parameters:
      type: object
      properties:
        prompt:
          type: string
          description: Technical semiconductor-memory research question.
      required:
        - prompt
      additionalProperties: false
    agent_owner: aina-veris
    protocol: A2A_JSONRPC
    base_url: http://test-aina-veris:8100
    path: /agents/veris-semiconductor-research-agent/
""")
    temp_reg = ExternalToolRegistry(registry_path)
    monkeypatch.setattr(reg_mod, "get_external_tool_registry", lambda: temp_reg)
    monkeypatch.setattr(a2a_mod, "get_external_tool_registry", lambda: temp_reg)
    monkeypatch.setattr(mcp_tools_mod, "get_external_tool_registry", lambda: temp_reg, raising=False)
    monkeypatch.setattr(api_integ_mod, "get_external_tool_registry", lambda: temp_reg)
    yield temp_reg


# ---------------------------------------------------------------------------
# Datasheet reference data tests
# ---------------------------------------------------------------------------

class TestDatasheetReferenceData:
    """Verify the reference datasheets exist and match catalog product IDs."""

    def test_datasheet_pdfs_exist(self):
        """All five reference datasheet PDFs are present in sample-data/."""
        for filename in EXPECTED_DATASHEETS:
            path = DATASHEET_DIR / filename
            assert path.exists(), f"missing datasheet: {filename}"
            assert path.stat().st_size > 0, f"empty datasheet: {filename}"

    def test_datasheet_products_in_catalog(self, pipeline_services):
        """The three products with dedicated datasheets exist in the catalog."""
        svc = pipeline_services
        for pid in DATASHEET_PRODUCTS:
            result = svc.search(
                "semiconductor_memory",
                filters=[__import__("backend.search.filters", fromlist=["FilterExpression"]).FilterExpression(
                    field="product_id", op="eq", value=pid,
                )],
                limit=1,
            )
            assert result["matched"] == 1, f"{pid} not found in catalog"
            row = result["rows"][0]
            assert row["product_id"] == pid
            assert row["product_family"] == "DDR5_COMPONENT"
            assert row["component_density_gb"] == 32
            assert row["part_status"] == "Production"

    def test_datasheet_ids_match_product_ids(self, pipeline_services):
        """Each product's datasheet_id follows the DS-{product_id} convention."""
        svc = pipeline_services
        from backend.search.filters import FilterExpression
        for pid, expected_dsid in zip(DATASHEET_PRODUCTS, DATASHEET_IDS):
            result = svc.search(
                "semiconductor_memory",
                filters=[FilterExpression(field="product_id", op="eq", value=pid)],
                columns=["product_id", "datasheet_id"],
                limit=1,
            )
            assert result["rows"][0]["datasheet_id"] == expected_dsid


# ---------------------------------------------------------------------------
# Search → Predict pipeline tests (using datasheet products)
# ---------------------------------------------------------------------------

class TestSearchPredictPipeline:
    """Test parametric search + supply prediction for the datasheet products."""

    def test_datasheet_products_match_broad_query(self, pipeline_services):
        """All three datasheet products match: DDR5_COMPONENT + >=24Gb +
        Production + supports 55°C."""
        svc = pipeline_services
        from backend.search.filters import FilterExpression
        result = svc.search(
            "semiconductor_memory",
            filters=[
                FilterExpression(field="product_family", op="eq", value="DDR5_COMPONENT"),
                FilterExpression(field="component_density_gb", op="gte", value=24),
                FilterExpression(field="part_status", op="eq", value="Production"),
                FilterExpression(field="operating_temperature", op="semantic", value=55),
            ],
            limit=500,
        )
        matched_ids = {r["product_id"] for r in result["rows"]}
        for pid in DATASHEET_PRODUCTS:
            assert pid in matched_ids, f"{pid} should match the broad query"

    def test_predict_on_search_includes_datasheet_products(self, pipeline_services):
        """predict_on_search with the broad query includes all three
        datasheet products and they have predictions."""
        svc = pipeline_services
        from backend.search.filters import FilterExpression
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
            columns=["product_id", "datasheet_id", "component_density_gb", "data_rate_mt_s"],
        )
        result_ids = {r["product_id"] for r in result["rows"]}
        for pid in DATASHEET_PRODUCTS:
            assert pid in result_ids, f"{pid} missing from predict-on-search results"
        # Each datasheet product should have a non-None prediction
        for row in result["rows"]:
            if row["product_id"] in DATASHEET_PRODUCTS:
                assert row["prediction"] is not None
                assert "prediction_label" in row

    def test_datasheet_products_have_distinct_risk_profiles(self, pipeline_services):
        """The three datasheet products have known, distinct risk profiles:
        ANM-D5C-0001: high risk (prob ~1.0)
        ANM-D5C-0002: low risk (prob ~0.0)
        ANM-D5C-0003: high risk (prob ~1.0)
        """
        svc = pipeline_services
        predictions = {}
        for pid in DATASHEET_PRODUCTS:
            pred = svc.predict(model_id="e2e-mem-risk", entity_id=pid, timestamp="2025-06-23")
            predictions[pid] = pred.get("probability", 0)

        # ANM-D5C-0002 should be low risk (lowest of the three)
        assert predictions["ANM-D5C-0002"] < 0.5, "ANM-D5C-0002 should be low risk"
        # ANM-D5C-0001 and ANM-D5C-0003 should be high risk
        assert predictions["ANM-D5C-0001"] > 0.5, "ANM-D5C-0001 should be high risk"
        assert predictions["ANM-D5C-0003"] > 0.5, "ANM-D5C-0003 should be high risk"

    def test_ranking_by_risk_puts_high_risk_first(self, pipeline_services):
        """When ranked by supply risk (descending), high-risk products
        appear before low-risk products."""
        svc = pipeline_services
        from backend.search.filters import FilterExpression
        result = svc.predict_on_search(
            search_dataset_id="semiconductor_memory",
            model_id="e2e-mem-risk",
            filters=[
                FilterExpression(field="product_family", op="eq", value="DDR5_COMPONENT"),
                FilterExpression(field="component_density_gb", op="gte", value=24),
                FilterExpression(field="part_status", op="eq", value="Production"),
            ],
            limit=500,
            columns=["product_id"],
        )
        # Find positions of the three datasheet products
        positions = {row["product_id"]: i for i, row in enumerate(result["rows"])}
        # ANM-D5C-0002 (low risk) should come after ANM-D5C-0001 and ANM-D5C-0003 (high risk)
        if all(pid in positions for pid in DATASHEET_PRODUCTS):
            assert positions["ANM-D5C-0002"] > positions["ANM-D5C-0001"], \
                "low-risk product should rank after high-risk products"
            assert positions["ANM-D5C-0002"] > positions["ANM-D5C-0003"], \
                "low-risk product should rank after high-risk products"

    def test_lead_time_predictions_for_datasheet_products(self, pipeline_services):
        """Lead-time predictions for the three datasheet products are
        positive and within a reasonable range."""
        svc = pipeline_services
        for pid in DATASHEET_PRODUCTS:
            pred = svc.predict(model_id="e2e-mem-lt", entity_id=pid, timestamp="2025-06-23")
            lt = pred.get("predicted_yield", 0)
            assert lt > 0, f"{pid} lead time should be positive"
            assert lt < 200, f"{pid} lead time should be reasonable (< 200 days)"


# ---------------------------------------------------------------------------
# Full pipeline: Search → Predict → A2A (with mocked Aina-Veris)
# ---------------------------------------------------------------------------

class TestFullPipelineSearchPredictRAG:
    """Test the complete Phase 3 acceptance query: search → predict → A2A.

    These tests mock the Aina-Veris A2A endpoint since the real service
    may not be running. They verify that Anistroph correctly:
    1. Searches the catalog
    2. Ranks by supply risk
    3. Identifies the lowest-risk products
    4. Constructs a prompt with product_id / datasheet_id context
    5. Invokes the Aina-Veris A2A agent
    """

    def test_full_pipeline_via_service(self, pipeline_services, ext_registry, monkeypatch):
        """Full pipeline: search → predict → identify lowest-risk → A2A invoke.

        This mirrors the Phase 3 acceptance query:
        'Find qualifying DDR5 products, rank by supply risk, then compare
        the three lowest-risk products for power-management behavior and
        initialization requirements using their datasheets.'
        """
        from backend.integrations.a2a import invoke_external_tool
        import httpx

        svc = pipeline_services
        from backend.search.filters import FilterExpression

        # Step 1+2: Search and rank by supply risk.
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
            columns=["product_id", "datasheet_id", "component_density_gb", "data_rate_mt_s"],
        )
        assert result["matched"] > 0

        # Step 3: Identify the three lowest-risk products.
        # Results are sorted by risk descending, so take the last 3.
        lowest_risk = result["rows"][-3:]
        lowest_risk_ids = [r["product_id"] for r in lowest_risk]
        lowest_risk_dsids = [r["datasheet_id"] for r in lowest_risk]

        # All three should have low risk probabilities (< 0.5)
        for row in lowest_risk:
            assert row["prediction"] is not None
            assert row["prediction"] < 0.5, \
                f"{row['product_id']} should be low-risk, got {row['prediction']}"

        # Step 4: Construct the A2A prompt with product/datasheet context.
        prompt = (
            "Compare the following three DDR5 components for power-management "
            "behavior and initialization requirements using their datasheets:\n"
        )
        for pid, dsid in zip(lowest_risk_ids, lowest_risk_dsids):
            prompt += f"  - {pid} (datasheet: {dsid})\n"

        # Step 5: Mock the A2A invocation and verify the prompt is passed.
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {
                "id": "task-e2e-1",
                "state": "completed",
                "artifacts": [
                    {"type": "text", "text": "Power-management comparison complete."},
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.close = MagicMock()
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        a2a_result = invoke_external_tool(
            "call_veris_semiconductor_research_agent",
            {"prompt": prompt},
        )

        assert a2a_result["state"] == "completed"
        assert mock_client.post.called

        # Verify the prompt was passed in the A2A request body.
        call_args = mock_client.post.call_args
        body = call_args.kwargs["json"]
        assert body["method"] == "SendMessage"
        sent_text = body["params"]["message"]["parts"][0]["text"]
        # The prompt should contain all three product IDs and datasheet IDs.
        for pid in lowest_risk_ids:
            assert pid in sent_text, f"{pid} should be in the A2A prompt"
        for dsid in lowest_risk_dsids:
            assert dsid in sent_text, f"{dsid} should be in the A2A prompt"

    def test_full_pipeline_via_rest(self, pipeline_services, ext_registry, monkeypatch):
        """Full pipeline via REST: search → predict → A2A invoke."""
        from fastapi.testclient import TestClient
        from backend.main import create_app
        import httpx

        app = create_app()
        client = TestClient(app)

        # Step 1+2: Search and rank by supply risk via REST.
        r = client.post("/datasets/semiconductor_memory/predict-on-search", json={
            "model_id": "e2e-mem-risk",
            "filters": [
                {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
                {"field": "component_density_gb", "op": "gte", "value": 24},
                {"field": "part_status", "op": "eq", "value": "Production"},
                {"field": "operating_temperature", "op": "semantic", "value": 55},
            ],
            "limit": 500,
            "columns": ["product_id", "datasheet_id"],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] > 0

        # Step 3: Get the three lowest-risk products.
        lowest_risk = data["rows"][-3:]
        prompt = "Compare power-management for: " + ", ".join(
            f"{r['product_id']} ({r['datasheet_id']})" for r in lowest_risk
        )

        # Step 4: Mock A2A and invoke via REST.
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

    def test_full_pipeline_via_mcp(self, pipeline_services, ext_registry, monkeypatch):
        """Full pipeline via MCP: search → predict → A2A invoke."""
        import asyncio
        import httpx
        from backend.integrations.mcp.tools import call_tool

        async def run():
            # Step 1+2: Search and rank by supply risk via MCP.
            result = await call_tool("anistroph_predict_on_search", {
                "dataset_id": "semiconductor_memory",
                "model_id": "e2e-mem-risk",
                "filters": [
                    {"field": "product_family", "op": "eq", "value": "DDR5_COMPONENT"},
                    {"field": "component_density_gb", "op": "gte", "value": 24},
                    {"field": "part_status", "op": "eq", "value": "Production"},
                    {"field": "operating_temperature", "op": "semantic", "value": 55},
                ],
                "limit": 500,
                "columns": ["product_id", "datasheet_id"],
            })
            data = json.loads(result[0].text)
            assert data["matched"] > 0

            # Step 3: Get the three lowest-risk products.
            lowest_risk = data["rows"][-3:]
            prompt = "Compare power-management for: " + ", ".join(
                f"{r['product_id']} ({r['datasheet_id']})" for r in lowest_risk
            )

            # Step 4: Mock A2A and invoke via MCP.
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

            a2a_result = await call_tool("call_veris_semiconductor_research_agent", {
                "prompt": prompt,
            })
            a2a_data = json.loads(a2a_result[0].text)
            assert a2a_data["state"] == "completed"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Datasheet context construction tests
# ---------------------------------------------------------------------------

class TestDatasheetContextConstruction:
    """Verify that product/datasheet context is correctly constructed
    for the A2A handoff."""

    def test_prompt_contains_product_and_datasheet_ids(self, pipeline_services):
        """When constructing an A2A prompt from search results, the
        product_id and datasheet_id fields are present and follow the
        expected naming convention. Some products use per-product
        datasheets (DS-{product_id}), others use family-level datasheets
        (DS-{family}-FAMILY)."""
        svc = pipeline_services
        from backend.search.filters import FilterExpression
        result = svc.predict_on_search(
            search_dataset_id="semiconductor_memory",
            model_id="e2e-mem-risk",
            filters=[
                FilterExpression(field="product_family", op="eq", value="DDR5_COMPONENT"),
                FilterExpression(field="component_density_gb", op="gte", value=24),
            ],
            limit=10,
            columns=["product_id", "datasheet_id", "component_density_gb", "data_rate_mt_s"],
        )
        for row in result["rows"]:
            assert "product_id" in row
            assert "datasheet_id" in row
            # datasheet_id should start with "DS-"
            assert row["datasheet_id"].startswith("DS-")
            # datasheet_id is either per-product (DS-{product_id}) or
            # family-level (DS-{FAMILY}-FAMILY).
            dsid = row["datasheet_id"]
            assert dsid == "DS-" + row["product_id"] or dsid.endswith("-FAMILY")

    def test_family_guide_datasheets_exist(self):
        """The two family guide PDFs exist (used for family-level queries
        that Aina-Veris might reference)."""
        for filename in ["DDR5_Component_Family_Guide.pdf", "DDR5_Module_Family_Guide.pdf"]:
            path = DATASHEET_DIR / filename
            assert path.exists()
            assert path.stat().st_size > 0
