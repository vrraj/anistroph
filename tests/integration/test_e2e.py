"""End-to-end acceptance test — the complete reference workflow."""

from __future__ import annotations

import json

import pytest

from scripts.generate_sensor_data import generate_dataset
from backend.services import AnistrophServices
import backend.services as svc_mod
from fastapi.testclient import TestClient
from backend.main import create_app


@pytest.fixture
def e2e_env(tmp_path):
    for d in ["artifacts/models", "data/raw", "data/processed", "data/synthetic"]:
        (tmp_path / d).mkdir(parents=True)
    csv_path = tmp_path / "data" / "synthetic" / "predictive_maintenance.csv"
    pq_path = tmp_path / "data" / "processed" / "predictive_maintenance.parquet"
    generate_dataset(n_machines=12, n_days=20, interval_minutes=15, seed=99, out_csv=str(csv_path))
    services = AnistrophServices(
        dataset_registry_path=tmp_path / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_path / "artifacts" / "models",
    )
    svc_mod._services = services
    yield services, csv_path, pq_path
    svc_mod._services = None


class TestEndToEnd:
    def test_full_workflow(self, e2e_env):
        """generate -> register -> profile -> train -> predict -> explain -> REST/MCP parity."""
        services, csv_path, pq_path = e2e_env

        services.register_dataset_from_config(
            "datasets/predictive_maintenance/dataset.yaml", str(csv_path), parquet_path=str(pq_path),
        )
        assert services.profile("predictive_maintenance")["entity_count"] == 12

        services.train("predictive_maintenance", "failure_within_horizon", "xgboost", model_id="e2e-xgb")
        pred = services.predict("e2e-xgb", entity_id="TOOL_000", timestamp="2026-06-15T12:00:00")
        assert 0.0 <= pred["probability"] <= 1.0

        expl = services.explain("e2e-xgb", entity_id="TOOL_000", timestamp="2026-06-15T12:00:00", top_k=5)
        assert len(expl["top_drivers"]) > 0

        # REST and MCP return the same prediction.
        client = TestClient(create_app())
        rest_prob = client.post("/predictions", json={
            "model_id": "e2e-xgb", "entity_id": "TOOL_000", "timestamp": "2026-06-15T12:00:00",
        }).json()["probability"]

        from backend.integrations.mcp.tools import call_tool
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            call_tool("anistroph_predict", {
                "model_id": "e2e-xgb", "entity_id": "TOOL_000", "timestamp": "2026-06-15T12:00:00",
            })
        )
        assert abs(rest_prob - json.loads(result[0].text)["probability"]) < 1e-6
