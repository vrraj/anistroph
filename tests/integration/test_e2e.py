"""End-to-end acceptance test — the complete reference workflow (spec §30).

generate synthetic → register DatasetSpec → ingest CSV → persist Parquet →
profile → build features → construct labels → chronological split →
train LR + XGB → evaluate → persist → reload → predict → explain →
REST result → repeat through MCP.
"""

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
    """Full end-to-end environment with isolated artifacts."""
    for d in ["artifacts/models", "data/raw", "data/processed", "data/synthetic"]:
        (tmp_path / d).mkdir(parents=True)

    # Generate synthetic data.
    csv_path = tmp_path / "data" / "synthetic" / "predictive_maintenance.csv"
    pq_path = tmp_path / "data" / "processed" / "predictive_maintenance.parquet"
    df = generate_dataset(
        n_machines=12, n_days=20, interval_minutes=15, seed=99,
        out_csv=str(csv_path),
    )
    assert df.height > 0

    services = AnistrophServices(
        dataset_registry_path=tmp_path / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_path / "artifacts" / "models",
    )
    svc_mod._services = services
    yield services, csv_path, pq_path
    svc_mod._services = None


class TestEndToEnd:
    def test_full_workflow(self, e2e_env):
        services, csv_path, pq_path = e2e_env

        # 1. Register DatasetSpec + ingest CSV → Parquet.
        meta = services.register_dataset_from_config(
            "datasets/predictive_maintenance/dataset.yaml",
            str(csv_path),
            parquet_path=str(pq_path),
        )
        assert meta.dataset_id == "predictive_maintenance"
        assert meta.row_count > 0

        # 2. Profile.
        prof = services.profile("predictive_maintenance")
        assert prof["row_count"] > 0
        assert prof["entity_count"] == 12

        # 3. Train Logistic Regression.
        lr_result = services.train(
            "predictive_maintenance", "failure_within_horizon", "logistic_regression",
            model_id="e2e-lr",
        )
        assert lr_result["metrics"]["roc_auc"] is not None
        assert lr_result["metrics"]["roc_auc"] > 0.5

        # 4. Train XGBoost.
        xgb_result = services.train(
            "predictive_maintenance", "failure_within_horizon", "xgboost",
            model_id="e2e-xgb",
        )
        assert xgb_result["metrics"]["roc_auc"] is not None
        assert xgb_result["metrics"]["roc_auc"] > 0.5

        # 5. Models perform meaningfully above random.
        assert xgb_result["metrics"]["roc_auc"] > 0.6

        # 6. Persist + reload (verify model is in registry).
        meta = services.get_model("e2e-xgb")
        assert meta is not None
        from pathlib import Path
        assert (Path(meta.artifact_path) / "model.joblib").exists()

        # 7. Predict.
        pred = services.predict("e2e-xgb", entity_id="TOOL_000", timestamp="2026-06-15T12:00:00")
        assert "probability" in pred
        assert 0.0 <= pred["probability"] <= 1.0

        # 8. Explain.
        expl = services.explain("e2e-xgb", entity_id="TOOL_000", timestamp="2026-06-15T12:00:00", top_k=5)
        assert "top_drivers" in expl
        assert len(expl["top_drivers"]) > 0

        # 9. Slice (analytical, independent of ML).
        sl = services.slice("predictive_maintenance", ["machine_type"], "failure", "mean")
        assert len(sl) > 0

    def test_rest_and_mcp_same_services(self, e2e_env):
        """REST and MCP invoke the same core services."""
        services, csv_path, pq_path = e2e_env

        services.register_dataset_from_config(
            "datasets/predictive_maintenance/dataset.yaml",
            str(csv_path),
            parquet_path=str(pq_path),
        )
        services.train("predictive_maintenance", "failure_within_horizon", "xgboost", model_id="e2e-rest-mcp")

        # REST prediction.
        app = create_app()
        client = TestClient(app)
        r = client.post("/predictions", json={
            "model_id": "e2e-rest-mcp",
            "entity_id": "TOOL_000",
            "timestamp": "2026-06-15T12:00:00",
        })
        assert r.status_code == 200
        rest_prob = r.json()["probability"]

        # MCP prediction (same model, same entity, same timestamp).
        from backend.integrations.mcp.tools import call_tool
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            call_tool("anistroph_predict", {
                "model_id": "e2e-rest-mcp",
                "entity_id": "TOOL_000",
                "timestamp": "2026-06-15T12:00:00",
            })
        )
        mcp_data = json.loads(result[0].text)
        mcp_prob = mcp_data["probability"]

        # Both should return the same probability (same model, same features).
        assert abs(rest_prob - mcp_prob) < 1e-6
