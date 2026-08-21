"""Tests for the semiconductor yield dataset and models."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from backend.datasets.config import load_dataset_config
from backend.services import AnistrophServices


@pytest.fixture
def semiconductor_config():
    return load_dataset_config("datasets/semiconductor_yield/dataset.yaml")


@pytest.fixture
def semi_registry(tmp_artifacts, semiconductor_config):
    """Register semiconductor_yield dataset with generated data."""
    from scripts.generate_semiconductor_yield_data import generate_wafers
    df = generate_wafers(n_wafers=500, seed=42)

    pq = tmp_artifacts / "data" / "semiconductor_yield" / "data.parquet"
    pq.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(pq))

    svc = AnistrophServices(
        dataset_registry_path=tmp_artifacts / "artifacts" / "dataset_registry.json",
        model_registry_dir=tmp_artifacts / "artifacts" / "models",
    )
    svc.register_dataset_from_config(
        "datasets/semiconductor_yield/dataset.yaml",
        str(pq),
    )
    svc.train("semiconductor_yield", "wafer_yield", "xgboost_regressor", model_id="test-semi-xgb")
    return svc


class TestSemiconductorData:
    def test_data_file_exists(self):
        assert Path("data/semiconductor_yield/data.parquet").exists()

    def test_config(self, semiconductor_config):
        spec = semiconductor_config.dataset_spec
        assert spec.entity_key == "wafer_id"


class TestSemiconductorTraining:
    def test_train_and_predict(self, semi_registry):
        pred = semi_registry.predict("test-semi-xgb", entity_id="WAFER_000001")
        assert "predicted_yield" in pred


class TestFindInterestingSlices:
    def test_find_interesting_slices(self, semi_registry):
        result = semi_registry.find_interesting_slices("semiconductor_yield", "wafer_yield", top_k=5)
        assert len(result) > 0
        assert "dimensions" in result[0]


class TestSHAPExplainability:
    def test_explain(self, semi_registry):
        expl = semi_registry.explain("test-semi-xgb", entity_id="WAFER_000001", top_k=5)
        assert "top_drivers" in expl
        assert len(expl["top_drivers"]) <= 5
