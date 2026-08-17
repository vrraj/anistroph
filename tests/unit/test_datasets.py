"""Unit tests — DatasetSpec parsing, validation, ingestion, profiling."""

from __future__ import annotations

import pytest
import polars as pl

from backend.datasets.config import load_dataset_config
from backend.datasets.loader import ingest
from backend.datasets.profiling import profile_dataset
from backend.datasets.registry import DatasetRegistry
from backend.datasets.spec import ColumnType, ColumnRole, DatasetSpec
from backend.datasets.validation import validate_dataset


class TestDatasetSpec:
    def test_load_config(self, config):
        spec = config.dataset_spec
        assert spec.dataset_id == "predictive_maintenance"
        assert spec.entity_key == "machine_id"
        assert spec.time_key == "timestamp"
        assert spec.is_temporal()

    def test_feature_columns(self, config):
        cols = config.dataset_spec.feature_columns
        assert "temperature" in cols
        assert "vibration" in cols
        assert "failure" not in cols  # event, not feature

    def test_event_columns(self, config):
        assert config.dataset_spec.event_columns == ["failure"]

    def test_column_types(self, config):
        spec = config.dataset_spec
        assert spec.column("temperature").type == ColumnType.NUMERIC
        assert spec.column("machine_id").type == ColumnType.CATEGORICAL
        assert spec.column("failure").type == ColumnType.BOOLEAN
        assert spec.column("timestamp").type == ColumnType.TIMESTAMP

    def test_split_config(self, config):
        split = config.dataset_spec.split
        assert split.strategy == "chronological"
        assert split.train == 0.70
        assert split.validation == 0.15
        assert split.test == 0.15


class TestValidation:
    def test_valid_dataset(self, small_dataset, config):
        report = validate_dataset(small_dataset, config.dataset_spec)
        assert report.valid, f"errors: {report.errors}"

    def test_missing_column(self, config):
        df = pl.DataFrame({"machine_id": ["A"], "timestamp": ["2026-01-01"]})
        report = validate_dataset(df, config.dataset_spec)
        assert not report.valid
        assert any("missing" in e for e in report.errors)

    def test_null_entity_key(self, config):
        df = pl.DataFrame({
            "timestamp": ["2026-01-01", None],
            "machine_id": ["A", None],
            "machine_type": ["X", "X"],
            "temperature": [70.0, 71.0],
            "vibration": [2.0, 2.1],
            "pressure": [100.0, 101.0],
            "current": [10.0, 10.1],
            "voltage": [230.0, 230.1],
            "rpm": [1800.0, 1790.0],
            "flow_rate": [50.0, 50.1],
            "maintenance_age_hours": [0.0, 0.5],
            "operating_hours": [100.0, 100.5],
            "failure": [0, 0],
            "failure_mode": ["NONE", "NONE"],
            "remaining_useful_life_hours": [100.0, 99.0],
            "maintenance_required": [0, 0],
        })
        report = validate_dataset(df, config.dataset_spec)
        assert not report.valid
        assert any("null" in e for e in report.errors)


class TestIngestion:
    def test_ingest_csv(self, tmp_artifacts, small_dataset, config):
        pq = tmp_artifacts / "data" / "processed" / "test.parquet"
        df, report, path = ingest(
            tmp_artifacts / "data" / "synthetic" / "small.csv",
            config.dataset_spec,
            pq,
        )
        assert report.valid
        assert path.exists()
        loaded = pl.read_parquet(pq)
        assert loaded.height == small_dataset.height

    def test_ingest_parquet(self, tmp_artifacts, small_dataset, config):
        pq = tmp_artifacts / "data" / "processed" / "test_pq.parquet"
        df, report, path = ingest(
            tmp_artifacts / "data" / "raw" / "small.parquet",
            config.dataset_spec,
            pq,
        )
        assert report.valid
        assert path.exists()


class TestProfiling:
    def test_profile(self, small_dataset, config):
        prof = profile_dataset(small_dataset, config.dataset_spec)
        assert prof["row_count"] == small_dataset.height
        assert prof["column_count"] == len(config.dataset_spec.columns)
        assert prof["entity_count"] == 8
        assert "time_range" in prof
        assert "event_distribution" in prof
        assert "failure" in prof["event_distribution"]

    def test_profile_numeric_stats(self, small_dataset, config):
        prof = profile_dataset(small_dataset, config.dataset_spec)
        temp = prof["columns"]["temperature"]
        assert "min" in temp
        assert "max" in temp
        assert "mean" in temp
        assert "std" in temp


class TestRegistry:
    def test_register_and_retrieve(self, tmp_artifacts, small_dataset, config):
        reg = DatasetRegistry(tmp_artifacts / "test_registry.json")
        meta = reg.register(
            spec=config.dataset_spec,
            source="test.csv",
            row_count=small_dataset.height,
            parquet_path="test.parquet",
            feature_spec=config.feature_spec,
            target_spec=config.target_spec,
        )
        assert meta.dataset_id == "predictive_maintenance"
        assert reg.exists("predictive_maintenance")
        retrieved = reg.get("predictive_maintenance")
        assert retrieved.row_count == small_dataset.height

    def test_list(self, tmp_artifacts, config):
        reg = DatasetRegistry(tmp_artifacts / "test_registry.json")
        assert reg.list() == []
