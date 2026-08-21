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
        assert spec.is_temporal()

    def test_column_types_and_roles(self, config):
        spec = config.dataset_spec
        assert spec.column("temperature").type == ColumnType.NUMERIC
        assert spec.column("failure").type == ColumnType.BOOLEAN
        assert "temperature" in spec.feature_columns
        assert "failure" not in spec.feature_columns


class TestValidation:
    def test_valid_dataset(self, small_dataset, config):
        report = validate_dataset(small_dataset, config.dataset_spec)
        assert report.valid, f"errors: {report.errors}"

    def test_missing_column(self, config):
        df = pl.DataFrame({"machine_id": ["A"], "timestamp": ["2026-01-01"]})
        report = validate_dataset(df, config.dataset_spec)
        assert not report.valid


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


class TestProfiling:
    def test_profile(self, small_dataset, config):
        prof = profile_dataset(small_dataset, config.dataset_spec)
        assert prof["row_count"] == small_dataset.height
        assert prof["entity_count"] == 8
        assert "time_range" in prof


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
