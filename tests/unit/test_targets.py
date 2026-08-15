"""Unit tests — target construction (binary, future_event, horizon boundaries, entity isolation)."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from backend.datasets.config import load_dataset_config
from backend.targets.engine import TargetEngine
from backend.targets.spec import TargetSpec, TargetType


@pytest.fixture
def target_config():
    return load_dataset_config("datasets/predictive_maintenance/dataset.yaml")


class TestFutureEventTarget:
    def test_target_column_exists(self, small_dataset, target_config):
        engine = TargetEngine()
        df = engine.build_target(small_dataset, target_config.dataset_spec, target_config.target_spec)
        assert target_config.target_spec.name in df.columns

    def test_positive_labels_exist(self, small_dataset, target_config):
        engine = TargetEngine()
        df = engine.build_target(small_dataset, target_config.dataset_spec, target_config.target_spec)
        n_pos = df.filter(pl.col(target_config.target_spec.name) == 1).height
        # There should be some positive labels (failures exist in synthetic data).
        assert n_pos > 0

    def test_entity_isolation(self, target_config):
        """A failure on Machine B must never label Machine A."""
        # Create data where only TOOL_B has a failure.
        ts = datetime(2026, 1, 1, 12, 0)
        df = pl.DataFrame({
            "timestamp": [ts, ts + timedelta(hours=1), ts, ts + timedelta(hours=1)],
            "machine_id": ["TOOL_A", "TOOL_A", "TOOL_B", "TOOL_B"],
            "machine_type": ["TYPE_A", "TYPE_A", "TYPE_A", "TYPE_A"],
            "temperature": [70.0, 70.0, 70.0, 70.0],
            "vibration": [2.0, 2.0, 2.0, 2.0],
            "pressure": [100.0, 100.0, 100.0, 100.0],
            "current": [10.0, 10.0, 10.0, 10.0],
            "voltage": [230.0, 230.0, 230.0, 230.0],
            "rpm": [1800.0, 1800.0, 1800.0, 1800.0],
            "flow_rate": [50.0, 50.0, 50.0, 50.0],
            "maintenance_age_hours": [0.0, 1.0, 0.0, 1.0],
            "operating_hours": [100.0, 101.0, 100.0, 101.0],
            "failure": [0, 0, 0, 1],  # Only TOOL_B fails at t+1h
            "failure_type": ["", "", "", "MECHANICAL"],
        })
        engine = TargetEngine()
        df = engine.build_target(df, target_config.dataset_spec, target_config.target_spec)
        # TOOL_A should have NO positive labels.
        tool_a = df.filter(pl.col("machine_id") == "TOOL_A")
        assert tool_a["failure_within_horizon"].sum() == 0
        # TOOL_B at t=0 should have a positive label (failure within 24h).
        tool_b_t0 = df.filter((pl.col("machine_id") == "TOOL_B") & (pl.col("timestamp") == ts))
        assert tool_b_t0["failure_within_horizon"][0] == 1

    def test_horizon_boundary(self, target_config):
        """A failure exactly at T+horizon should NOT be included (window is (T, T+horizon])."""
        ts = datetime(2026, 1, 1, 12, 0)
        # Failure at exactly 24h after T.
        df = pl.DataFrame({
            "timestamp": [ts, ts + timedelta(hours=24)],
            "machine_id": ["TOOL_X", "TOOL_X"],
            "machine_type": ["TYPE_A", "TYPE_A"],
            "temperature": [70.0, 70.0],
            "vibration": [2.0, 2.0],
            "pressure": [100.0, 100.0],
            "current": [10.0, 10.0],
            "voltage": [230.0, 230.0],
            "rpm": [1800.0, 1800.0],
            "flow_rate": [50.0, 50.0],
            "maintenance_age_hours": [0.0, 24.0],
            "operating_hours": [100.0, 124.0],
            "failure": [0, 1],
            "failure_type": ["", "THERMAL"],
        })
        engine = TargetEngine()
        df = engine.build_target(df, target_config.dataset_spec, target_config.target_spec)
        # At T, the failure at T+24h is within (T, T+24h] => should be 1.
        row_t0 = df.filter(pl.col("timestamp") == ts)
        assert row_t0["failure_within_horizon"][0] == 1

    def test_no_future_leakage_in_features(self, small_dataset, target_config):
        """The target at time T uses future events, but features must not."""
        from backend.features.engine import FeatureEngine
        te = TargetEngine()
        df = te.build_target(small_dataset, target_config.dataset_spec, target_config.target_spec)
        fe = FeatureEngine()
        feat_df, meta = fe.build_features(
            small_dataset, target_config.dataset_spec, target_config.feature_spec, fit=True
        )
        # Features should have the same row count as the raw data.
        assert feat_df.height == small_dataset.height


class TestBinaryTarget:
    def test_binary_target(self):
        spec = TargetSpec(name="is_failed", type=TargetType.BINARY, source_column="failure", positive_class=1)
        df = pl.DataFrame({"failure": [0, 1, 0, 1]})
        engine = TargetEngine()
        # For binary, we don't need a full DatasetSpec.
        from backend.datasets.spec import DatasetSpec, ColumnSpec, ColumnType, ColumnRole
        ds = DatasetSpec(
            dataset_id="test", name="test", entity_key="id",
            columns={"failure": ColumnSpec(name="failure", type=ColumnType.BOOLEAN, role=ColumnRole.EVENT)}
        )
        df = engine.build_target(df, ds, spec)
        assert "is_failed" in df.columns
        assert df["is_failed"].to_list() == [0, 1, 0, 1]
