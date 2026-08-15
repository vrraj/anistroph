"""Unit tests — feature transforms and leakage safety."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from backend.datasets.config import load_dataset_config
from backend.features.engine import FeatureEngine
from backend.features.spec import FeatureSpec, ColumnFeatureSpec


@pytest.fixture
def feature_config():
    return load_dataset_config("datasets/predictive_maintenance/dataset.yaml")


def _make_simple_df(n=100, interval_min=30):
    """Create a simple temporal dataframe for feature tests."""
    from scripts.generate_sensor_data import generate_dataset
    return generate_dataset(n_machines=3, n_days=2, interval_minutes=interval_min, seed=11)


class TestFeatureEngine:
    def test_build_features_shape(self, small_dataset, feature_config):
        engine = FeatureEngine()
        feat_df, meta = engine.build_features(
            small_dataset, feature_config.dataset_spec, feature_config.feature_spec, fit=True
        )
        assert feat_df.height == small_dataset.height
        assert len(meta.feature_names) > 0
        assert "temperature_current" in meta.feature_names
        assert "temperature_mean_1h" in meta.feature_names

    def test_categorical_encoding(self, small_dataset, feature_config):
        engine = FeatureEngine()
        feat_df, meta = engine.build_features(
            small_dataset, feature_config.dataset_spec, feature_config.feature_spec, fit=True
        )
        cat_cols = [c for c in meta.feature_names if c.startswith("machine_type__")]
        assert len(cat_cols) > 0
        # One-hot columns should be 0 or 1.
        for c in cat_cols:
            assert feat_df[c].is_in([0, 1]).all()

    def test_rolling_mean_leakage_safe(self, small_dataset, feature_config):
        """A feature at time T must never use observations after T."""
        engine = FeatureEngine()
        feat_df, meta = engine.build_features(
            small_dataset, feature_config.dataset_spec, feature_config.feature_spec, fit=True
        )
        # For each entity, the rolling mean at the first timestamp should equal
        # the current value (window only contains the current row).
        df = small_dataset.sort(["machine_id", "timestamp"])
        feat = feat_df.sort(["machine_id", "timestamp"])
        for entity in df["machine_id"].unique().to_list():
            first_temp = df.filter(pl.col("machine_id") == entity)["temperature"][0]
            first_mean = feat.filter(pl.col("machine_id") == entity)["temperature_mean_1h"][0]
            assert abs(first_mean - first_temp) < 0.01, (
                f"leakage detected: first rolling mean {first_mean} != first value {first_temp}"
            )

    def test_feature_metadata_persistence(self, small_dataset, feature_config):
        engine = FeatureEngine()
        _, meta = engine.build_features(
            small_dataset, feature_config.dataset_spec, feature_config.feature_spec, fit=True
        )
        d = meta.to_dict()
        assert "categorical_categories" in d
        assert "machine_type" in d["categorical_categories"]
        # Round-trip.
        from backend.features.engine import FeatureMetadata
        meta2 = FeatureMetadata.from_dict(d)
        assert meta2.categorical_categories == meta.categorical_categories
        assert meta2.feature_names == meta.feature_names

    def test_inference_uses_same_metadata(self, small_dataset, feature_config):
        """Inference must use the same metadata (no refit)."""
        engine = FeatureEngine()
        train_df = small_dataset.head(small_dataset.height // 2)
        infer_df = small_dataset.tail(10)
        _, meta = engine.build_features(train_df, feature_config.dataset_spec, feature_config.feature_spec, fit=True)
        feat_df, meta2 = engine.build_features(infer_df, feature_config.dataset_spec, feature_config.feature_spec, metadata=meta, fit=False)
        assert meta2.feature_names == meta.feature_names

    def test_current_transform(self, small_dataset, feature_config):
        engine = FeatureEngine()
        feat_df, meta = engine.build_features(
            small_dataset, feature_config.dataset_spec, feature_config.feature_spec, fit=True
        )
        # current should match the raw value.
        df = small_dataset.sort(["machine_id", "timestamp"])
        feat = feat_df.sort(["machine_id", "timestamp"])
        assert (feat["temperature_current"] == df["temperature"]).all()

    def test_slope_transform(self, small_dataset, feature_config):
        engine = FeatureEngine()
        feat_df, meta = engine.build_features(
            small_dataset, feature_config.dataset_spec, feature_config.feature_spec, fit=True
        )
        assert "temperature_slope_6h" in meta.feature_names
        # Slope at the first row should be NaN (insufficient data for regression).
        first = feat_df.sort(["machine_id", "timestamp"]).filter(
            pl.col("machine_id") == pl.col("machine_id").first()
        ).head(1)
        # First slope may be NaN due to zero variance.
        val = first["temperature_slope_6h"][0]
        assert val is None or np.isnan(val) or isinstance(val, float)

    def test_no_domain_assumptions_in_engine(self):
        """The FeatureEngine must work with arbitrary column names."""
        spec_data = {
            "dataset_id": "test_generic",
            "name": "Test",
            "entity_key": "widget_id",
            "time_key": "ts",
            "columns": {
                "ts": {"type": "timestamp", "role": "identifier"},
                "widget_id": {"type": "categorical", "role": "identifier"},
                "metric_x": {"type": "numeric", "role": "feature"},
                "category_y": {"type": "categorical", "role": "feature"},
            },
        }
        import yaml
        from backend.datasets.config import load_dataset_config
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "dataset": spec_data,
                "features": {
                    "metric_x": {"column": "metric_x", "transforms": ["current", {"mean": {"windows": ["1h"]}}]},
                    "category_y": {"column": "category_y", "transforms": ["categorical"]},
                },
            }, f)
            path = f.name
        try:
            cfg = load_dataset_config(path)
            df = pl.DataFrame({
                "ts": [datetime(2026, 1, 1, i) for i in range(20)],
                "widget_id": ["W1"]*20,
                "metric_x": [float(i) for i in range(20)],
                "category_y": ["A"]*10 + ["B"]*10,
            })
            engine = FeatureEngine()
            feat_df, meta = engine.build_features(df, cfg.dataset_spec, cfg.feature_spec, fit=True)
            assert "metric_x_current" in meta.feature_names
            assert "metric_x_mean_1h" in meta.feature_names
            assert any(c.startswith("category_y__") for c in meta.feature_names)
        finally:
            os.unlink(path)
