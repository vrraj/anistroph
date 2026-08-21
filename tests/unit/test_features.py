"""Unit tests — feature transforms and leakage safety."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from backend.datasets.config import load_dataset_config
from backend.features.engine import FeatureEngine
from backend.features.spec import FeatureSpec, ColumnFeatureSpec


@pytest.fixture
def feature_config():
    return load_dataset_config("datasets/predictive_maintenance/dataset.yaml")


class TestFeatureEngine:
    def test_build_features_shape(self, small_dataset, feature_config):
        engine = FeatureEngine()
        feat_df, meta = engine.build_features(
            small_dataset, feature_config.dataset_spec, feature_config.feature_spec, fit=True
        )
        assert feat_df.height == small_dataset.height
        assert "temperature_current" in meta.feature_names
        assert "temperature_mean_1h" in meta.feature_names

    def test_rolling_mean_leakage_safe(self, small_dataset, feature_config):
        """A feature at time T must never use observations after T."""
        engine = FeatureEngine()
        feat_df, meta = engine.build_features(
            small_dataset, feature_config.dataset_spec, feature_config.feature_spec, fit=True
        )
        df = small_dataset.sort(["machine_id", "timestamp"])
        feat = feat_df.sort(["machine_id", "timestamp"])
        for entity in df["machine_id"].unique().to_list():
            first_temp = df.filter(pl.col("machine_id") == entity)["temperature"][0]
            first_mean = feat.filter(pl.col("machine_id") == entity)["temperature_mean_1h"][0]
            assert abs(first_mean - first_temp) < 0.01

    def test_inference_uses_same_metadata(self, small_dataset, feature_config):
        """Inference must use the same metadata (no refit)."""
        engine = FeatureEngine()
        train_df = small_dataset.head(small_dataset.height // 2)
        infer_df = small_dataset.tail(10)
        _, meta = engine.build_features(train_df, feature_config.dataset_spec, feature_config.feature_spec, fit=True)
        feat_df, meta2 = engine.build_features(infer_df, feature_config.dataset_spec, feature_config.feature_spec, metadata=meta, fit=False)
        assert meta2.feature_names == meta.feature_names
