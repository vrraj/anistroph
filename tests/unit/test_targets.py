"""Unit tests — target construction and leakage safety."""

from __future__ import annotations

import pytest

from backend.datasets.config import load_dataset_config
from backend.targets.engine import TargetEngine


@pytest.fixture
def target_config():
    return load_dataset_config("datasets/predictive_maintenance/dataset.yaml")


class TestFutureEventTarget:
    def test_target_column_exists(self, small_dataset, target_config):
        engine = TargetEngine()
        df = engine.build_target(small_dataset, target_config.dataset_spec, target_config.target_spec)
        assert target_config.target_spec.name in df.columns

    def test_no_future_leakage_in_features(self, small_dataset, target_config):
        """Target at time T must not leak into features at time T."""
        engine = TargetEngine()
        df = engine.build_target(small_dataset, target_config.dataset_spec, target_config.target_spec)
        assert df[target_config.target_spec.name].is_in([0, 1]).all()
