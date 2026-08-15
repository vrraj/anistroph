"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Repo root — always resolve from this file's location.
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_artifacts(tmp_path, monkeypatch):
    """Create temp artifact dirs. Does NOT change cwd (config paths are relative to repo root)."""
    (tmp_path / "artifacts" / "models").mkdir(parents=True)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "synthetic").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def small_dataset(tmp_artifacts):
    """Generate a small synthetic dataset for fast tests."""
    from scripts.generate_sensor_data import generate_dataset

    df = generate_dataset(
        n_machines=8,
        n_days=10,
        interval_minutes=30,
        seed=7,
        out_csv=str(tmp_artifacts / "data" / "synthetic" / "small.csv"),
        out_parquet=str(tmp_artifacts / "data" / "raw" / "small.parquet"),
    )
    return df


@pytest.fixture
def config():
    """Load the predictive-maintenance config using an absolute path."""
    from backend.datasets.config import load_dataset_config

    return load_dataset_config(REPO_ROOT / "datasets" / "predictive_maintenance" / "dataset.yaml")


@pytest.fixture
def config_path():
    """Absolute path to the predictive-maintenance config."""
    return str(REPO_ROOT / "datasets" / "predictive_maintenance" / "dataset.yaml")
