"""Application configuration — environment-based settings.

Loads global defaults from a ``.env`` file at the repo root using
pydantic-settings. These provide fallback values for dataset partitioning
percentages; per-dataset YAML ``split:`` sections override these defaults.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Global application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Dataset partitioning defaults (fractions of the full dataset).
    # YAML split sections override these per-dataset.
    TRAIN_DATASET_PCT: float = 0.80
    EVAL_DATASET_PCT: float = 0.20
    VALIDATE_DATASET_PCT: float = 0.0


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
