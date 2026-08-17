"""Dataset partitioning — split a dataset into train / evaluation / validate
partitions and persist each as a separate Parquet file.

Partitioning happens at registration time so the held-out evaluation set is
fixed on disk and can never accidentally leak into model fitting. Training
loads only ``train.parquet``; runtime / MCP evaluation loads
``evaluation.parquet`` and compares predictions against known actuals.

Sorting:
    Temporal datasets (time_key set) are sorted chronologically — oldest rows
    go to train, newest to evaluation. Non-temporal datasets are shuffled with
    a fixed seed before splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import polars as pl

from backend.datasets.spec import DatasetSpec


# Partition names in deterministic order.
TRAIN = "train"
EVAL = "evaluation"
VALIDATE = "validate"
PARTITION_NAMES = (TRAIN, EVAL, VALIDATE)


def resolve_split_percentages(
    spec: DatasetSpec,
    env_train: float,
    env_eval: float,
    env_validate: float,
) -> tuple[float, float, float]:
    """Resolve split percentages with YAML-overrides-.env precedence.

    The YAML ``split:`` section (on DatasetSpec) takes precedence when its
    values differ from the all-zero / all-default sentinel. Otherwise the
    ``.env`` defaults are used.

    Returns a normalised (train, eval, validate) tuple that sums to 1.0.
    """
    yaml_split = spec.split

    # Use YAML values if the split section was explicitly customised
    # (train < 1.0 indicates the author intended a real split).
    if yaml_split.train < 1.0:
        train = yaml_split.train
        # YAML uses train/validation/test; map test → eval for partitioning.
        eval_pct = yaml_split.test
        validate = yaml_split.validation
    else:
        train = env_train
        eval_pct = env_eval
        validate = env_validate

    total = train + eval_pct + validate
    if total <= 0:
        raise ValueError(f"split percentages must be positive, got sum={total}")

    # Normalise to sum to 1.0.
    train, eval_pct, validate = train / total, eval_pct / total, validate / total
    return train, eval_pct, validate


def partition_dataframe(
    df: pl.DataFrame,
    spec: DatasetSpec,
    train_pct: float,
    eval_pct: float,
    validate_pct: float = 0.0,
    seed: int = 42,
) -> dict[str, pl.DataFrame]:
    """Split a DataFrame into train / evaluation / validate partitions.

    For temporal datasets, sorts by ``time_key`` and slices chronologically
    (oldest → train, newest → evaluation). For non-temporal datasets, shuffles
    with a fixed seed before slicing.

    Args:
        df: full dataset DataFrame.
        spec: DatasetSpec (used for time_key / is_temporal).
        train_pct: fraction for training (normalised internally).
        eval_pct: fraction for evaluation (normalised internally).
        validate_pct: fraction for validation (default 0).
        seed: random seed for non-temporal shuffling.

    Returns a dict with keys ``train``, ``evaluation``, and ``validate``.
    Partitions with 0 fraction are returned as empty DataFrames (same schema).
    """
    total = train_pct + eval_pct + validate_pct
    if total <= 0:
        raise ValueError(f"split percentages must be positive, got sum={total}")
    train_pct, eval_pct, validate_pct = (
        train_pct / total,
        eval_pct / total,
        validate_pct / total,
    )

    if spec.is_temporal() and spec.time_key in df.columns:
        df = df.sort(spec.time_key)
    else:
        df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    n = df.height
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + validate_pct))

    train_df = df[:train_end]
    validate_df = df[train_end:val_end]
    eval_df = df[val_end:]

    return {
        TRAIN: train_df,
        EVAL: eval_df,
        VALIDATE: validate_df,
    }


def persist_partitions(
    partitions: dict[str, pl.DataFrame],
    base_dir: str | Path,
    dataset_id: str,
) -> dict[str, Optional[str]]:
    """Persist partition DataFrames as separate Parquet files.

    Files are written as ``{base_dir}/{dataset_id}.train.parquet``,
    ``{base_dir}/{dataset_id}.evaluation.parquet``, and
    ``{base_dir}/{dataset_id}.validate.parquet``.

    Empty partitions (0 rows) are skipped — their path is ``None``.

    Returns a dict mapping partition name → absolute parquet path (or None).
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Optional[str]] = {}
    for name in PARTITION_NAMES:
        df = partitions.get(name)
        if df is None or df.height == 0:
            paths[name] = None
            continue
        path = base_dir / f"{dataset_id}.{name}.parquet"
        df.write_parquet(str(path))
        paths[name] = str(path)

    return paths


def partition_summary(partitions: dict[str, pl.DataFrame]) -> dict[str, Any]:
    """Return a summary dict of partition row counts and percentages."""
    total = sum(df.height for df in partitions.values())
    summary: dict[str, Any] = {}
    for name in PARTITION_NAMES:
        df = partitions.get(name)
        count = df.height if df is not None else 0
        summary[name] = {
            "row_count": count,
            "pct": round(count / total, 4) if total > 0 else 0.0,
        }
    summary["total"] = total
    return summary
