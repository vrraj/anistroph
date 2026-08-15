"""Generic dataset profiling — operates from DatasetSpec, not domain assumptions.

Returns: row count, column count, column types, missing values, unique counts,
numeric distributions, categorical distributions, time range, entity count,
target/event distribution.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from backend.datasets.spec import ColumnType, ColumnRole, DatasetSpec


def profile_dataset(df: pl.DataFrame, spec: DatasetSpec) -> dict[str, Any]:
    """Profile a dataset generically using the DatasetSpec."""
    profile: dict[str, Any] = {
        "dataset_id": spec.dataset_id,
        "row_count": df.height,
        "column_count": df.width,
        "columns": {},
    }

    for col_name, col_spec in spec.columns.items():
        if col_name not in df.columns:
            continue
        col = df[col_name]
        col_profile: dict[str, Any] = {
            "name": col_name,
            "type": col_spec.type.value,
            "role": col_spec.role.value,
            "null_count": col.null_count(),
            "unique_count": col.n_unique(),
        }

        if col_spec.type == ColumnType.NUMERIC:
            col_profile.update(_numeric_stats(col))
        elif col_spec.type == ColumnType.CATEGORICAL:
            col_profile.update(_categorical_stats(col))
        elif col_spec.type == ColumnType.BOOLEAN:
            col_profile.update(_boolean_stats(col))

        profile["columns"][col_name] = col_profile

    # Time range.
    if spec.time_key and spec.time_key in df.columns:
        tk = spec.time_key
        profile["time_range"] = {
            "start": str(df[tk].min()),
            "end": str(df[tk].max()),
        }

    # Entity count.
    if spec.entity_key in df.columns:
        profile["entity_count"] = df[spec.entity_key].n_unique()

    # Event distribution.
    event_cols = spec.event_columns
    if event_cols:
        profile["event_distribution"] = {}
        for ec in event_cols:
            if ec in df.columns:
                profile["event_distribution"][ec] = _value_counts(df[ec])

    return profile


def _numeric_stats(col: pl.Series) -> dict[str, Any]:
    return {
        "min": _safe_float(col.min()),
        "max": _safe_float(col.max()),
        "mean": _safe_float(col.mean()),
        "std": _safe_float(col.std()),
        "median": _safe_float(col.median()),
        "p25": _safe_float(col.quantile(0.25)),
        "p75": _safe_float(col.quantile(0.75)),
    }


def _categorical_stats(col: pl.Series) -> dict[str, Any]:
    vc = col.value_counts().sort("count", descending=True)
    top = vc.head(10).to_dicts()
    return {
        "top_values": [{"value": str(r[col.name]), "count": r["count"]} for r in top],
    }


def _boolean_stats(col: pl.Series) -> dict[str, Any]:
    return {"distribution": _value_counts(col)}


def _value_counts(col: pl.Series) -> dict[str, int]:
    vc = col.value_counts()
    return {str(r[col.name]): int(r["count"]) for r in vc.to_dicts()}


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
