"""Temporal feature transforms — leakage-safe calendar/elapsed features.

These derive from the time_key and do not look into the future.
"""

from __future__ import annotations

import polars as pl


def hour_of_day(df: pl.DataFrame, time_col: str, out_col: str) -> pl.DataFrame:
    return df.with_columns(pl.col(time_col).dt.hour().cast(pl.Float64).alias(out_col))


def day_of_week(df: pl.DataFrame, time_col: str, out_col: str) -> pl.DataFrame:
    return df.with_columns(pl.col(time_col).dt.weekday().cast(pl.Float64).alias(out_col))


def elapsed_time(
    df: pl.DataFrame,
    entity_col: str,
    time_col: str,
    out_col: str,
    unit: str = "s",
) -> pl.DataFrame:
    """Seconds (or other unit) elapsed since the first observation per entity."""
    df = df.sort([entity_col, time_col])
    expr = pl.col(time_col).cum_count().over(entity_col)  # placeholder for grouping
    # Use diff from min per entity.
    df = df.with_columns(
        (pl.col(time_col) - pl.col(time_col).min().over(entity_col))
        .dt.total_seconds()
        .cast(pl.Float64)
        .alias(out_col)
    )
    return df
