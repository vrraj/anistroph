"""Regression target construction.

Architectural placeholder — maps a numeric source column to a regression
target. Full regression model implementation is optional for v0.1.
"""

from __future__ import annotations

import polars as pl

from backend.targets.spec import TargetSpec


def build_regression_target(df: pl.DataFrame, spec: TargetSpec) -> pl.DataFrame:
    """Build a regression target column from the source column."""
    out_name = spec.name
    return df.with_columns(
        pl.col(spec.source_column).cast(pl.Float64).alias(out_name)
    )
