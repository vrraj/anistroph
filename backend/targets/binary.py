"""Binary target construction.

Directly maps a source boolean/integer column to a binary target.
"""

from __future__ import annotations

import polars as pl

from backend.targets.spec import TargetSpec


def build_binary_target(df: pl.DataFrame, spec: TargetSpec) -> pl.DataFrame:
    """Build a binary target column from the source column."""
    out_name = spec.name
    positive = spec.positive_class
    return df.with_columns(
        (pl.col(spec.source_column) == positive).cast(pl.Int8).alias(out_name)
    )
