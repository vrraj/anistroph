"""Categorical feature transforms — one-hot encoding with unknown-category handling.

The engine learns categories from training data and stores them in the
feature metadata so inference applies the same encoding. Unknown categories
at inference map to an "unknown" bucket.
"""

from __future__ import annotations

from typing import Any

import polars as pl

UNKNOWN_TOKEN = "__unknown__"


def onehot_columns(source_col: str, categories: list[str]) -> list[str]:
    """Return the output column names for one-hot encoding of ``source_col``."""
    return [f"{source_col}__{c}" for c in categories]


def fit_onehot(df: pl.DataFrame, col: str, min_frequency: int = 1) -> list[str]:
    """Learn the category set from a training DataFrame."""
    vc = df.group_by(col).len().sort("len", descending=True)
    if min_frequency > 1:
        vc = vc.filter(pl.col("len") >= min_frequency)
    return [str(r[col]) for r in vc.to_dicts()]


def transform_onehot(
    df: pl.DataFrame,
    col: str,
    categories: list[str],
) -> pl.DataFrame:
    """Apply one-hot encoding using learned categories.

    Unknown categories become all-zero (and an optional ``__unknown__`` column).
    """
    out_cols = onehot_columns(col, categories)
    # Normalize value to known category or UNKNOWN_TOKEN.
    norm_col = f"{col}__norm"
    df = df.with_columns(
        pl.when(pl.col(col).cast(pl.Utf8).is_in(categories))
        .then(pl.col(col).cast(pl.Utf8))
        .otherwise(pl.lit(UNKNOWN_TOKEN))
        .alias(norm_col)
    )
    # Build one indicator column per known category.
    exprs = []
    for cat in categories:
        exprs.append((pl.col(norm_col) == cat).cast(pl.Int8).alias(f"{col}__{cat}"))
    df = df.with_columns(exprs)
    df = df.drop(norm_col)
    return df
