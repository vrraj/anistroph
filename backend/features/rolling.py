"""Rolling window feature transforms.

Provides leakage-safe rolling aggregations grouped by entity and ordered by
time. Features at time T only use observations up to and including T.
"""

from __future__ import annotations

import polars as pl


def rolling_aggregate(
    df: pl.DataFrame,
    value_col: str,
    entity_col: str,
    time_col: str,
    op: str,
    window: str,
    out_col: str,
) -> pl.DataFrame:
    """Compute a per-entity rolling aggregation that is leakage-safe.

    Uses ``rolling_*_by`` grouped per entity, ordered by time, with a trailing
    window that includes the current row but no future rows.
    """
    op_map = {
        "mean": "rolling_mean_by",
        "min": "rolling_min_by",
        "max": "rolling_max_by",
        "std": "rolling_std_by",
        "median": "rolling_median_by",
    }
    func_name = op_map.get(op)
    if func_name is None:
        raise ValueError(f"unsupported rolling op: {op!r}")

    df = df.sort([entity_col, time_col])
    if df.height == 0:
        # No rows: return with the output column filled as null.
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out_col))
    parts = []
    for (entity,), g in df.group_by(entity_col):
        g = g.sort(time_col)
        expr = getattr(pl.col(value_col), func_name)(
            by=pl.col(time_col),
            window_size=window,
            closed="right",
        ).alias(out_col)
        g = g.with_columns(expr)
        parts.append(g)
    return pl.concat(parts).sort([entity_col, time_col])
