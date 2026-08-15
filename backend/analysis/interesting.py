"""Interesting slice discovery — deterministic multidimensional search.

Searches 1, 2, and 3-dimensional combinations of categorical columns,
computes the metric aggregation for each combination, and ranks by
difference from the overall baseline. No AI/LLM involved.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Optional

import polars as pl

from backend.datasets.registry import DatasetRegistry


def find_interesting_slices(
    dataset_id: str,
    dataset_registry: DatasetRegistry,
    metric: str,
    dimensions: Optional[list[str]] = None,
    min_sample_size: int = 100,
    max_dimensions: int = 3,
    aggregation: str = "mean",
    filters: Optional[dict[str, Any]] = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Find slices with the largest deviation from the overall metric baseline.

    Args:
        dataset_id: registered dataset to search.
        metric: metric column to aggregate.
        dimensions: categorical columns to combine. If None, auto-detect
            categorical columns from the dataset.
        min_sample_size: minimum rows per slice to be considered.
        max_dimensions: max number of dimensions to combine (1-3).
        aggregation: aggregation function (mean, median, etc.).
        filters: optional filters to apply before searching.
        top_k: number of top slices to return.

    Returns a list of slices sorted by absolute difference from baseline,
    each with: dimension values, row count, mean, median, std, difference
    from overall baseline.
    """
    meta = dataset_registry.get(dataset_id)
    if meta is None:
        raise ValueError(f"dataset {dataset_id!r} not registered")

    df = pl.read_parquet(meta.parquet_path)

    # Apply filters.
    if filters:
        for col, val in filters.items():
            if isinstance(val, list):
                df = df.filter(pl.col(col).is_in(val))
            else:
                df = df.filter(pl.col(col) == val)

    # Auto-detect categorical columns if not specified.
    if dimensions is None:
        dimensions = [c for c in df.columns if df[c].dtype == pl.Utf8 and c != metric]

    # Compute overall baseline.
    agg_map = {
        "mean": pl.col(metric).mean,
        "sum": pl.col(metric).sum,
        "min": pl.col(metric).min,
        "max": pl.col(metric).max,
        "count": pl.col(metric).count,
        "std": pl.col(metric).std,
        "median": pl.col(metric).median,
    }
    if aggregation not in agg_map:
        raise ValueError(f"unsupported aggregation: {aggregation!r}")

    overall_val = df.select(agg_map[aggregation]().alias("overall"))[0, 0]
    if overall_val is None:
        return []

    results: list[dict[str, Any]] = []

    # Search 1, 2, and 3-dimensional combinations.
    for n_dims in range(1, min(max_dimensions, len(dimensions)) + 1):
        for combo in combinations(dimensions, n_dims):
            grouped = df.group_by(list(combo)).agg(
                agg_map[aggregation]().alias("metric_value"),
                pl.col(metric).median().alias("metric_median"),
                pl.col(metric).std().alias("metric_std"),
                pl.len().alias("row_count"),
            )

            # Filter by min sample size.
            grouped = grouped.filter(pl.col("row_count") >= min_sample_size)

            for row in grouped.to_dicts():
                diff = row["metric_value"] - overall_val
                results.append({
                    "dimensions": list(combo),
                    "values": {d: row[d] for d in combo},
                    "row_count": row["row_count"],
                    "metric_value": float(row["metric_value"]),
                    "metric_median": float(row["metric_median"]) if row["metric_median"] is not None else None,
                    "metric_std": float(row["metric_std"]) if row["metric_std"] is not None else None,
                    "overall_baseline": float(overall_val),
                    "difference": float(diff),
                    "abs_difference": abs(float(diff)),
                })

    # Sort by absolute difference, descending.
    results.sort(key=lambda x: x["abs_difference"], reverse=True)
    return results[:top_k]
