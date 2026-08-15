"""Analytical engine — deterministic, independent of ML.

Generic operations: slice, filter, group, aggregate, compare.
Uses DuckDB/Polars. Independent of model training.
"""

from __future__ import annotations

from typing import Any, Optional

import duckdb
import polars as pl

from backend.datasets.registry import DatasetRegistry


def slice_data(
    dataset_id: str,
    dataset_registry: DatasetRegistry,
    dimensions: list[str],
    metric: str,
    aggregation: str = "mean",
    filters: Optional[dict[str, Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Slice a dataset by dimensions with an aggregation over a metric.

    Example:
        slice_data("predictive_maintenance", dimensions=["machine_type"],
                   metric="failure", aggregation="mean")
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

    # Validate aggregation.
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

    result = df.group_by(dimensions).agg(agg_map[aggregation]().alias(f"{metric}_{aggregation}"))

    if order_by:
        result = result.sort(order_by, descending=True)
    if limit:
        result = result.head(limit)

    return result.to_dicts()


def aggregate_data(
    dataset_id: str,
    dataset_registry: DatasetRegistry,
    group_by: list[str],
    aggregations: dict[str, str],
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Aggregate a dataset with multiple metrics.

    Args:
        aggregations: mapping of metric column -> aggregation function.
    """
    meta = dataset_registry.get(dataset_id)
    if meta is None:
        raise ValueError(f"dataset {dataset_id!r} not registered")

    df = pl.read_parquet(meta.parquet_path)
    if filters:
        for col, val in filters.items():
            if isinstance(val, list):
                df = df.filter(pl.col(col).is_in(val))
            else:
                df = df.filter(pl.col(col) == val)

    agg_map = {
        "mean": pl.col, "sum": pl.col, "min": pl.col, "max": pl.col,
        "count": pl.col, "std": pl.col, "median": pl.col,
    }
    exprs = []
    for col, agg in aggregations.items():
        if agg not in agg_map:
            raise ValueError(f"unsupported aggregation: {agg!r}")
        exprs.append(getattr(pl.col(col), agg)().alias(f"{col}_{agg}"))

    result = df.group_by(group_by).agg(exprs)
    return result.to_dicts()


def compare_data(
    dataset_id: str,
    dataset_registry: DatasetRegistry,
    dimension: str,
    metric: str,
    aggregation: str = "mean",
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Compare a metric across values of a dimension.

    Returns one row per dimension value with the aggregated metric,
    plus a relative difference from the overall mean.
    """
    meta = dataset_registry.get(dataset_id)
    if meta is None:
        raise ValueError(f"dataset {dataset_id!r} not registered")

    df = pl.read_parquet(meta.parquet_path)
    if filters:
        for col, val in filters.items():
            if isinstance(val, list):
                df = df.filter(pl.col(col).is_in(val))
            else:
                df = df.filter(pl.col(col) == val)

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

    result = df.group_by(dimension).agg(agg_map[aggregation]().alias(f"{metric}_{aggregation}"))
    overall = df.select(agg_map[aggregation]().alias("overall"))[0, 0]
    result = result.with_columns(
        ((pl.col(f"{metric}_{aggregation}") - overall) / (abs(overall) + 1e-12)).alias("relative_diff")
    )
    result = result.sort(f"{metric}_{aggregation}", descending=True)
    rows = result.to_dicts()
    for r in rows:
        r["overall"] = float(overall) if overall is not None else None
    return rows
