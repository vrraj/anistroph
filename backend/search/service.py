"""Search service — search_dataset and get_search_contract.

Generic structured search over a registered dataset's full Parquet file.
Reads ``meta.parquet_path`` (all rows), never a partition, so search
results include every catalog product.
"""

from __future__ import annotations

from typing import Any, Optional

import polars as pl

from backend.datasets.registry import DatasetRegistry
from backend.search.filters import (
    FilterExpression,
    SortExpression,
    apply_filters,
    expand_semantic,
)


def search_dataset(
    dataset_registry: DatasetRegistry,
    dataset_id: str,
    filters: list[FilterExpression],
    sort: Optional[list[SortExpression]] = None,
    limit: int = 50,
    columns: Optional[list[str]] = None,
    search_config: Optional[Any] = None,
) -> dict[str, Any]:
    """Run a deterministic structured search over a dataset.

    Args:
        dataset_registry: the dataset registry.
        dataset_id: registered dataset to search.
        filters: list of FilterExpression (may include semantic filter names).
        sort: optional list of SortExpression (applied in order).
        limit: max rows to return (capped at 1000).
        columns: optional column subset to return.
        search_config: optional SearchConfig for semantic-filter expansion.

    Returns a dict with dataset_id, matched (count after filtering),
    returned, columns, rows, and applied_filters (the normalized filter
    expressions after semantic expansion, for audit/debugging).
    """
    limit = max(1, min(int(limit), 1000))
    meta = dataset_registry.get(dataset_id)
    if meta is None:
        raise ValueError(f"dataset {dataset_id!r} not registered")
    df = pl.read_parquet(meta.parquet_path)

    # Expand semantic filters into concrete predicates.
    expanded = expand_semantic(filters, search_config)

    df = apply_filters(df, expanded)

    matched = df.height

    if sort:
        known = set(df.columns)
        for s in sort:
            if s.field not in known:
                raise ValueError(f"unknown sort column {s.field!r}")
        # Apply sorts in reverse order so the first sort is the primary key.
        for s in reversed(sort):
            df = df.sort(s.field, descending=s.descending)

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"unknown columns: {missing}")
        df = df.select(columns)

    df = df.head(limit)

    # Serialize applied filters for audit (JSON-safe).
    applied = []
    for f in expanded:
        d = f.model_dump()
        # Convert enum to string value for JSON serialization.
        if "op" in d and hasattr(d["op"], "value"):
            d["op"] = d["op"].value
        applied.append(d)

    return {
        "dataset_id": dataset_id,
        "matched": matched,
        "returned": df.height,
        "columns": df.columns,
        "rows": df.to_dicts(),
        "applied_filters": applied,
    }


def get_search_contract(
    dataset_registry: DatasetRegistry,
    dataset_id: str,
    search_config: Optional[Any] = None,
    profile: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a self-describing search contract for a dataset.

    Merges the YAML ``search:`` config with live profile data (categorical
    values and numeric ranges) so the contract is current. Computed on-demand;
    never cached.

    Returns:
        dataset_id, supported_operators, searchable_fields (list with field,
        type, unit, operators, aliases, categorical_values or range,
        description), semantic_filters (list with name, type, description,
        and type-specific fields).
    """
    meta = dataset_registry.get(dataset_id)
    if meta is None:
        raise ValueError(f"dataset {dataset_id!r} not registered")

    supported = ["eq", "in", "gte", "lte", "between", "contains_range", "semantic"]

    # Build a lookup of column -> profile info.
    # profile["columns"] is a dict keyed by column name.
    profile_cols: dict[str, dict[str, Any]] = {}
    if profile and "columns" in profile:
        profile_cols = profile["columns"]

    fields_out: list[dict[str, Any]] = []
    if search_config is not None:
        for name, spec in search_config.searchable_fields.items():
            entry: dict[str, Any] = {
                "name": name,
                "field": spec.field,
                "operators": spec.operators,
                "unit": spec.unit,
                "aliases": spec.aliases,
                "description": spec.description,
            }
            # Enrich with live profile data.
            pc = profile_cols.get(spec.field)
            if pc:
                col_type = pc.get("type", "")
                entry["type"] = col_type
                if col_type == "categorical":
                    top = pc.get("top_values")
                    if top:
                        entry["categorical_values"] = [
                            t["value"] for t in top
                        ]
                elif col_type == "numeric":
                    if pc.get("min") is not None and pc.get("max") is not None:
                        entry["range"] = {
                            "min": pc.get("min"),
                            "max": pc.get("max"),
                        }
            fields_out.append(entry)

    semantic_out: list[dict[str, Any]] = []
    if search_config is not None:
        for name, spec in search_config.semantic_filters.items():
            entry: dict[str, Any] = {
                "name": name,
                "type": spec.type,
                "description": spec.description,
                "unit": spec.unit,
            }
            if spec.type == "range_contains":
                entry["min_field"] = spec.min_field
                entry["max_field"] = spec.max_field
            elif spec.type == "expands_to":
                entry["expands_to"] = spec.expands_to
            semantic_out.append(entry)

    return {
        "dataset_id": dataset_id,
        "supported_operators": supported,
        "searchable_fields": fields_out,
        "semantic_filters": semantic_out,
    }
