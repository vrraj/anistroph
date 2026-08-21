"""Filter expressions and operator application via Polars.

The filter engine is generic: it applies deterministic predicates to a
Polars DataFrame. Semantic filters (e.g. "operating_temperature") are
expanded into concrete FilterExpression objects before application.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import polars as pl
from pydantic import BaseModel, Field, model_validator

from backend.search.spec import SearchConfig


class Operator(str, Enum):
    EQ = "eq"
    IN = "in"
    GTE = "gte"
    LTE = "lte"
    BETWEEN = "between"
    CONTAINS_RANGE = "contains_range"
    SEMANTIC = "semantic"  # references a named semantic filter; expanded before application


# Operators that work on categorical/string columns.
_CATEGORICAL_OPS = {Operator.EQ, Operator.IN}
# Operators that work on numeric columns.
_NUMERIC_OPS = {Operator.EQ, Operator.IN, Operator.GTE, Operator.LTE,
                Operator.BETWEEN, Operator.CONTAINS_RANGE}


class FilterExpression(BaseModel):
    """A single structured filter.

    For ``contains_range``: provide ``value``, ``min_field``, ``max_field``.
    The predicate is ``min_field <= value AND max_field >= value`` (inclusive).
    For ``between``: provide ``low`` and ``high``.
    For ``eq``: provide ``value`` (scalar).
    For ``in``: provide ``value`` (list).
    For ``gte``/``lte``: provide ``value`` (scalar).
    """

    field: str
    op: Operator
    value: Optional[Any] = None
    # contains_range:
    min_field: Optional[str] = None
    max_field: Optional[str] = None
    # between:
    low: Optional[float] = None
    high: Optional[float] = None

    @model_validator(mode="after")
    def _validate_fields(self) -> "FilterExpression":
        if self.op == Operator.SEMANTIC:
            # Semantic filter references are expanded before application.
            # Only `field` is required (the semantic filter name); `value`
            # is needed for range_contains semantics but not for expands_to.
            return self
        if self.op == Operator.CONTAINS_RANGE:
            if self.min_field is None or self.max_field is None:
                raise ValueError(
                    "contains_range requires min_field and max_field"
                )
            if self.value is None:
                raise ValueError("contains_range requires value")
        elif self.op == Operator.BETWEEN:
            if self.low is None or self.high is None:
                raise ValueError("between requires low and high")
        elif self.op in (Operator.EQ, Operator.GTE, Operator.LTE):
            if self.value is None:
                raise ValueError(f"{self.op} requires value")
        elif self.op == Operator.IN:
            if self.value is None or not isinstance(self.value, list):
                raise ValueError("in requires value as a list")
        return self


class SortExpression(BaseModel):
    """A single sort directive."""

    field: str
    descending: bool = False


def apply_filter(df: pl.DataFrame, expr: FilterExpression) -> pl.DataFrame:
    """Apply a single filter expression to a DataFrame, returning the filtered frame."""
    op = expr.op
    if op == Operator.EQ:
        return df.filter(pl.col(expr.field) == expr.value)
    if op == Operator.IN:
        return df.filter(pl.col(expr.field).is_in(expr.value))
    if op == Operator.GTE:
        return df.filter(pl.col(expr.field) >= expr.value)
    if op == Operator.LTE:
        return df.filter(pl.col(expr.field) <= expr.value)
    if op == Operator.BETWEEN:
        return df.filter(
            (pl.col(expr.field) >= expr.low) & (pl.col(expr.field) <= expr.high)
        )
    if op == Operator.CONTAINS_RANGE:
        return df.filter(
            (pl.col(expr.min_field) <= expr.value)
            & (pl.col(expr.max_field) >= expr.value)
        )
    raise ValueError(f"unsupported operator: {op}")


def apply_filters(
    df: pl.DataFrame,
    filters: list[FilterExpression],
    columns: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Apply a list of filters (AND-combined). Validates fields against columns."""
    known = set(df.columns)
    for f in filters:
        # Validate the field(s) exist.
        check_fields = [f.field]
        if f.op == Operator.CONTAINS_RANGE:
            check_fields = [f.min_field, f.max_field]
        for cf in check_fields:
            if cf not in known:
                raise ValueError(f"unknown filter column {cf!r}")
    for f in filters:
        df = apply_filter(df, f)
    return df


def expand_semantic(
    filters: list[FilterExpression],
    search_config: Optional[SearchConfig],
) -> list[FilterExpression]:
    """Expand semantic filter references into concrete FilterExpressions.

    A filter is treated as a semantic reference when either:
    - its ``op`` is ``semantic`` (explicit), or
    - its ``field`` matches a semantic filter name in the SearchConfig.

    For ``range_contains`` semantics: the filter must supply ``value``; the
    expansion uses the semantic filter's ``min_field``/``max_field``.
    For ``expands_to`` semantics: the filter's value is ignored; the
    expansion is the literal list of predicates from the config.

    A filter with ``op=semantic`` whose field is NOT a known semantic filter
    raises a clear ValueError.
    """
    expanded: list[FilterExpression] = []
    for f in filters:
        semantic = None
        if search_config is not None:
            semantic = search_config.semantic_filter(f.field)

        # If op is explicitly semantic but no matching semantic filter exists,
        # raise a clear error.
        if f.op == Operator.SEMANTIC and semantic is None:
            available = list(search_config.semantic_filters.keys()) if search_config else []
            raise ValueError(
                f"unknown semantic filter {f.field!r}; "
                f"available: {available}"
            )

        # If not a semantic reference, pass through unchanged.
        if semantic is None:
            expanded.append(f)
            continue

        # Expand the semantic filter.
        if semantic.type == "range_contains":
            if f.value is None:
                raise ValueError(
                    f"semantic filter {f.field!r} (range_contains) requires value"
                )
            expanded.append(FilterExpression(
                field=semantic.min_field,
                op=Operator.CONTAINS_RANGE,
                value=f.value,
                min_field=semantic.min_field,
                max_field=semantic.max_field,
            ))
        elif semantic.type == "expands_to":
            for pred in semantic.expands_to:
                expanded.append(FilterExpression(
                    field=pred["field"],
                    op=Operator(pred["op"]),
                    value=pred.get("value"),
                    low=pred.get("low"),
                    high=pred.get("high"),
                    min_field=pred.get("min_field"),
                    max_field=pred.get("max_field"),
                ))
        else:
            raise ValueError(
                f"semantic filter {f.field!r} has unknown type {semantic.type!r}"
            )
    return expanded


def from_simple_dict(
    filters: Optional[dict[str, Any]],
) -> list[FilterExpression]:
    """Convert a simple equality/IN dict (sample_rows style) to FilterExpressions.

    ``{"col": value}`` -> eq
    ``{"col": [v1, v2]}`` -> in
    """
    if not filters:
        return []
    result: list[FilterExpression] = []
    for col, val in filters.items():
        if isinstance(val, list):
            result.append(FilterExpression(field=col, op=Operator.IN, value=val))
        else:
            result.append(FilterExpression(field=col, op=Operator.EQ, value=val))
    return result
