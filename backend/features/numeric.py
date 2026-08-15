"""Numeric feature transforms.

Each transform operates on a Polars Series/Expr and is leakage-safe:
features at time T never use observations after T.

Window durations are expressed as Polars duration strings (e.g. "1h", "6h").
"""

from __future__ import annotations

import polars as pl


def current(expr: pl.Expr) -> pl.Expr:
    """The raw current value."""
    return expr


def mean(expr: pl.Expr) -> pl.Expr:
    """Rolling mean (window applied by engine with entity grouping)."""
    return expr


def min_(expr: pl.Expr) -> pl.Expr:
    return expr


def max_(expr: pl.Expr) -> pl.Expr:
    return expr


def std(expr: pl.Expr) -> pl.Expr:
    return expr


def median(expr: pl.Expr) -> pl.Expr:
    return expr


def delta(expr: pl.Expr) -> pl.Expr:
    """Difference from the first value in the window (current - window_start)."""
    return expr


def slope(expr: pl.Expr) -> pl.Expr:
    """Linear slope over the window (approximated as rolling linear regression slope)."""
    return expr


# Registry of simple rolling-aggregation ops.
# For these, the engine applies rolling_<op> over the window.
ROLLING_OPS: dict[str, str] = {
    "mean": "mean",
    "min": "min",
    "max": "max",
    "std": "std",
    "median": "median",
}

# Ops that need custom handling.
CUSTOM_OPS = {"current", "delta", "slope"}
