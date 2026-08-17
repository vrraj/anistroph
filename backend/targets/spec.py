"""TargetSpec — configurable target construction.

Supports these task types:
  - regression       → numerical outcome (XGBoost regressor)
  - classification   → binary class / probability (XGBoost classifier)
  - binary           → alias for classification (legacy)
  - future_event     → classification with a time horizon (legacy)

The ``classification`` type is the canonical name for binary classification
in v0.1. ``binary`` and ``future_event`` are kept as aliases for backward
compatibility with existing dataset configs. Future task types (forecasting,
anomaly detection) can be added to the enum without changing existing
datasets.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel


class TargetType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    BINARY = "binary"            # alias for classification (legacy)
    FUTURE_EVENT = "future_event"  # classification with horizon (legacy)

    @property
    def is_classification(self) -> bool:
        """True for any classification-like task type."""
        return self in (
            TargetType.CLASSIFICATION,
            TargetType.BINARY,
            TargetType.FUTURE_EVENT,
        )

    @property
    def is_regression(self) -> bool:
        """True for regression task types."""
        return self == TargetType.REGRESSION


class TargetSpec(BaseModel):
    """Specification for target construction."""

    name: str
    type: TargetType
    source_column: str
    horizon: Optional[str] = None  # e.g. "24h" — required for future_event
    positive_class: Any = 1  # value indicating the positive/event class

    def horizon_seconds(self) -> Optional[float]:
        if self.horizon is None:
            return None
        return parse_duration(self.horizon)


def parse_duration(s: str) -> float:
    """Parse a human duration string like '24h', '30m', '1d' into seconds."""
    s = s.strip()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if not s:
        raise ValueError("empty duration")
    unit = s[-1].lower()
    if unit not in units:
        raise ValueError(f"unknown duration unit {unit!r} in {s!r}")
    value = float(s[:-1])
    return value * units[unit]


def load_target_spec(path: str | Path) -> TargetSpec:
    """Load a TargetSpec from a YAML file (top-level ``target`` key)."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"invalid TargetSpec file: {path}")
    data: dict[str, Any] = raw.get("target", raw)
    return TargetSpec(**data)
