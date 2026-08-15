"""TargetSpec — configurable target construction.

Supports three conceptual target types architecturally:
  - binary
  - regression
  - future_event

Only ``future_event``/``binary`` is fully exercised by the v0.1 reference
implementation, but the architecture recognises all three.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel


class TargetType(str, Enum):
    BINARY = "binary"
    REGRESSION = "regression"
    FUTURE_EVENT = "future_event"


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
