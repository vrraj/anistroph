"""DatasetSpec — domain-agnostic description of a dataset's meaning.

The DatasetSpec describes the structure and semantics of a dataset
independently of its physical data. It is the single source of truth that the
generic ML pipeline consults; no domain-specific assumptions live in the core
engine.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    STRING = "string"


class ColumnRole(str, Enum):
    IDENTIFIER = "identifier"
    FEATURE = "feature"
    EVENT = "event"
    TARGET = "target"
    METADATA = "metadata"
    IGNORE = "ignore"


class ColumnSpec(BaseModel):
    name: str
    type: ColumnType
    role: ColumnRole = ColumnRole.FEATURE
    description: Optional[str] = None


class SplitSpec(BaseModel):
    """Train/validation/test split configuration."""

    strategy: str = "chronological"
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        v = v.lower()
        if v not in {"chronological", "random"}:
            raise ValueError(f"split strategy must be 'chronological' or 'random', got {v!r}")
        return v


class DatasetSpec(BaseModel):
    """First-class dataset specification.

    Describes the meaning of a dataset independently of its physical data.
    Domain-specific column names (e.g. machine_id, temperature) live here, not
    in the generic pipeline.
    """

    dataset_id: str
    name: str
    entity_key: str
    time_key: Optional[str] = None
    columns: dict[str, ColumnSpec] = Field(default_factory=dict)
    split: SplitSpec = Field(default_factory=SplitSpec)

    def column(self, name: str) -> ColumnSpec:
        if name not in self.columns:
            raise KeyError(f"column {name!r} not declared in DatasetSpec {self.dataset_id!r}")
        return self.columns[name]

    @property
    def feature_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.role == ColumnRole.FEATURE]

    @property
    def identifier_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.role == ColumnRole.IDENTIFIER]

    @property
    def event_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.role == ColumnRole.EVENT]

    def required_columns(self) -> list[str]:
        """All declared column names."""
        return list(self.columns.keys())

    def is_temporal(self) -> bool:
        return self.time_key is not None


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    """Load a DatasetSpec from a YAML file.

    The YAML may contain a top-level ``dataset`` mapping or be flat.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"invalid DatasetSpec file: {path}")
    data: dict[str, Any] = raw.get("dataset", raw)
    columns_raw = data.pop("columns", {})
    cols: dict[str, ColumnSpec] = {}
    for col_name, col_def in columns_raw.items():
        if not isinstance(col_def, dict):
            raise ValueError(f"column {col_name!r} must be a mapping")
        cols[col_name] = ColumnSpec(name=col_name, **col_def)
    return DatasetSpec(**data, columns=cols)
