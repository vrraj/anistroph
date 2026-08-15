"""FeatureSpec — configuration-driven feature engineering definitions.

The Feature Engine interprets these definitions generically. It must never
know what ``temperature`` or ``vibration`` means.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field


class WindowTransform(BaseModel):
    """A windowed aggregation transform (e.g. mean over 6h)."""

    op: str
    windows: list[str] = Field(default_factory=list)


class CategoricalTransform(BaseModel):
    """Categorical encoding transform."""

    op: str = "categorical"
    encoding: str = "onehot"
    min_frequency: int = 1


# A transform entry is either a plain op name ("current") or a mapping
# describing parameters.
TransformDef = Union[str, dict[str, Any]]


class ColumnFeatureSpec(BaseModel):
    """Feature specification for a single source column."""

    column: str
    transforms: list[TransformDef] = Field(default_factory=list)


class FeatureSpec(BaseModel):
    """Full feature specification for a dataset."""

    dataset_id: str
    features: dict[str, ColumnFeatureSpec] = Field(default_factory=dict)

    def column_specs(self) -> list[ColumnFeatureSpec]:
        return list(self.features.values())

    def source_columns(self) -> list[str]:
        return [fs.column for fs in self.features.values()]


def _parse_transform(td: TransformDef) -> dict[str, Any]:
    if isinstance(td, str):
        return {"op": td}
    if isinstance(td, dict):
        if "op" in td:
            return dict(td)
        # form like {mean: {windows: [1h, 6h]}}
        if len(td) == 1:
            op, params = next(iter(td.items()))
            out: dict[str, Any] = {"op": op}
            if isinstance(params, dict):
                out.update(params)
            return out
        raise ValueError(f"ambiguous transform definition: {td!r}")
    raise TypeError(f"unsupported transform type: {type(td)!r}")


def normalize_transforms(transforms: list[TransformDef]) -> list[dict[str, Any]]:
    """Normalize transform entries to dicts with an ``op`` key."""
    return [_parse_transform(t) for t in transforms]


def load_feature_spec(path: str | Path) -> FeatureSpec:
    """Load a FeatureSpec from a YAML file.

    Expected top-level key ``features`` mapping feature-name -> {column, transforms}.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"invalid FeatureSpec file: {path}")
    data: dict[str, Any] = raw.get("features_spec", raw)
    dataset_id = data.get("dataset_id", "")
    features_raw = data.get("features", {})
    features: dict[str, ColumnFeatureSpec] = {}
    for feat_name, feat_def in features_raw.items():
        if not isinstance(feat_def, dict):
            raise ValueError(f"feature {feat_name!r} must be a mapping")
        column = feat_def.get("column", feat_name)
        transforms = feat_def.get("transforms", [])
        features[feat_name] = ColumnFeatureSpec(column=column, transforms=transforms)
    return FeatureSpec(dataset_id=dataset_id, features=features)
