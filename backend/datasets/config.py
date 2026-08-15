"""Dataset configuration bundle — loads DatasetSpec, FeatureSpec, TargetSpec
from a single dataset.yaml file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from backend.datasets.spec import DatasetSpec, SplitSpec, ColumnSpec, ColumnType, ColumnRole
from backend.features.spec import FeatureSpec, ColumnFeatureSpec
from backend.targets.spec import TargetSpec


@dataclass
class DatasetConfig:
    """Bundle of all specs for a dataset."""

    dataset_spec: DatasetSpec
    feature_spec: Optional[FeatureSpec] = None
    target_spec: Optional[TargetSpec] = None

    @property
    def dataset_id(self) -> str:
        return self.dataset_spec.dataset_id


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load a complete dataset configuration from a single YAML file.

    The YAML may contain ``dataset``, ``features``, and ``target`` sections.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"invalid dataset config file: {path}")

    # --- DatasetSpec ---
    ds_raw = raw.get("dataset", raw)
    columns_raw = ds_raw.pop("columns", {})
    cols: dict[str, ColumnSpec] = {}
    for col_name, col_def in columns_raw.items():
        if not isinstance(col_def, dict):
            raise ValueError(f"column {col_name!r} must be a mapping")
        cols[col_name] = ColumnSpec(name=col_name, **col_def)
    split_raw = ds_raw.pop("split", {})
    dataset_spec = DatasetSpec(**ds_raw, columns=cols, split=SplitSpec(**split_raw))

    # --- FeatureSpec ---
    feature_spec = None
    features_raw = raw.get("features")
    if features_raw:
        features: dict[str, ColumnFeatureSpec] = {}
        for feat_name, feat_def in features_raw.items():
            if not isinstance(feat_def, dict):
                raise ValueError(f"feature {feat_name!r} must be a mapping")
            column = feat_def.get("column", feat_name)
            transforms = feat_def.get("transforms", [])
            features[feat_name] = ColumnFeatureSpec(column=column, transforms=transforms)
        feature_spec = FeatureSpec(dataset_id=dataset_spec.dataset_id, features=features)

    # --- TargetSpec ---
    target_spec = None
    target_raw = raw.get("target")
    if target_raw:
        target_spec = TargetSpec(**target_raw)

    return DatasetConfig(
        dataset_spec=dataset_spec,
        feature_spec=feature_spec,
        target_spec=target_spec,
    )
