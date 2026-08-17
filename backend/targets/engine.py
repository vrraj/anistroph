"""Target Engine — dispatches target construction based on TargetSpec.type.

This is the single entry point for target construction, used by the training
pipeline. It ensures entity isolation and no future leakage.
"""

from __future__ import annotations

import polars as pl

from backend.datasets.spec import DatasetSpec
from backend.targets.binary import build_binary_target
from backend.targets.horizon import build_future_event_target
from backend.targets.regression import build_regression_target
from backend.targets.spec import TargetSpec, TargetType


class TargetEngine:
    """Generic target construction engine."""

    def build_target(
        self,
        df: pl.DataFrame,
        spec: DatasetSpec,
        target_spec: TargetSpec,
    ) -> pl.DataFrame:
        """Build the target column and return df with it added."""
        if target_spec.type in (TargetType.BINARY, TargetType.CLASSIFICATION):
            return build_binary_target(df, target_spec)
        elif target_spec.type == TargetType.REGRESSION:
            return build_regression_target(df, target_spec)
        elif target_spec.type == TargetType.FUTURE_EVENT:
            if not spec.is_temporal():
                raise ValueError(
                    f"future_event target requires a temporal dataset (time_key); "
                    f"dataset {spec.dataset_id!r} has no time_key"
                )
            return build_future_event_target(
                df, target_spec, spec.entity_key, spec.time_key
            )
        else:
            raise ValueError(f"unsupported target type: {target_spec.type!r}")
