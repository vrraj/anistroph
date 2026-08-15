"""Feature Engine — single engine used by both training and inference.

Interprets a FeatureSpec generically. It must never know what
``temperature`` or ``vibration`` means. All transforms are leakage-safe:
features at time T never use observations after T.

The engine produces a feature matrix (Polars DataFrame) from raw observations
plus the FeatureSpec. Categorical encodings are *fitted* during training and
the learned categories are stored in ``FeatureMetadata`` so inference applies
the identical encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from backend.datasets.spec import DatasetSpec
from backend.features.categorical import (
    fit_onehot,
    onehot_columns,
    transform_onehot,
)
from backend.features.rolling import rolling_aggregate
from backend.features.spec import FeatureSpec, normalize_transforms
from backend.features.temporal import day_of_week, elapsed_time, hour_of_day


@dataclass
class FeatureMetadata:
    """Learned state from fitting the feature engine (categorical categories)."""

    categorical_categories: dict[str, list[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "categorical_categories": self.categorical_categories,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureMetadata":
        return cls(
            categorical_categories=d.get("categorical_categories", {}),
            feature_names=d.get("feature_names", []),
        )


class FeatureEngine:
    """Generic, configuration-driven feature engine.

    Used identically by training and inference. The persisted model artifact
    retains the FeatureSpec and FeatureMetadata so inference reconstructs the
    exact same feature vector.
    """

    def build_features(
        self,
        df: pl.DataFrame,
        spec: DatasetSpec,
        feature_spec: FeatureSpec,
        metadata: FeatureMetadata | None = None,
        fit: bool = False,
    ) -> tuple[pl.DataFrame, FeatureMetadata]:
        """Build feature columns from raw observations.

        Args:
            df: raw observations (must include entity_key, time_key, source cols).
            spec: DatasetSpec.
            feature_spec: FeatureSpec describing transforms.
            metadata: previously-fit metadata (required for inference when
                fit=False and categorical features exist).
            fit: if True, learn categorical categories from this data.

        Returns:
            (feature_dataframe, metadata) where feature_dataframe includes
            the entity_key, time_key, and all engineered feature columns.
        """
        if metadata is None:
            metadata = FeatureMetadata()

        if spec.is_temporal():
            df = df.sort([spec.entity_key, spec.time_key])

        # Work on a working copy that retains all source columns.
        work = df
        feature_cols: list[str] = []
        keep_keys = [spec.entity_key] + ([spec.time_key] if spec.time_key else [])

        for feat_name, col_spec in feature_spec.features.items():
            source_col = col_spec.column
            transforms = normalize_transforms(col_spec.transforms)

            for t in transforms:
                op = t["op"]

                if op == "categorical":
                    cats = metadata.categorical_categories.get(source_col)
                    if cats is None and fit:
                        cats = fit_onehot(work, source_col, min_frequency=t.get("min_frequency", 1))
                        metadata.categorical_categories[source_col] = cats
                    if cats is None:
                        raise ValueError(
                            f"categorical feature {source_col!r} has no learned categories; "
                            "call build_features with fit=True during training first."
                        )
                    work = transform_onehot(work, source_col, cats)
                    for c in onehot_columns(source_col, cats):
                        feature_cols.append(c)

                elif op == "current":
                    out_name = f"{source_col}_current"
                    if out_name != source_col:
                        work = work.with_columns(pl.col(source_col).alias(out_name))
                    feature_cols.append(out_name)

                elif op in ("mean", "min", "max", "std", "median"):
                    windows = t.get("windows", [])
                    for w in windows:
                        out_name = f"{source_col}_{op}_{w}"
                        work = rolling_aggregate(
                            work, source_col, spec.entity_key, spec.time_key, op, w, out_name
                        )
                        feature_cols.append(out_name)

                elif op == "slope":
                    windows = t.get("windows", [])
                    for w in windows:
                        out_name = f"{source_col}_slope_{w}"
                        work = _rolling_slope(
                            work, source_col, spec.entity_key, spec.time_key, w, out_name
                        )
                        feature_cols.append(out_name)

                elif op == "delta":
                    windows = t.get("windows", [])
                    for w in windows:
                        out_name = f"{source_col}_delta_{w}"
                        work = _rolling_delta(
                            work, source_col, spec.entity_key, spec.time_key, w, out_name
                        )
                        feature_cols.append(out_name)

                elif op in ("hour_of_day", "day_of_week", "elapsed_time"):
                    out_name = op
                    if op == "hour_of_day":
                        work = hour_of_day(work, spec.time_key, out_name)
                    elif op == "day_of_week":
                        work = day_of_week(work, spec.time_key, out_name)
                    else:
                        work = elapsed_time(work, spec.entity_key, spec.time_key, out_name)
                    feature_cols.append(out_name)

                else:
                    raise ValueError(f"unsupported transform op: {op!r}")

        metadata.feature_names = feature_cols
        result = work.select(keep_keys + feature_cols)
        return result, metadata


def _rolling_slope(
    df: pl.DataFrame,
    value_col: str,
    entity_col: str,
    time_col: str,
    window: str,
    out_col: str,
) -> pl.DataFrame:
    """Approximate rolling linear-regression slope over a time window.

    slope = cov(t, y) / var(t) computed over the trailing window.
    Leakage-safe: only uses rows up to and including the current time.
    """
    df = df.sort([entity_col, time_col])
    # Compute t as seconds elapsed from the entity's first timestamp.
    df = df.with_columns(
        (pl.col(time_col) - pl.col(time_col).min().over(entity_col))
        .dt.total_seconds()
        .alias("_t")
    )
    df = df.with_columns((pl.col("_t") * pl.col(value_col)).alias("_ty"))
    parts = []
    for (entity,), g in df.group_by(entity_col):
        g = g.sort(time_col)
        g = g.with_columns(
            pl.col("_t").rolling_mean_by(by=pl.col(time_col), window_size=window, closed="right").alias("_t_mean"),
            pl.col(value_col).rolling_mean_by(by=pl.col(time_col), window_size=window, closed="right").alias("_y_mean"),
            pl.col("_ty").rolling_mean_by(by=pl.col(time_col), window_size=window, closed="right").alias("_ty_mean"),
            pl.col("_t").rolling_var_by(by=pl.col(time_col), window_size=window, closed="right").alias("_t_var"),
        )
        g = g.with_columns(
            ((pl.col("_ty_mean") - pl.col("_t_mean") * pl.col("_y_mean")) / pl.col("_t_var")).alias(out_col)
        )
        parts.append(g)
    result = pl.concat(parts).drop(["_t", "_ty", "_t_mean", "_y_mean", "_ty_mean", "_t_var"])
    return result.sort([entity_col, time_col])


def _rolling_delta(
    df: pl.DataFrame,
    value_col: str,
    entity_col: str,
    time_col: str,
    window: str,
    out_col: str,
) -> pl.DataFrame:
    """Current value minus the value at the start of the trailing window."""
    df = df.sort([entity_col, time_col])
    parts = []
    for (entity,), g in df.group_by(entity_col):
        g = g.sort(time_col)
        g = g.with_columns(
            pl.col(value_col).rolling_min_by(by=pl.col(time_col), window_size=window, closed="right").alias("_w_min"),
        )
        g = g.with_columns((pl.col(value_col) - pl.col("_w_min")).alias(out_col))
        parts.append(g.drop("_w_min"))
    return pl.concat(parts).sort([entity_col, time_col])
