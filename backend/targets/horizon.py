"""Future-event horizon target construction.

Generates a binary label that is 1 when an event (e.g. failure) occurs for
the same entity within the configured future horizon.

Critical leakage rules:
  - No future observations may enter the feature vector.
  - A failure on Machine B must never label Machine A (entity isolation).
  - The label at time T looks forward into (T, T+horizon] for the same entity.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from backend.targets.spec import TargetSpec


def build_future_event_target(df: pl.DataFrame, spec: TargetSpec, entity_key: str, time_key: str) -> pl.DataFrame:
    """Build a future-event horizon target.

    For each row at time T for entity E, label = 1 if any event
    (source_column == positive_class) occurs for entity E in (T, T+horizon].
    """
    horizon_seconds = spec.horizon_seconds()
    if horizon_seconds is None:
        raise ValueError(f"future_event target {spec.name!r} requires a horizon")
    horizon = timedelta(seconds=horizon_seconds)

    out_name = spec.name
    positive = spec.positive_class

    # Sort by entity then time.
    df = df.sort([entity_key, time_key])

    # For each entity, check if any event occurs in the future window.
    # We use a self-join approach: for each row, find events on the same entity
    # with event_time in (row_time, row_time + horizon].
    events = df.filter(pl.col(spec.source_column) == positive).select(
        [entity_key, time_key]
    ).rename({time_key: "_event_time"})

    # Join events to all rows on entity, then filter by time window.
    joined = df.join(events, on=entity_key, how="left")
    joined = joined.with_columns(
        (pl.col("_event_time") > pl.col(time_key)).alias("_after"),
        (pl.col("_event_time") <= pl.col(time_key) + pl.duration(seconds=int(horizon_seconds))).alias("_before_horizon"),
    )
    joined = joined.with_columns(
        (pl.col("_after") & pl.col("_before_horizon")).alias("_within")
    )
    # Aggregate: does any event fall in the window for this (entity, time)?
    has_event = joined.group_by([entity_key, time_key]).agg(
        pl.col("_within").any().alias(out_name)
    )
    df = df.join(has_event, on=[entity_key, time_key], how="left")
    df = df.with_columns(pl.col(out_name).fill_null(False).cast(pl.Int8).alias(out_name))
    return df
