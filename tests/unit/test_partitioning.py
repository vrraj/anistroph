"""Unit tests — dataset partitioning."""

from __future__ import annotations

import polars as pl

from backend.datasets.partitioning import partition_dataframe
from backend.datasets.spec import DatasetSpec, SplitSpec


class TestPartitioning:
    def test_chronological_split(self):
        spec = DatasetSpec(
            dataset_id="test", name="test", entity_key="id", time_key="ts",
            split=SplitSpec(strategy="chronological", train=0.7, validation=0.15, test=0.15),
        )
        df = pl.DataFrame({"ts": list(range(100)), "id": ["A"] * 100})
        parts = partition_dataframe(df, spec, train_pct=0.7, eval_pct=0.15, validate_pct=0.15)
        total = parts["train"].height + parts["evaluation"].height + parts["validate"].height
        assert total == 100
        assert parts["train"]["ts"].max() <= parts["evaluation"]["ts"].min()

    def test_random_split(self):
        spec = DatasetSpec(
            dataset_id="test", name="test", entity_key="id",
            split=SplitSpec(strategy="random", train=0.7, validation=0.15, test=0.15),
        )
        df = pl.DataFrame({"ts": list(range(100)), "id": ["A"] * 100})
        parts = partition_dataframe(df, spec, train_pct=0.7, eval_pct=0.15, validate_pct=0.15, seed=42)
        total = parts["train"].height + parts["evaluation"].height + parts["validate"].height
        assert total == 100
