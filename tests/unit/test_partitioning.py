"""Unit tests — dataset partitioning."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from backend.datasets.partitioning import (
    EVAL,
    PARTITION_NAMES,
    TRAIN,
    VALIDATE,
    partition_dataframe,
    partition_summary,
    persist_partitions,
    resolve_split_percentages,
)
from backend.datasets.spec import DatasetSpec, SplitSpec


def _make_temporal_spec(dataset_id: str = "test") -> DatasetSpec:
    return DatasetSpec(
        dataset_id=dataset_id,
        name="Test",
        entity_key="id",
        time_key="ts",
        split=SplitSpec(strategy="chronological", train=0.8, validation=0.0, test=0.2),
    )


def _make_non_temporal_spec(dataset_id: str = "test") -> DatasetSpec:
    return DatasetSpec(
        dataset_id=dataset_id,
        name="Test",
        entity_key="id",
        split=SplitSpec(strategy="random", train=0.8, validation=0.0, test=0.2),
    )


def _make_df(n: int = 100) -> pl.DataFrame:
    return pl.DataFrame({
        "id": [f"row_{i}" for i in range(n)],
        "ts": pl.datetime_range(
            start=pl.datetime(2026, 1, 1),
            end=pl.datetime(2026, 1, 1) + pl.duration(hours=n - 1),
            interval="1h",
            eager=True,
        ),
        "value": list(range(n)),
    })


class TestResolveSplitPercentages:
    def test_yaml_overrides_env(self):
        spec = _make_temporal_spec()  # train=0.8, test=0.2
        train, eval_pct, validate = resolve_split_percentages(
            spec, env_train=0.5, env_eval=0.5, env_validate=0.0,
        )
        assert train == pytest.approx(0.8)
        assert eval_pct == pytest.approx(0.2)
        assert validate == pytest.approx(0.0)

    def test_env_defaults_when_yaml_not_customised(self):
        spec = DatasetSpec(
            dataset_id="t",
            name="T",
            entity_key="id",
            split=SplitSpec(train=1.0, validation=0.0, test=0.0),
        )
        train, eval_pct, validate = resolve_split_percentages(
            spec, env_train=0.7, env_eval=0.2, env_validate=0.1,
        )
        assert train == pytest.approx(0.7)
        assert eval_pct == pytest.approx(0.2)
        assert validate == pytest.approx(0.1)

    def test_normalises_to_sum_one(self):
        spec = _make_temporal_spec()
        train, eval_pct, validate = resolve_split_percentages(
            spec, env_train=0.8, env_eval=0.2, env_validate=0.0,
        )
        assert train + eval_pct + validate == pytest.approx(1.0)


class TestPartitionDataframe:
    def test_temporal_chronological(self):
        df = _make_df(100)
        spec = _make_temporal_spec()
        parts = partition_dataframe(df, spec, 0.8, 0.2, 0.0)
        assert parts[TRAIN].height == 80
        assert parts[EVAL].height == 20
        assert parts[VALIDATE].height == 0
        # Oldest rows in train, newest in eval.
        assert parts[TRAIN]["ts"].max() < parts[EVAL]["ts"].min()

    def test_non_temporal_random(self):
        df = _make_df(100)
        spec = _make_non_temporal_spec()
        parts = partition_dataframe(df, spec, 0.8, 0.2, 0.0, seed=42)
        assert parts[TRAIN].height == 80
        assert parts[EVAL].height == 20

    def test_three_way_split(self):
        df = _make_df(100)
        spec = _make_temporal_spec()
        parts = partition_dataframe(df, spec, 0.7, 0.2, 0.1)
        assert parts[TRAIN].height == 70
        assert parts[VALIDATE].height == 10
        assert parts[EVAL].height == 20

    def test_empty_partition_has_schema(self):
        df = _make_df(100)
        spec = _make_temporal_spec()
        parts = partition_dataframe(df, spec, 1.0, 0.0, 0.0)
        assert parts[TRAIN].height == 100
        assert parts[EVAL].height == 0
        # Empty partition still has columns.
        assert parts[EVAL].columns == df.columns

    def test_normalises_percentages(self):
        df = _make_df(100)
        spec = _make_non_temporal_spec()
        # 4 + 1 = 5 → normalised to 0.8 / 0.2.
        parts = partition_dataframe(df, spec, 4.0, 1.0, 0.0)
        assert parts[TRAIN].height == 80
        assert parts[EVAL].height == 20

    def test_zero_total_raises(self):
        df = _make_df(10)
        spec = _make_non_temporal_spec()
        with pytest.raises(ValueError):
            partition_dataframe(df, spec, 0.0, 0.0, 0.0)


class TestPersistPartitions:
    def test_persist_and_skip_empty(self, tmp_path):
        df = _make_df(100)
        spec = _make_temporal_spec()
        parts = partition_dataframe(df, spec, 0.8, 0.2, 0.0)
        paths = persist_partitions(parts, tmp_path, "test_ds")

        assert paths[TRAIN] is not None
        assert paths[EVAL] is not None
        assert paths[VALIDATE] is None  # empty → skipped

        assert Path(paths[TRAIN]).exists()
        assert Path(paths[EVAL]).exists()
        assert Path(paths[TRAIN]).name == "test_ds.train.parquet"
        assert Path(paths[EVAL]).name == "test_ds.evaluation.parquet"

        # Verify content.
        train_df = pl.read_parquet(paths[TRAIN])
        assert train_df.height == 80

    def test_all_empty(self, tmp_path):
        empty = pl.DataFrame({"id": [], "ts": [], "value": []})
        paths = persist_partitions(
            {TRAIN: empty, EVAL: empty, VALIDATE: empty},
            tmp_path,
            "ds",
        )
        for name in PARTITION_NAMES:
            assert paths[name] is None


class TestPartitionSummary:
    def test_summary(self):
        df = _make_df(100)
        spec = _make_temporal_spec()
        parts = partition_dataframe(df, spec, 0.8, 0.2, 0.0)
        summary = partition_summary(parts)
        assert summary["total"] == 100
        assert summary[TRAIN]["row_count"] == 80
        assert summary[EVAL]["row_count"] == 20
        assert summary[VALIDATE]["row_count"] == 0
        assert summary[TRAIN]["pct"] == pytest.approx(0.8)
