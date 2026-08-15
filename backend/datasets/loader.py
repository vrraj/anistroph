"""Dataset loader — CSV/Parquet ingestion with Parquet persistence.

Ingestion flow:
    read file
    → apply DatasetSpec
    → validate required columns
    → validate types
    → validate entity/time keys
    → (caller profiles + registers)
    → persist Parquet
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from backend.datasets.spec import ColumnType, DatasetSpec
from backend.datasets.validation import ValidationReport, validate_dataset


def read_source(path: str | Path, spec: DatasetSpec) -> pl.DataFrame:
    """Read a CSV or Parquet source file into a Polars DataFrame.

    For CSV, attempts to parse the time_key column as datetime if temporal.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pl.read_csv(str(path), try_parse_dates=True)
        df = _coerce_types(df, spec)
    elif suffix == ".parquet":
        df = pl.read_parquet(str(path))
    else:
        raise ValueError(f"unsupported file format: {suffix}")
    return df


def _coerce_types(df: pl.DataFrame, spec: DatasetSpec) -> pl.DataFrame:
    """Coerce CSV-loaded columns to the types declared in the DatasetSpec."""
    exprs = []
    for col_name, col_spec in spec.columns.items():
        if col_name not in df.columns:
            continue
        if col_spec.type == ColumnType.TIMESTAMP and df.schema[col_name] == pl.Utf8:
            exprs.append(pl.col(col_name).str.to_datetime(strict=False).alias(col_name))
        elif col_spec.type == ColumnType.NUMERIC and df.schema[col_name] == pl.Utf8:
            exprs.append(pl.col(col_name).cast(pl.Float64, strict=False).alias(col_name))
        elif col_spec.type == ColumnType.BOOLEAN and df.schema[col_name] in (pl.Utf8, pl.Int64):
            exprs.append(pl.col(col_name).cast(pl.Int8, strict=False).alias(col_name))
    if exprs:
        df = df.with_columns(exprs)
    return df


def persist_parquet(df: pl.DataFrame, path: str | Path) -> Path:
    """Persist a DataFrame as Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))
    return path


def load_parquet(path: str | Path) -> pl.DataFrame:
    """Load a persisted Parquet analytical dataset."""
    return pl.read_parquet(str(path))


def ingest(
    source_path: str | Path,
    spec: DatasetSpec,
    parquet_path: str | Path,
) -> tuple[pl.DataFrame, ValidationReport, Path]:
    """Full ingestion: read → validate → persist Parquet.

    Returns (dataframe, validation_report, parquet_path).
    """
    df = read_source(source_path, spec)
    report = validate_dataset(df, spec)
    if not report.valid:
        raise ValueError(
            f"dataset validation failed with {len(report.errors)} errors: "
            + "; ".join(report.errors)
        )
    path = persist_parquet(df, parquet_path)
    return df, report, path
