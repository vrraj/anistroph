"""Dataset validation — validates physical data against a DatasetSpec.

Validates required columns, types, and entity/time keys. Domain-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from backend.datasets.spec import ColumnType, DatasetSpec


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_dataset(df: pl.DataFrame, spec: DatasetSpec) -> ValidationReport:
    """Validate a Polars DataFrame against a DatasetSpec."""
    report = ValidationReport(valid=True)

    # 1. Required columns present.
    for col in spec.required_columns():
        if col not in df.columns:
            report.add_error(f"missing required column: {col!r}")

    if not report.valid:
        return report

    # 2. Type validation.
    for col_name, col_spec in spec.columns.items():
        dtype = df.schema[col_name]
        expected = col_spec.type
        ok = _check_type(dtype, expected)
        if not ok:
            report.add_error(
                f"column {col_name!r} has type {dtype}, expected {expected.value}"
            )

    # 3. Entity key non-null and non-empty.
    ek = spec.entity_key
    if ek in df.columns:
        null_count = df[ek].null_count()
        if null_count > 0:
            report.add_error(f"entity key {ek!r} has {null_count} null values")

    # 4. Time key parsing (if temporal).
    tk = spec.time_key
    if tk and tk in df.columns:
        dtype = df.schema[tk]
        if dtype not in (pl.Datetime, pl.Date, pl.Utf8):
            report.add_warning(
                f"time key {tk!r} has type {dtype}; expected datetime/date/str"
            )

    # 5. Duplicate entity+time check for temporal datasets.
    if tk and ek in df.columns and tk in df.columns:
        dup_count = df.group_by([ek, tk]).len().filter(pl.col("len") > 1).height
        if dup_count > 0:
            report.add_warning(f"{dup_count} duplicate (entity, time) key pairs found")

    return report


_POLARS_TYPE_MAP: dict[ColumnType, tuple] = {
    ColumnType.NUMERIC: (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                         pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64),
    ColumnType.CATEGORICAL: (pl.Utf8, pl.Categorical),
    ColumnType.BOOLEAN: (pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64),
    ColumnType.TIMESTAMP: (pl.Datetime, pl.Date, pl.Utf8),
    ColumnType.STRING: (pl.Utf8,),
}


def _check_type(dtype: pl.DataType, expected: ColumnType) -> bool:
    allowed = _POLARS_TYPE_MAP.get(expected, ())
    return dtype in allowed
