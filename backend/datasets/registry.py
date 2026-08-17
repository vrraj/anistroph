"""Dataset registry — lightweight local metadata persistence.

Maintains metadata independently from the physical Parquet file.
Uses a local JSON file for v0.1. Abstracted so another persistence
implementation could replace filesystem storage later.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from backend.datasets.spec import DatasetSpec
from backend.features.spec import FeatureSpec
from backend.targets.spec import TargetSpec


class DatasetMeta(BaseModel):
    """Metadata for a registered dataset."""

    dataset_id: str
    name: str
    version: str = "1"
    source: str
    row_count: int
    columns: list[str]
    entity_key: str
    time_key: Optional[str] = None
    target_name: Optional[str] = None
    target_type: Optional[str] = None
    feature_names: list[str] = Field(default_factory=list)
    created_at: str = ""
    data_start: Optional[str] = None
    data_end: Optional[str] = None
    parquet_path: str = ""
    spec_path: Optional[str] = None
    # Partitioned parquet paths (set when the dataset is partitioned at
    # registration time). Training uses train_parquet_path; evaluation uses
    # eval_parquet_path. None when no partition exists.
    partitioned: bool = False
    train_parquet_path: Optional[str] = None
    eval_parquet_path: Optional[str] = None
    validate_parquet_path: Optional[str] = None


class DatasetRegistry:
    """Lightweight filesystem-backed dataset registry."""

    def __init__(self, registry_path: str | Path = "artifacts/dataset_registry.json") -> None:
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._datasets: dict[str, DatasetMeta] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            for did, meta in raw.items():
                self._datasets[did] = DatasetMeta(**meta)

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(
                {k: v.model_dump() for k, v in self._datasets.items()},
                indent=2,
                default=str,
            )
        )

    def register(
        self,
        spec: DatasetSpec,
        source: str,
        row_count: int,
        parquet_path: str,
        data_start: Optional[str] = None,
        data_end: Optional[str] = None,
        feature_spec: Optional[FeatureSpec] = None,
        target_spec: Optional[TargetSpec] = None,
        spec_path: Optional[str] = None,
        partitioned: bool = False,
        train_parquet_path: Optional[str] = None,
        eval_parquet_path: Optional[str] = None,
        validate_parquet_path: Optional[str] = None,
    ) -> DatasetMeta:
        meta = DatasetMeta(
            dataset_id=spec.dataset_id,
            name=spec.name,
            source=source,
            row_count=row_count,
            columns=list(spec.columns.keys()),
            entity_key=spec.entity_key,
            time_key=spec.time_key,
            target_name=target_spec.name if target_spec else None,
            target_type=target_spec.type.value if target_spec else None,
            feature_names=list(feature_spec.features.keys()) if feature_spec else [],
            created_at=datetime.now(timezone.utc).isoformat(),
            data_start=data_start,
            data_end=data_end,
            parquet_path=parquet_path,
            spec_path=spec_path,
            partitioned=partitioned,
            train_parquet_path=train_parquet_path,
            eval_parquet_path=eval_parquet_path,
            validate_parquet_path=validate_parquet_path,
        )
        self._datasets[spec.dataset_id] = meta
        self._save()
        return meta

    def get(self, dataset_id: str) -> Optional[DatasetMeta]:
        self._load()
        return self._datasets.get(dataset_id)

    def list(self) -> list[DatasetMeta]:
        self._load()
        return list(self._datasets.values())

    def exists(self, dataset_id: str) -> bool:
        self._load()
        return dataset_id in self._datasets

    def remove(self, dataset_id: str) -> bool:
        if dataset_id in self._datasets:
            del self._datasets[dataset_id]
            self._save()
            return True
        return False
