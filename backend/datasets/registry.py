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


def _repo_root(registry_path: Path) -> Path:
    """Derive the repo root from the registry path (artifacts/dataset_registry.json)."""
    return registry_path.resolve().parent.parent


def _resolve_path(path_str: str, repo_root: Path) -> str:
    """Resolve a stored path that may be a host absolute path.

    The registry may have been created on the host (e.g. /Users/raj/.../datasets/foo/dataset.yaml)
    but is now being read inside a Docker container with a different filesystem layout
    (e.g. /app/datasets/foo/dataset.yaml). This function:

    1. Returns the path as-is if it already exists (works on the original host).
    2. Tries the path relative to ``repo_root`` (handles relative paths stored in the registry).
    3. If the path is absolute but doesn't exist, tries to find the matching subpath
       relative to ``repo_root`` (handles host absolute paths read in a container).
    """
    p = Path(path_str)
    # For relative paths, always resolve against repo_root (never CWD).
    if not p.is_absolute():
        candidate = repo_root / path_str
        if candidate.exists():
            return str(candidate)
        return str(candidate)  # return resolved path even if it doesn't exist yet
    # Absolute path: return as-is if it exists (works on the original host).
    if p.exists():
        return str(p)
    # Absolute but doesn't exist: try to strip the host prefix and resolve
    # relative to repo_root (handles host paths read in a container).
    # e.g. /Users/raj/Documents/Raj/anistroph/datasets/foo/dataset.yaml
    #      → repo_root / datasets/foo/dataset.yaml
    try:
        rel = p.relative_to(repo_root)
        candidate = repo_root / rel
        if candidate.exists():
            return str(candidate)
    except ValueError:
        pass
    # Walk the path components from the start, looking for the first
    # segment that exists under repo_root.
    parts = p.parts
    for i in range(1, len(parts)):
        candidate = repo_root / Path(*parts[i:])
        if candidate.exists():
            return str(candidate)
    # Last resort: return the original path (caller will get a clear FileNotFoundError)
    return path_str


_PATH_FIELDS = (
    "parquet_path",
    "spec_path",
    "train_parquet_path",
    "eval_parquet_path",
    "validate_parquet_path",
)


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
            root = _repo_root(self.path)
            for did, meta in raw.items():
                # Resolve stored paths that may be host absolute paths.
                for field in _PATH_FIELDS:
                    val = meta.get(field)
                    if val:
                        meta[field] = _resolve_path(val, root)
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
