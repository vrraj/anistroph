"""Model registry — persists trained models as durable artifacts.

Artifact layout:
    artifacts/models/<model_id>/
        model.joblib
        metadata.json
        feature_spec.json
        target_spec.json
        metrics.json

The registry is abstracted so another persistence implementation could
replace filesystem storage later. No MLflow for v0.1.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from backend.features.engine import FeatureMetadata
from backend.features.spec import FeatureSpec
from backend.targets.spec import TargetSpec


class ModelMetadata(BaseModel):
    """Metadata for a persisted model."""

    model_id: str
    model_type: str
    dataset_id: str
    dataset_version: str = "1"
    created_at: str = ""
    target_name: str = ""
    target_type: str = ""
    feature_names: list[str] = Field(default_factory=list)
    training_period: Optional[dict[str, str]] = None
    validation_period: Optional[dict[str, str]] = None
    test_period: Optional[dict[str, str]] = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    decision_threshold: float = 0.5
    parquet_path: str = ""
    artifact_path: str = ""


class ModelRegistry:
    """Filesystem-backed model registry."""

    def __init__(self, artifacts_dir: str | Path = "artifacts/models") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.artifacts_dir / "model_index.json"
        self._models: dict[str, ModelMetadata] = {}
        self._load()

    def _load(self) -> None:
        if self._index_path.exists():
            raw = json.loads(self._index_path.read_text())
            for mid, meta in raw.items():
                self._models[mid] = ModelMetadata(**meta)

    def _save(self) -> None:
        self._index_path.write_text(
            json.dumps({k: v.model_dump() for k, v in self._models.items()}, indent=2, default=str)
        )

    def register(
        self,
        model_id: str,
        model_type: str,
        dataset_id: str,
        target_spec: TargetSpec,
        feature_spec: FeatureSpec,
        feature_metadata: FeatureMetadata,
        metrics: dict[str, Any],
        hyperparameters: dict[str, Any],
        decision_threshold: float,
        training_period: Optional[dict[str, str]] = None,
        validation_period: Optional[dict[str, str]] = None,
        test_period: Optional[dict[str, str]] = None,
        parquet_path: str = "",
        dataset_version: str = "1",
    ) -> ModelMetadata:
        artifact_path = self.artifacts_dir / model_id
        artifact_path.mkdir(parents=True, exist_ok=True)

        meta = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            target_name=target_spec.name,
            target_type=target_spec.type.value,
            feature_names=feature_metadata.feature_names,
            training_period=training_period,
            validation_period=validation_period,
            test_period=test_period,
            hyperparameters=hyperparameters,
            metrics=metrics,
            decision_threshold=decision_threshold,
            parquet_path=parquet_path,
            artifact_path=str(artifact_path),
        )

        # Persist spec/metadata files alongside the model.
        (artifact_path / "metadata.json").write_text(meta.model_dump_json(indent=2))
        (artifact_path / "feature_spec.json").write_text(
            json.dumps(_feature_spec_to_dict(feature_spec), indent=2)
        )
        (artifact_path / "feature_metadata.json").write_text(
            json.dumps(feature_metadata.to_dict(), indent=2)
        )
        (artifact_path / "target_spec.json").write_text(
            json.dumps(target_spec.model_dump(), indent=2, default=str)
        )
        (artifact_path / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

        self._models[model_id] = meta
        self._save()
        return meta

    def get(self, model_id: str) -> Optional[ModelMetadata]:
        self._load()
        return self._models.get(model_id)

    def list(self) -> list[ModelMetadata]:
        self._load()
        return list(self._models.values())

    def exists(self, model_id: str) -> bool:
        self._load()
        return model_id in self._models

    def remove(self, model_id: str) -> bool:
        if model_id in self._models:
            del self._models[model_id]
            self._save()
            return True
        return False

    def delete(self, model_id: str) -> bool:
        """Remove a model from the registry and delete its artifact directory.

        Returns True if the model was found and removed, False if not found.
        The artifact directory is removed if it exists (best-effort — errors
        during file deletion are ignored to ensure the registry entry is
        always cleaned up).
        """
        meta = self.get(model_id)
        removed = self.remove(model_id)
        if removed and meta and meta.artifact_path:
            import shutil
            try:
                shutil.rmtree(meta.artifact_path, ignore_errors=True)
            except Exception:
                pass  # best-effort cleanup
        return removed

    def load_feature_metadata(self, model_id: str) -> FeatureMetadata:
        meta = self.get(model_id)
        if meta is None:
            raise KeyError(f"model {model_id!r} not found")
        path = Path(meta.artifact_path) / "feature_metadata.json"
        return FeatureMetadata.from_dict(json.loads(path.read_text()))

    def load_feature_spec(self, model_id: str) -> FeatureSpec:
        meta = self.get(model_id)
        if meta is None:
            raise KeyError(f"model {model_id!r} not found")
        path = Path(meta.artifact_path) / "feature_spec.json"
        return _feature_spec_from_dict(json.loads(path.read_text()))

    def load_target_spec(self, model_id: str) -> TargetSpec:
        meta = self.get(model_id)
        if meta is None:
            raise KeyError(f"model {model_id!r} not found")
        path = Path(meta.artifact_path) / "target_spec.json"
        return TargetSpec(**json.loads(path.read_text()))


def _feature_spec_to_dict(fs: FeatureSpec) -> dict[str, Any]:
    return {
        "dataset_id": fs.dataset_id,
        "features": {
            name: {"column": col.column, "transforms": col.transforms}
            for name, col in fs.features.items()
        },
    }


def _feature_spec_from_dict(d: dict[str, Any]) -> FeatureSpec:
    from backend.features.spec import ColumnFeatureSpec
    features = {}
    for name, fdef in d.get("features", {}).items():
        features[name] = ColumnFeatureSpec(column=fdef["column"], transforms=fdef.get("transforms", []))
    return FeatureSpec(dataset_id=d.get("dataset_id", ""), features=features)
