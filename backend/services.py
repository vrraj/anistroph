"""Central application services — single entry point used by REST, MCP, and UI.

All interfaces ultimately invoke these same core Python services.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import polars as pl

from backend.analysis.slice import compare_data, slice_data
from backend.analysis.interesting import find_interesting_slices
from backend.datasets.config import DatasetConfig, load_dataset_config
from backend.datasets.loader import ingest
from backend.datasets.profiling import profile_dataset
from backend.datasets.registry import DatasetMeta, DatasetRegistry
from backend.datasets.spec import DatasetSpec
from backend.datasets.validation import validate_dataset
from backend.features.spec import FeatureSpec
from backend.ml.evaluation import evaluate_binary
from backend.ml.explain import explain_prediction
from backend.ml.inference import predict
from backend.ml.registry import ModelRegistry
from backend.ml.training import available_model_types, train_model
from backend.targets.spec import TargetSpec


# Default paths relative to repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_REGISTRY_PATH = _REPO_ROOT / "artifacts" / "dataset_registry.json"
_MODEL_REGISTRY_DIR = _REPO_ROOT / "artifacts" / "models"
_DATASETS_DIR = _REPO_ROOT / "datasets"


class AnistrophServices:
    """Singleton-style service container for all Anistroph operations."""

    def __init__(
        self,
        dataset_registry_path: str | Path = _DATASET_REGISTRY_PATH,
        model_registry_dir: str | Path = _MODEL_REGISTRY_DIR,
    ) -> None:
        self.dataset_registry = DatasetRegistry(dataset_registry_path)
        self.model_registry = ModelRegistry(model_registry_dir)
        self._config_cache: dict[str, DatasetConfig] = {}

    # --- Dataset operations ---

    def list_datasets(self) -> list[DatasetMeta]:
        return self.dataset_registry.list()

    def get_dataset(self, dataset_id: str) -> Optional[DatasetMeta]:
        return self.dataset_registry.get(dataset_id)

    def register_dataset_from_config(
        self,
        config_path: str | Path,
        source_path: str | Path,
        parquet_path: Optional[str | Path] = None,
    ) -> DatasetMeta:
        """Register a dataset from a config YAML + source data file."""
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = _REPO_ROOT / config_path
        config = load_dataset_config(config_path)
        spec = config.dataset_spec
        dataset_id = spec.dataset_id

        if parquet_path is None:
            parquet_path = _REPO_ROOT / "data" / "processed" / f"{dataset_id}.parquet"
        else:
            parquet_path = Path(parquet_path)
            if not parquet_path.is_absolute():
                parquet_path = _REPO_ROOT / parquet_path

        df, report, pq_path = ingest(source_path, spec, parquet_path)

        # Profile.
        profile = profile_dataset(df, spec)

        data_start = profile.get("time_range", {}).get("start") if spec.is_temporal() else None
        data_end = profile.get("time_range", {}).get("end") if spec.is_temporal() else None

        meta = self.dataset_registry.register(
            spec=spec,
            source=str(source_path),
            row_count=df.height,
            parquet_path=str(pq_path),
            data_start=data_start,
            data_end=data_end,
            feature_spec=config.feature_spec,
            target_spec=config.target_spec,
            spec_path=str(config_path),
        )
        self._config_cache[dataset_id] = config
        return meta

    def get_config(self, dataset_id: str) -> DatasetConfig:
        """Load the DatasetConfig for a registered dataset."""
        if dataset_id in self._config_cache:
            return self._config_cache[dataset_id]
        meta = self.dataset_registry.get(dataset_id)
        if meta is None or meta.spec_path is None:
            raise ValueError(f"no config available for dataset {dataset_id!r}")
        config = load_dataset_config(meta.spec_path)
        self._config_cache[dataset_id] = config
        return config

    def profile(self, dataset_id: str) -> dict[str, Any]:
        meta = self.dataset_registry.get(dataset_id)
        if meta is None:
            raise ValueError(f"dataset {dataset_id!r} not registered")
        config = self.get_config(dataset_id)
        df = pl.read_parquet(meta.parquet_path)
        return profile_dataset(df, config.dataset_spec)

    # --- Analysis operations ---

    def slice(self, dataset_id: str, dimensions: list[str], metric: str,
              aggregation: str = "mean", filters: Optional[dict[str, Any]] = None,
              limit: Optional[int] = None) -> list[dict[str, Any]]:
        return slice_data(dataset_id, self.dataset_registry, dimensions, metric, aggregation, filters, limit=limit)

    def compare(self, dataset_id: str, dimension: str, metric: str,
                aggregation: str = "mean", filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        return compare_data(dataset_id, self.dataset_registry, dimension, metric, aggregation, filters)

    def find_interesting_slices(self, dataset_id: str, metric: str,
                                dimensions: Optional[list[str]] = None,
                                min_sample_size: int = 100,
                                max_dimensions: int = 3,
                                aggregation: str = "mean",
                                filters: Optional[dict[str, Any]] = None,
                                top_k: int = 20) -> list[dict[str, Any]]:
        return find_interesting_slices(
            dataset_id, self.dataset_registry, metric, dimensions,
            min_sample_size, max_dimensions, aggregation, filters, top_k,
        )

    def sample_rows(self, dataset_id: str, n: int = 10,
                    filters: Optional[dict[str, Any]] = None,
                    columns: Optional[list[str]] = None,
                    sort_by: Optional[str] = None,
                    descending: bool = False) -> dict[str, Any]:
        """Return up to ``n`` raw rows from a registered dataset.

        Args:
            dataset_id: registered dataset to sample.
            n: maximum number of rows to return (capped at 1000).
            filters: optional equality filters, e.g. {"wafer_id": "WAFER_015000"}
                or {"etch_tool": ["ETCH_02", "ETCH_03"]} for IN-style filters.
            columns: optional column subset to return. If None, returns all.
            sort_by: optional column to sort by before sampling.
            descending: sort direction (only used when sort_by is set).

        Returns a dict with dataset_id, row_count (after filtering), columns
        returned, and a list of row dicts.
        """
        n = max(1, min(int(n), 1000))
        meta = self.dataset_registry.get(dataset_id)
        if meta is None:
            raise ValueError(f"dataset {dataset_id!r} not registered")
        df = pl.read_parquet(meta.parquet_path)

        if filters:
            for col, val in filters.items():
                if col not in df.columns:
                    raise ValueError(f"unknown filter column {col!r}")
                if isinstance(val, list):
                    df = df.filter(pl.col(col).is_in(val))
                else:
                    df = df.filter(pl.col(col) == val)

        matched = df.height
        if sort_by:
            if sort_by not in df.columns:
                raise ValueError(f"unknown sort column {sort_by!r}")
            df = df.sort(sort_by, descending=descending)

        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"unknown columns: {missing}")
            df = df.select(columns)

        df = df.head(n)
        return {
            "dataset_id": dataset_id,
            "row_count": matched,
            "returned": df.height,
            "columns": df.columns,
            "rows": df.to_dicts(),
        }

    # --- Model operations ---

    def list_model_types(self) -> list[str]:
        return available_model_types()

    def list_models(self) -> list:
        return self.model_registry.list()

    def get_model(self, model_id: str):
        return self.model_registry.get(model_id)

    def get_model_metrics(self, model_id: str) -> dict[str, Any]:
        meta = self.model_registry.get(model_id)
        if meta is None:
            raise ValueError(f"model {model_id!r} not found")
        return meta.metrics

    def train(self, dataset_id: str, target_name: str, model_type: str,
              model_parameters: Optional[dict[str, Any]] = None,
              model_id: Optional[str] = None) -> dict[str, Any]:
        config = self.get_config(dataset_id)
        return train_model(
            dataset_id=dataset_id,
            target_name=target_name,
            model_type=model_type,
            dataset_registry=self.dataset_registry,
            model_registry=self.model_registry,
            config=config,
            model_parameters=model_parameters,
            model_id=model_id,
        )

    # --- Prediction operations ---

    def predict(self, model_id: str, entity_id: Optional[str] = None,
                timestamp: Optional[str] = None, records: Optional[list[dict]] = None) -> dict[str, Any]:
        meta = self.model_registry.get(model_id)
        if meta is None:
            raise ValueError(f"model {model_id!r} not found")
        config = self.get_config(meta.dataset_id)
        return predict(
            model_id=model_id,
            model_registry=self.model_registry,
            dataset_registry=self.dataset_registry,
            config=config,
            entity_id=entity_id,
            timestamp=timestamp,
            records=records,
        )

    def explain(self, model_id: str, entity_id: Optional[str] = None,
                timestamp: Optional[str] = None, records: Optional[list[dict]] = None,
                top_k: int = 10) -> dict[str, Any]:
        meta = self.model_registry.get(model_id)
        if meta is None:
            raise ValueError(f"model {model_id!r} not found")
        config = self.get_config(meta.dataset_id)
        return explain_prediction(
            model_id=model_id,
            model_registry=self.model_registry,
            dataset_registry=self.dataset_registry,
            config=config,
            entity_id=entity_id,
            timestamp=timestamp,
            records=records,
            top_k=top_k,
        )


# Module-level singleton for convenience.
_services: Optional[AnistrophServices] = None


def get_services() -> AnistrophServices:
    global _services
    if _services is None:
        _services = AnistrophServices()
    return _services
