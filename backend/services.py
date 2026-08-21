"""Central application services — single entry point used by REST, MCP, and UI.

All interfaces ultimately invoke these same core Python services.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import polars as pl

from backend.analysis.slice import compare_data, slice_data
from backend.analysis.interesting import find_interesting_slices
from backend.config import get_settings
from backend.datasets.config import DatasetConfig, load_dataset_config
from backend.datasets.loader import ingest
from backend.datasets.partitioning import (
    PARTITION_NAMES,
    partition_dataframe,
    partition_summary,
    persist_partitions,
    resolve_split_percentages,
)
from backend.datasets.profiling import profile_dataset
from backend.datasets.registry import DatasetMeta, DatasetRegistry
from backend.datasets.spec import DatasetSpec
from backend.datasets.validation import validate_dataset
from backend.features.spec import FeatureSpec
from backend.ml.evaluation import evaluate_binary
from backend.ml.evaluation_runner import evaluate_on_eval_set, find_evaluation_slices
from backend.ml.explain import explain_prediction
from backend.ml.inference import predict
from backend.ml.registry import ModelRegistry
from backend.ml.training import available_model_types, train_model
from backend.search.filters import FilterExpression, SortExpression, from_simple_dict
from backend.search.service import get_search_contract as _get_search_contract
from backend.search.service import search_dataset as _search_dataset
from backend.search.spec import SearchConfig
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
        """Register a dataset from a config YAML + source data file.

        Always partitions the ingested data into train / evaluation / validate
        Parquet files. Split percentages are resolved with YAML-overrides-.env
        precedence: the YAML ``split:`` section wins if customised, otherwise
        the ``.env`` defaults (TRAIN_DATASET_PCT, EVAL_DATASET_PCT,
        VALIDATE_DATASET_PCT) are used.

        The full dataset is also persisted at ``parquet_path`` for backward
        compatibility (e.g. profiling, sample_rows).
        """
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

        # Profile (on the full dataset).
        profile = profile_dataset(df, spec)

        data_start = profile.get("time_range", {}).get("start") if spec.is_temporal() else None
        data_end = profile.get("time_range", {}).get("end") if spec.is_temporal() else None

        # --- Partition into train / evaluation / validate ---
        settings = get_settings()
        train_pct, eval_pct, validate_pct = resolve_split_percentages(
            spec,
            env_train=settings.TRAIN_DATASET_PCT,
            env_eval=settings.EVAL_DATASET_PCT,
            env_validate=settings.VALIDATE_DATASET_PCT,
        )

        partitions = partition_dataframe(
            df, spec, train_pct, eval_pct, validate_pct,
        )

        partition_dir = pq_path.parent
        partition_paths = persist_partitions(partitions, partition_dir, dataset_id)

        summary = partition_summary(partitions)

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
            partitioned=True,
            train_parquet_path=partition_paths.get("train"),
            eval_parquet_path=partition_paths.get("evaluation"),
            validate_parquet_path=partition_paths.get("validate"),
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

        Internally delegates to the generic search engine (sample_rows does
        not use semantic filters or the search contract — it stays a simple
        row-inspection tool with equality/IN filters only).
        """
        filter_exprs = from_simple_dict(filters)
        sort_exprs = [SortExpression(field=sort_by, descending=descending)] if sort_by else None
        result = _search_dataset(
            self.dataset_registry, dataset_id, filter_exprs,
            sort=sort_exprs, limit=n, columns=columns,
        )
        # Preserve the original return shape (row_count, not matched).
        return {
            "dataset_id": result["dataset_id"],
            "row_count": result["matched"],
            "returned": result["returned"],
            "columns": result["columns"],
            "rows": result["rows"],
        }

    # --- Search operations ---

    def search(self, dataset_id: str,
               filters: list[FilterExpression],
               sort: Optional[list[SortExpression]] = None,
               limit: int = 50,
               columns: Optional[list[str]] = None) -> dict[str, Any]:
        """Run a deterministic structured search over a dataset.

        Supports operators eq, in, gte, lte, between, contains_range, plus
        semantic filter expansion (when the dataset declares a ``search:``
        config). Reads the full dataset Parquet (all rows), never a partition.

        Returns a dict with dataset_id, matched, returned, columns, rows,
        and applied_filters (the normalized filter expressions after semantic
        expansion, for audit/debugging).
        """
        config = self.get_config(dataset_id)
        search_config = config.search_config
        return _search_dataset(
            self.dataset_registry, dataset_id, filters,
            sort=sort, limit=limit, columns=columns,
            search_config=search_config,
        )

    def get_search_contract(self, dataset_id: str) -> dict[str, Any]:
        """Return a self-describing search contract for a dataset.

        Lists searchable fields (with types, units, operators, aliases,
        categorical values / numeric ranges from the live profile) and
        semantic filters. Computed on-demand from the YAML ``search:`` config
        merged with the current dataset profile.
        """
        config = self.get_config(dataset_id)
        if config.search_config is None:
            raise ValueError(
                f"dataset {dataset_id!r} has no search configuration "
                "(no 'search:' section in its dataset.yaml)"
            )
        profile = self.profile(dataset_id)
        return _get_search_contract(
            self.dataset_registry, dataset_id,
            search_config=config.search_config,
            profile=profile,
        )

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

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry and remove its artifact directory.

        Returns True if the model was found and deleted, False if not found.
        """
        return self.model_registry.delete(model_id)

    def train(self, dataset_id: str, target_name: str,
              model_type: Optional[str] = None,
              model_parameters: Optional[dict[str, Any]] = None,
              model_id: Optional[str] = None) -> dict[str, Any]:
        config = self.get_config(dataset_id)
        # When the dataset is partitioned, train only on train.parquet.
        # The held-out evaluation file is never used during model fitting.
        dmeta = self.dataset_registry.get(dataset_id)
        train_parquet = None
        validate_parquet = None
        is_partitioned = False
        if dmeta is not None and dmeta.partitioned and dmeta.train_parquet_path:
            train_parquet = dmeta.train_parquet_path
            validate_parquet = dmeta.validate_parquet_path
            is_partitioned = True
        return train_model(
            dataset_id=dataset_id,
            target_name=target_name,
            model_type=model_type,
            dataset_registry=self.dataset_registry,
            model_registry=self.model_registry,
            config=config,
            model_parameters=model_parameters,
            model_id=model_id,
            parquet_path=train_parquet,
            is_train_partition=is_partitioned,
            validate_parquet_path=validate_parquet,
        )

    # --- Prediction operations ---

    def get_model_inputs(self, model_id: str) -> dict[str, Any]:
        """Return the prediction input schema for a model.

        Tells the caller what they need to supply to predict with this model:
        - prediction_mode: 'entity_lookup' (rolling-window transforms require
          history, so only entity_id+timestamp works) or
          'entity_lookup_or_records' (only current/categorical transforms,
          so records with raw values also work)
        - entity_key: the column name used for entity_id lookups
        - requires_timestamp: whether timestamp is needed (temporal datasets)
        - required_columns: for records-based prediction, the source columns
          the caller must include, with their types and transforms
        """
        meta = self.model_registry.get(model_id)
        if meta is None:
            raise ValueError(f"model {model_id!r} not found")
        config = self.get_config(meta.dataset_id)
        spec = config.dataset_spec
        fs = self.model_registry.load_feature_spec(model_id)

        # Transforms that require historical observations (rolling windows,
        # time-derived features). If any are present, records-based prediction
        # won't work — the caller must use entity_id + timestamp so Anistroph
        # can load history and build the rolling features.
        history_transforms = {"mean", "min", "max", "std", "median", "slope",
                              "delta", "hour_of_day", "day_of_week", "elapsed_time"}

        required_columns = []
        has_history_transforms = False
        for feat_name, col_spec in fs.features.items():
            col = spec.columns.get(col_spec.column)
            transforms = [
                t if isinstance(t, str) else list(t.keys())[0]
                for t in col_spec.transforms
            ]
            if any(t in history_transforms for t in transforms):
                has_history_transforms = True
            required_columns.append({
                "column": col_spec.column,
                "type": col.type.value if col else "unknown",
                "transforms": transforms,
            })

        if has_history_transforms:
            prediction_mode = "entity_lookup"
            requires_timestamp = True
            mode_note = (
                "This model uses rolling-window transforms that require historical "
                "observations. Use entity_id + timestamp — Anistroph loads the entity's "
                "history and builds the rolling features. Records-based prediction is "
                "not supported for this model."
            )
        else:
            prediction_mode = "entity_lookup_or_records"
            requires_timestamp = False
            mode_note = (
                "Both prediction modes work: (1) entity_id (+ timestamp if temporal) "
                "to look up an existing row, or (2) records — a list of dicts with the "
                "required_columns below. Column order does not matter — columns are "
                "matched by name."
            )

        return {
            "model_id": model_id,
            "dataset_id": meta.dataset_id,
            "target_name": meta.target_name,
            "target_type": meta.target_type,
            "prediction_mode": prediction_mode,
            "entity_key": spec.entity_key,
            "requires_timestamp": requires_timestamp,
            "inference_history_window": fs.max_history_window(),
            "required_columns": required_columns,
            "note": mode_note,
        }

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

    # --- Evaluation operations ---

    def evaluate_model(
        self,
        model_id: str,
        sample_size: int = 50,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Evaluate a trained model against the held-out evaluation partition.

        Loads ``evaluation.parquet`` for the model's dataset, runs inference
        using the persisted model, and compares predictions against the known
        actual target values.

        Args:
            model_id: registered model to evaluate.
            sample_size: number of prediction-vs-actual rows to include in the
                response (capped at 1000). Aggregate metrics are over the full
                evaluation set (or the filtered subset when ``filters`` is
                provided).
            filters: optional equality / IN-style filters, e.g.
                ``{"city": "Saratoga"}`` or ``{"lot_id": ["LOT_001"]}``.
                When provided, the response includes both ``metrics`` (overall)
                and ``filtered_metrics`` (filtered subset), plus
                ``filtered_row_count``.

        Returns a dict with model_id, dataset_id, evaluation row count,
        aggregate metrics, optionally filtered metrics, and a sample of
        prediction-vs-actual rows.
        """
        mmeta = self.model_registry.get(model_id)
        if mmeta is None:
            raise ValueError(f"model {model_id!r} not found")
        dmeta = self.dataset_registry.get(mmeta.dataset_id)
        if dmeta is None:
            raise ValueError(f"dataset {mmeta.dataset_id!r} not registered")
        if not dmeta.partitioned or not dmeta.eval_parquet_path:
            raise ValueError(
                f"dataset {mmeta.dataset_id!r} has no evaluation partition; "
                "re-register to create train/eval splits"
            )
        config = self.get_config(mmeta.dataset_id)
        return evaluate_on_eval_set(
            model_id=model_id,
            model_registry=self.model_registry,
            config=config,
            eval_parquet_path=dmeta.eval_parquet_path,
            sample_size=sample_size,
            filters=filters,
        )

    def find_evaluation_slices(
        self,
        model_id: str,
        metric: str = "abs_error",
        dimensions: Optional[list[str]] = None,
        min_sample_size: int = 50,
        max_dimensions: int = 3,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Find slices where model error deviates most from the overall baseline.

        Runs inference on the held-out evaluation partition and searches
        categorical dimension combinations for populations where the
        prediction error differs materially from the overall average.

        Args:
            model_id: registered model to evaluate.
            metric: error metric to aggregate (``abs_error``, ``error``,
                ``pct_error`` for regression; ``log_loss`` for classification).
            dimensions: categorical columns to combine. If None, auto-detect
                from the dataset spec.
            min_sample_size: minimum rows per slice.
            max_dimensions: max dimensions to combine (1-3).
            top_k: number of top slices to return.

        Returns a list of slices sorted by absolute difference from the
        overall error baseline.
        """
        mmeta = self.model_registry.get(model_id)
        if mmeta is None:
            raise ValueError(f"model {model_id!r} not found")
        dmeta = self.dataset_registry.get(mmeta.dataset_id)
        if dmeta is None:
            raise ValueError(f"dataset {mmeta.dataset_id!r} not registered")
        if not dmeta.partitioned or not dmeta.eval_parquet_path:
            raise ValueError(
                f"dataset {mmeta.dataset_id!r} has no evaluation partition; "
                "re-register to create train/eval splits"
            )
        config = self.get_config(mmeta.dataset_id)
        return find_evaluation_slices(
            model_id=model_id,
            model_registry=self.model_registry,
            config=config,
            eval_parquet_path=dmeta.eval_parquet_path,
            metric=metric,
            dimensions=dimensions,
            min_sample_size=min_sample_size,
            max_dimensions=max_dimensions,
            top_k=top_k,
        )


# Module-level singleton for convenience.
_services: Optional[AnistrophServices] = None


def get_services() -> AnistrophServices:
    global _services
    if _services is None:
        _services = AnistrophServices()
    return _services
