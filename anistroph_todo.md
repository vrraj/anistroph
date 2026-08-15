# Anistroph v0.1 — Implementation Plan

Living checklist for the Anistroph predictive-analytics platform. Items are marked
`[x]` when complete. Built from the v0.1 specification and the repository setup
instructions.

Repository root: `/Users/raj/Documents/Raj/Ainstroph`

---

## 0. Project Setup

- [x] 0.1 Create full directory structure from spec §5
- [x] 0.2 Initialize git repository
- [x] 0.3 Write `pyproject.toml` (FastAPI, Uvicorn, Polars, DuckDB, pyarrow,
      scikit-learn, xgboost, joblib, pydantic, pyyaml, mcp, pytest, httpx)
- [x] 0.4 Write `.gitignore` (`.venv`, data artifacts, model artifacts, caches,
      IDE files, local env/config)
- [x] 0.5 Create Python `.venv`, install package editable, confirm imports

## 1. Core Specifications (domain-agnostic)

- [x] 1.1 `backend/datasets/spec.py` — `DatasetSpec` Pydantic model
      (dataset_id, name, entity_key, time_key, columns w/ type+role)
- [x] 1.2 `backend/features/spec.py` — `FeatureSpec` (per-column transforms,
      windows, categorical handling)
- [x] 1.3 `backend/targets/spec.py` — `TargetSpec` (binary, regression,
      future_event w/ horizon)
- [x] 1.4 YAML loading helpers for all specs (incl. `config.py` bundle loader)

## 2. Predictive-Maintenance Reference Configuration

- [x] 2.1 `datasets/predictive_maintenance/dataset.yaml` (DatasetSpec +
      FeatureSpec + TargetSpec + split config)

## 3. Synthetic Data Generation

- [x] 3.1 `scripts/generate_sensor_data.py` — 50 machines, 60 days, 5-min
      observations, learnable deterioration patterns, realistic failure
      imbalance, machine/type differences, anomalous readings
- [x] 3.2 Output CSV to `data/synthetic/` and Parquet to `data/raw/`

## 4. Dataset Layer

- [x] 4.1 `backend/datasets/loader.py` — CSV/Parquet ingestion → Parquet
- [x] 4.2 `backend/datasets/validation.py` — required columns, types,
      entity/time keys
- [x] 4.3 `backend/datasets/profiling.py` — generic profiling from DatasetSpec
- [x] 4.4 `backend/datasets/registry.py` — lightweight local metadata
      persistence (JSON), exposes all required registry fields

## 5. Feature Engine (single engine for train + inference)

- [x] 5.1 `backend/features/engine.py` — registry-driven transform dispatch,
      leakage-safe (no future observations), shared by training & inference
- [x] 5.2 `backend/features/numeric.py` — current, mean, min, max, std,
      median, delta, slope
- [x] 5.3 `backend/features/categorical.py` — one-hot, unknown-category handling
- [x] 5.4 `backend/features/temporal.py` — hour_of_day, day_of_week,
      elapsed_time
- [x] 5.5 `backend/features/rolling.py` — rolling window aggregations

## 6. Target Engine

- [x] 6.1 `backend/targets/binary.py` — binary target construction
- [x] 6.2 `backend/targets/regression.py` — regression target (architectural)
- [x] 6.3 `backend/targets/horizon.py` — future_event horizon labeling,
      entity-isolated, no leakage

## 7. ML Layer

- [x] 7.1 `backend/ml/base.py` — `Predictor` abstract contract
- [x] 7.2 `backend/models/logistic.py` — LogisticRegression predictor
- [x] 7.3 `backend/models/xgboost.py` — XGBoost predictor
- [x] 7.4 `backend/ml/training.py` — `train_model(...)` pipeline
      (registry → spec → load → features → target → chronological split →
      preprocess → fit → evaluate → persist → register)
- [x] 7.5 `backend/ml/evaluation.py` — ROC-AUC, PR-AUC, precision, recall, F1,
      confusion matrix, configurable threshold
- [x] 7.6 `backend/ml/inference.py` — generic `predict(model_id, entity_id,
      timestamp, records)`; reconstructs features from history via same engine
- [x] 7.7 `backend/ml/explain.py` — feature importance, structured
      top-drivers output
- [x] 7.8 `backend/ml/registry.py` — model artifact persistence
      (model.joblib, metadata.json, feature_spec.json, target_spec.json,
      metrics.json, imputer.joblib); abstracted storage

## 8. Analytical Engine (independent of ML)

- [x] 8.1 `backend/analysis/slice.py` — slice/filter/group/aggregate via
      DuckDB/Polars
- [x] 8.2 `backend/analysis/aggregate.py`
- [x] 8.3 `backend/analysis/compare.py`

## 9. REST API (FastAPI)

- [x] 9.1 `backend/main.py` — app factory, router mounting, static UI mount
- [x] 9.2 `backend/api/datasets.py` — POST /datasets, GET /datasets,
      GET /datasets/{id}, GET /datasets/{id}/profile
- [x] 9.3 `backend/api/analysis.py` — POST /analysis/slice, /analysis/compare
- [x] 9.4 `backend/api/models.py` — POST /models/train, GET /models,
      GET /models/{id}, GET /models/{id}/metrics
- [x] 9.5 `backend/api/predictions.py` — POST /predictions, /predictions/batch,
      /predictions/explain
- [x] 9.6 `GET /health`

## 10. MCP Server

- [x] 10.1 `backend/integrations/mcp/tools.py` — tool definitions calling the
      same core services
- [x] 10.2 `backend/integrations/mcp/server.py` — MCP server entrypoint
- [x] 10.3 Tools: list_datasets, profile_dataset, slice_data, compare_data,
      list_models, get_model_metrics, predict, explain_prediction

## 11. Web UI (lightweight)

- [x] 11.1 `frontend/` static UI — Dataset, Analysis, Training, Model,
      Prediction workspaces; adapts to DatasetSpec

## 12. Tests

- [x] 12.1 Unit: DatasetSpec parsing, validation, ingestion, profiling
- [x] 12.2 Unit: every feature transform + leakage assertions
- [x] 12.3 Unit: target construction (binary, future_event, horizon
      boundaries, entity isolation)
- [x] 12.4 Unit: ML training, evaluation, persistence, reload, prediction,
      train/inference feature parity
- [x] 12.5 Integration: REST API (register, profile, slice, train, metrics,
      predict, explain)
- [x] 12.6 Integration: MCP (tool discovery, schemas, profiling, slice,
      prediction, explanation, invalid inputs)
- [x] 12.7 End-to-end acceptance test (spec §30): generate → register →
      ingest → profile → features → target → split → train LR + XGB →
      evaluate → persist → reload → predict → explain → REST → MCP
- [x] 12.8 Synthetic data acceptance: models perform meaningfully above random

## 13. Documentation

- [x] 13.1 `README.md` (what it is, architecture, dataset abstraction, PM
      reference, install, data gen, registration, training, evaluation,
      inference, REST, MCP, tests)
- [x] 13.2 `TECHNICAL_ARCHITECTURE.md` (concise reference style)
- [x] 13.3 `docs/architecture.md`
- [x] 13.4 `docker-compose.yml`

## 14. Verification

- [x] 14.1 Clean venv → `pip install -e .` → `pytest` all green (66 passed)
- [x] 14.2 `uvicorn backend.main:app --reload` starts successfully
- [x] 14.3 Reference workflow runs end-to-end
- [x] 14.4 REST and MCP invoke the same core services (verified by test)
- [x] 14.5 No changes made outside the repository root
