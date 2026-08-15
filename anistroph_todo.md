# Anistroph v0.1 — Implementation Plan

Living checklist for the Anistroph predictive-analytics platform. Items are marked
`[x]` when complete. Built from the v0.1 specification and the repository setup
instructions.

Repository root: `/Users/raj/Documents/Raj/anistroph`

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

---

## 15. Semiconductor Yield Dataset (v0.1 Reduced Scope)

### 15.1 Synthetic Data Generation
- [x] 15.1.1 `scripts/generate_semiconductor_yield_data.py` — ~30,000 wafer rows,
      flat Parquet output to `data/semiconductor_yield/data.parquet`
- [x] 15.1.2 Inject learnable yield effects: ETCH_02, CH_B, high temp_std,
      interaction effects, product×recipe, deposition_tool×pressure_std,
      maintenance_age×tool, process_route×temp variability
- [x] 15.1.3 wafer_yield between 0.0 and 1.0, baseline ~96-98%
- [x] 15.1.4 Configurable size for tests (--wafers, --seed)

### 15.2 Dataset Configuration
- [x] 15.2.1 `datasets/semiconductor_yield/dataset.yaml` — DatasetSpec +
      FeatureSpec + TargetSpec (regression target: wafer_yield)
- [x] 15.2.2 Columns: timestamp, lot_id, wafer_id, product_id, fab_id,
      process_route, etch_tool, etch_chamber, etch_recipe, deposition_tool,
      deposition_chamber, deposition_recipe, etch_temperature_mean/std,
      etch_pressure_mean/std, etch_gas_flow_mean, etch_rf_power_mean,
      etch_process_time, deposition_temperature_mean/std,
      deposition_pressure_mean/std, deposition_process_time, exposure_dose,
      focus_offset, maintenance_age_etch, maintenance_age_deposition,
      wafer_yield

### 15.3 Model Adapters (separate folders, shared inference)
- [x] 15.3.1 `backend/models/xgboost_regressor.py` — XGBRegressor adapter
      (implements Predictor contract, returns regression values)
- [x] 15.3.2 `backend/models/linear_regression.py` — LinearRegression/
      ElasticNet baseline adapter
- [x] 15.3.3 Register new model types in MODEL_FACTORIES and _load_predictor

### 15.4 Regression Evaluation
- [x] 15.4.1 `backend/ml/evaluation.py` — add `evaluate_regression()`
      with MAE, RMSE, R², mean prediction error, median absolute error,
      95th percentile absolute error
- [x] 15.4.2 Add baseline comparison (vs. mean-yield predictor)

### 15.5 Training Pipeline Extension
- [x] 15.5.1 `backend/ml/training.py` — branch on target type:
      classification → evaluate_binary + threshold; regression →
      evaluate_regression, no threshold
- [x] 15.5.2 `scripts/train_model.py` — admin CLI:
      `python scripts/train_model.py --dataset semiconductor_yield --model-type xgboost_regressor`

### 15.6 Inference Extension (one shared path)
- [x] 15.6.1 `backend/ml/inference.py` — support non-temporal entity_id
      lookup: load parquet, filter by entity_key, build features, predict
- [x] 15.6.2 Branch on target type: regression → return predicted_yield +
      actual_yield; classification → return probability + prediction
- [x] 15.6.3 `backend/ml/explain.py` — same regression/classification branch

### 15.7 Analysis: find_interesting_slices
- [x] 15.7.1 `backend/analysis/interesting.py` — search 1/2/3-dim
      combinations, min sample size 100, rank by difference from baseline
- [x] 15.7.2 Return: dimension values, row count, mean, median, std,
      difference from overall baseline

### 15.8 MCP Tools
- [x] 15.8.1 Add `anistroph_find_interesting_slices` tool
- [x] 15.8.2 Update tools.py with new tool definitions (9 tools total)

### 15.9 Tests
- [x] 15.9.1 Semiconductor data generation tests (~30K rows, yield 0-1,
      hidden interactions present)
- [x] 15.9.2 Dataset isolation (semiconductor doesn't affect PM — verified
      by running full suite)
- [x] 15.9.3 Regression model training test (XGBRegressor trains,
      MAE/RMSE/R² produced)
- [x] 15.9.4 Chronological split test for semiconductor
- [x] 15.9.5 Model persist/reload test
- [x] 15.9.6 Prediction doesn't retrain (inference loads model, never trains)
- [x] 15.9.7 slice_data works on semiconductor dataset (via shared services)
- [x] 15.9.8 find_interesting_slices works (dedicated tests)
- [x] 15.9.9 Existing PM tests still pass (66 original + 26 new = 92 total)
- [x] 15.9.10 MCP stdio starts with new tools (tools.py updated)

### 15.10 Documentation
- [x] 15.10.1 Update README_SETUP_USAGE.md with semiconductor model details
- [x] 15.10.2 Update anistroph_todo.md (this file)

### 15.11 Verification
- [x] 15.11.1 Full test suite passes (92 tests)
- [x] 15.11.2 XGBoost R²=0.816, beats baseline (MAE 0.0065 vs 0.0139)
- [x] 15.11.3 Linear baseline R²=0.610 (XGBoost captures nonlinear interactions)
- [x] 15.11.4 find_interesting_slices identifies ETCH_02+CH_B as worst combo
- [x] 15.11.5 PM model (anistroph-sentinel-v1) still works unchanged
