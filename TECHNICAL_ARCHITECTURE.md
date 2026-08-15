# Anistroph v0.1 — Technical Architecture

## 1. Design Principle

Domain-specific concepts are confined to configuration (`DatasetSpec`,
`FeatureSpec`, `TargetSpec`). The core pipeline is generic.

## 2. Core Components

### 2.1 DatasetSpec (`backend/datasets/spec.py`)

Pydantic model. Declares `dataset_id`, `entity_key`, `time_key`, and a
column map with `type` (numeric, categorical, boolean, timestamp, string)
and `role` (identifier, feature, event, target, metadata, ignore).

### 2.2 FeatureSpec (`backend/features/spec.py`)

Declares per-column transforms: `current`, `mean`, `min`, `max`, `std`,
`median`, `slope`, `delta`, `categorical`, `hour_of_day`, `day_of_week`,
`elapsed_time`. Windowed transforms take a list of duration strings
(`1h`, `6h`).

### 2.3 TargetSpec (`backend/targets/spec.py`)

Three target types: `binary`, `regression`, `future_event`.
`future_event` requires a `horizon` (e.g. `24h`).

### 2.4 Feature Engine (`backend/features/engine.py`)

Single engine used by both training and inference. Interprets `FeatureSpec`
generically. Leakage-safe: features at time T never use observations after T.
Categorical encodings are fitted during training and stored in
`FeatureMetadata` so inference applies the identical encoding.

### 2.5 Target Engine (`backend/targets/engine.py`)

Dispatches target construction by type. `future_event` labels are
entity-isolated: a failure on entity B never labels entity A.

### 2.6 ML Engine

- `backend/ml/base.py` — `Predictor` abstract contract (`fit`, `predict`,
  `predict_proba`, `save`, `load`, `feature_importance`).
- `backend/models/logistic.py` — LogisticRegression with StandardScaler.
- `backend/models/xgboost.py` — XGBoost classifier.
- `backend/ml/training.py` — `train_model()` pipeline: registry → spec →
  load → features → target → chronological split → impute → fit → evaluate
  → persist → register.
- `backend/ml/evaluation.py` — ROC-AUC, PR-AUC, precision, recall, F1,
  confusion matrix. Configurable threshold (optimized for F1 on validation).
- `backend/ml/inference.py` — `predict(model_id, entity_id, timestamp,
  records)`. Reconstructs features from history via the same engine.
- `backend/ml/explain.py` — Feature importance weighted by instance values.
- `backend/ml/registry.py` — Filesystem-backed model artifact store.

### 2.7 Analytical Engine (`backend/analysis/slice.py`)

Deterministic operations (slice, aggregate, compare) via Polars. Independent
of ML training.

### 2.8 Services (`backend/services.py`)

`AnistrophServices` — single service container used by REST, MCP, and UI.

## 3. Interfaces

### 3.1 REST (`backend/main.py`, `backend/api/`)

FastAPI routers for datasets, analysis, models, predictions. All invoke
`AnistrophServices`.

### 3.2 MCP (`backend/integrations/mcp/`)

Low-level `mcp.server.Server` over stdio. Tools call `AnistrophServices`.
No arbitrary Python execution. Training not exposed.

### 3.3 UI (`frontend/index.html`)

Single-page app with Dataset, Analysis, Training, Model, and Prediction
workspaces. Adapts to DatasetSpec.

## 4. Data Flow

```
generate → register → ingest (CSV→Parquet) → validate → profile →
build features (fit) → construct target → chronological split →
impute → train → evaluate → persist (model + specs + metadata) →
register model → reload → predict (reconstruct features) → explain
```

## 5. Leakage Prevention

- Rolling windows: per-entity, trailing, `closed="right"` (includes current
  row, no future rows).
- Target: future-event labels look forward only; features never do.
- Categorical fit: categories learned from training data only.
- Split: chronological for temporal datasets (no random row shuffling).

## 6. Extensibility

Adding a dataset requires only: data file + DatasetSpec + FeatureSpec +
TargetSpec. The core pipeline is unchanged.

## 7. Artifact Layout

```
artifacts/models/<model_id>/
    model.joblib
    imputer.joblib
    metadata.json
    feature_spec.json
    feature_metadata.json
    target_spec.json
    metrics.json
```

## 8. Registries

- **DatasetRegistry** (`artifacts/dataset_registry.json`) — lightweight JSON
  metadata, independent of Parquet files.
- **ModelRegistry** (`artifacts/models/model_index.json`) — model metadata
  index. Both are abstracted for future replacement.
