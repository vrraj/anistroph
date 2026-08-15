# Anistroph v0.1

**Extensible, domain-agnostic predictive analytics platform.**

Anistroph is a Python prototype of a predictive analytics platform for
structured datasets. The first reference implementation uses
semiconductor/manufacturing equipment sensor data for **predictive
maintenance**, but the architecture contains no predictive-maintenance-specific
assumptions in the core pipeline.

## What Anistroph Is

Anistroph provides a complete lifecycle:

```
Dataset → DatasetSpec → Validation/Profiling → Feature Engineering →
Target Construction → Training → Evaluation → Persisted Model →
Inference → Analysis/Explanation
```

Anistroph is **not** an AutoML platform and does not train foundation models.

## Architecture

```
                         Web UI
                            │
                         FastAPI
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
     Dataset API        Analysis API         Model API
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ↓
                    ANISTROPH CORE
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
 Dataset Registry      Feature Engine       ML Engine
        │                   │                    │
 DatasetSpec           FeatureSpec            Models
        │                   │                    │
        └──────────────┬────┴─────────────┬─────┘
                       │                  │
                   DuckDB/Polars     Model Registry
                       │                  │
                    Parquet          Model Artifacts

                            ▲
                            │
                       MCP Server
                            │
                    External AI/Agent
```

All interfaces (REST, MCP, UI) invoke the same core Python services.

## Dataset Abstraction

Dataset-specific concepts never enter the generic ML pipeline. Instead:

```
             Dataset
                ↓
          DatasetSpec
                ↓
       ┌────────┼────────┐
       ↓        ↓        ↓
   FeatureSpec TargetSpec ModelSpec
       │        │        │
       └────────┼────────┘
                ↓
        Generic Pipeline
```

- **DatasetSpec** (`backend/datasets/spec.py`) — describes columns, types,
  roles, entity/time keys.
- **FeatureSpec** (`backend/features/spec.py`) — declares transforms per
  column (current, mean, std, slope, rolling windows, categorical encoding).
- **TargetSpec** (`backend/targets/spec.py`) — declares target type (binary,
  regression, future_event) and horizon.

Predictive maintenance is a **reference configuration**, not an architectural
dependency.

## Predictive-Maintenance Reference Implementation

The reference dataset (`datasets/predictive_maintenance/dataset.yaml`) defines:

- 50 machines, 60 days, 5-minute observations
- Sensors: temperature, vibration, pressure, current, voltage, rpm, flow_rate
- Maintenance age, operating hours
- Failure event with failure type
- 24-hour future-failure target (`failure_within_horizon`)
- Learnable deterioration patterns (vibration drift, temperature drift,
  pressure instability, maintenance age → increased failure probability)

## Installation

```bash
cd anistroph

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

> **macOS note:** XGBoost requires `libomp`. Install it with
> `brew install libomp` if you see an XGBoost library loading error.

## Synthetic Data Generation

```bash
python scripts/generate_sensor_data.py
```

Generates `data/synthetic/predictive_maintenance.csv` and
`data/raw/predictive_maintenance.parquet`.

Options:
```bash
python scripts/generate_sensor_data.py --machines 50 --days 60 --interval 5 --seed 42
```

## Dataset Registration

```python
from backend.services import get_services

svc = get_services()
meta = svc.register_dataset_from_config(
    "datasets/predictive_maintenance/dataset.yaml",
    "data/synthetic/predictive_maintenance.csv",
)
```

Or via REST:
```bash
curl -X POST http://localhost:8000/datasets \
  -H "Content-Type: application/json" \
  -d '{"config_path": "datasets/predictive_maintenance/dataset.yaml", "source_path": "data/synthetic/predictive_maintenance.csv"}'
```

## Training

```python
result = svc.train(
    dataset_id="predictive_maintenance",
    target_name="failure_within_horizon",
    model_type="xgboost",
)
```

Available model types: `xgboost`, `logistic_regression`.

## Evaluation

Models are evaluated with ROC-AUC, PR-AUC, precision, recall, F1, and
confusion matrix. Decision thresholds are optimized for F1 on the validation
set. PR-AUC and recall are emphasized for rare-event problems.

```python
metrics = svc.get_model_metrics("predictive-maintenance-xgb-...")
```

## Inference

The caller does not construct features. For temporal datasets, provide
entity_id and timestamp:

```python
pred = svc.predict(
    model_id="...",
    entity_id="TOOL_000",
    timestamp="2026-07-15T12:00:00",
)
```

Anistroph retrieves historical observations, builds features using the same
Feature Engine + persisted FeatureMetadata, and predicts.

## Explainability

```python
expl = svc.explain(model_id="...", entity_id="TOOL_000",
                   timestamp="2026-07-15T12:00:00", top_k=10)
```

Returns structured top drivers (feature importance weighted by instance
values). No LLM generates or fabricates drivers.

## REST API

```bash
uvicorn backend.main:app --reload
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/datasets` | Register dataset |
| GET | `/datasets` | List datasets |
| GET | `/datasets/{id}` | Get dataset metadata |
| GET | `/datasets/{id}/profile` | Profile dataset |
| POST | `/analysis/slice` | Slice data |
| POST | `/analysis/compare` | Compare data |
| POST | `/models/train` | Train model |
| GET | `/models` | List models |
| GET | `/models/types` | List model types |
| GET | `/models/{id}` | Get model metadata |
| GET | `/models/{id}/metrics` | Get model metrics |
| POST | `/predictions` | Predict |
| POST | `/predictions/batch` | Batch predict |
| POST | `/predictions/explain` | Explain prediction |

## MCP

Anistroph exposes deterministic capabilities through MCP:

```
anistroph_list_datasets
anistroph_profile_dataset
anistroph_slice_data
anistroph_compare_data
anistroph_list_models
anistroph_get_model_metrics
anistroph_predict
anistroph_explain_prediction
```

Run the MCP server:
```bash
python -m backend.integrations.mcp.server
```

MCP tools call the same core services as REST. No arbitrary Python execution
is exposed. Model training is not exposed through MCP.

## Tests

```bash
pytest
```

The suite includes:
- **Unit tests:** DatasetSpec parsing, validation, ingestion, profiling,
  feature transforms (with leakage assertions), target construction (entity
  isolation, horizon boundaries), ML training/evaluation/persistence/reload,
  prediction, feature parity.
- **Integration tests:** REST API (all endpoints), MCP tools (discovery,
  schemas, all tool calls, invalid inputs).
- **End-to-end acceptance test:** generate → register → ingest → profile →
  features → target → split → train LR + XGB → evaluate → persist → reload →
  predict → explain → REST → MCP (verifying same results).

## Adding a New Dataset

Adding a fundamentally different dataset should not require rewriting
Anistroph's architecture. You need:

1. A data file (CSV or Parquet)
2. A `DatasetSpec` (columns, types, roles, entity/time keys)
3. A `FeatureSpec` (transforms per column)
4. A `TargetSpec` (target type, source column, horizon if applicable)

See `datasets/predictive_maintenance/dataset.yaml` for the reference format.

## Technology Stack

| Category | Technology |
|----------|-----------|
| Runtime | Python |
| Web framework | FastAPI |
| ASGI | Uvicorn |
| Dataframe engine | Polars |
| Analytical SQL | DuckDB |
| Storage format | Parquet |
| ML | scikit-learn, XGBoost |
| Model persistence | joblib |
| MCP | Python MCP SDK |
| Testing | pytest |

## License

MIT
