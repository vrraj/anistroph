# Anistroph — Testing Guide

How to test Anistroph from automated tests to manual exploration of the REST
API, web UI, and MCP server.

---

## Prerequisites

```bash
cd /Users/raj/Documents/Raj/Ainstroph
source .venv/bin/activate
```

> **macOS note:** XGBoost requires `libomp`. If you see an XGBoost library
> loading error, run `brew install libomp`.

---

## 1. Run the Automated Test Suite

### Full suite

```bash
pytest
```

Expected: **66 passed**.

### Verbose output with test names

```bash
pytest -v
```

### Unit tests only

```bash
pytest tests/unit/ -v
```

Covers:
- DatasetSpec parsing, validation, ingestion, profiling
- Every feature transform + leakage assertions
- Target construction (binary, future_event, horizon boundaries, entity isolation)
- ML training, evaluation, persistence, reload, prediction, train/inference feature parity

### Integration tests only

```bash
pytest tests/integration/ -v
```

Covers:
- REST API (register, profile, slice, compare, train, metrics, predict, explain)
- MCP tools (discovery, schemas, all tool calls, invalid inputs)
- End-to-end acceptance (generate → register → ingest → profile → features → target → split → train LR + XGB → evaluate → persist → reload → predict → explain → REST → MCP parity)

### Single test file

```bash
pytest tests/unit/test_features.py -v
pytest tests/integration/test_e2e.py -v
```

### Stop on first failure with full traceback

```bash
pytest -x --tb=long
```

---

## 2. Start the Web UI + REST API

```bash
uvicorn backend.main:app --reload
```

Then open **http://127.0.0.1:9500** in your browser.

Verify health:
```bash
curl http://127.0.0.1:9500/health
# {"status":"ok","version":"0.1.0"}
```

---

## 3. Manual End-to-End Workflow (Python)

```bash
# Step 1: Generate synthetic data (50 machines, 60 days, 5-min intervals)
python scripts/generate_sensor_data.py

# Step 2: Register, train, predict, explain
python -c "
from backend.services import get_services
svc = get_services()

# Register dataset
svc.register_dataset_from_config(
    'datasets/predictive_maintenance/dataset.yaml',
    'data/synthetic/predictive_maintenance.csv',
)

# Profile
prof = svc.profile('predictive_maintenance')
print('Entities:', prof['entity_count'])
print('Failures:', prof['event_distribution']['failure'])

# Train XGBoost
result = svc.train('predictive_maintenance', 'failure_within_horizon', 'xgboost')
print('Model ID:', result['model_id'])
print('ROC-AUC:', result['metrics']['roc_auc'])
print('PR-AUC:', result['metrics']['pr_auc'])
print('Recall:', result['metrics']['recall'])
print('Precision:', result['metrics']['precision'])

# Predict failure probability
pred = svc.predict(result['model_id'], entity_id='TOOL_000', timestamp='2026-07-15T12:00:00')
print('Probability:', pred['probability'])
print('Prediction:', pred['prediction'])

# Explain prediction
expl = svc.explain(result['model_id'], entity_id='TOOL_000', timestamp='2026-07-15T12:00:00', top_k=5)
print('Top drivers:')
for d in expl['top_drivers']:
    print(f'  {d[\"feature\"]}: {d[\"impact\"]:.1%}')

# Slice data (analytical, independent of ML)
sl = svc.slice('predictive_maintenance', ['machine_type'], 'failure', 'mean')
print('Failure rate by machine type:', sl)
"
```

---

## 4. Test via REST API (curl)

With uvicorn running on http://127.0.0.1:9500:

```bash
# Health check
curl http://127.0.0.1:9500/health

# Register dataset
curl -X POST http://127.0.0.1:9500/datasets \
  -H "Content-Type: application/json" \
  -d '{"config_path": "datasets/predictive_maintenance/dataset.yaml", "source_path": "data/synthetic/predictive_maintenance.csv"}'

# List datasets
curl http://127.0.0.1:9500/datasets

# Profile dataset
curl http://127.0.0.1:9500/datasets/predictive_maintenance/profile

# List available model types
curl http://127.0.0.1:9500/models/types

# Train XGBoost
curl -X POST http://127.0.0.1:9500/models/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "predictive_maintenance", "target_name": "failure_within_horizon", "model_type": "xgboost"}'

# List models (copy a model_id from the output)
curl http://127.0.0.1:9500/models

# Get model metadata
curl http://127.0.0.1:9500/models/<MODEL_ID>

# Get model metrics
curl http://127.0.0.1:9500/models/<MODEL_ID>/metrics

# Slice data
curl -X POST http://127.0.0.1:9500/analysis/slice \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "predictive_maintenance", "dimensions": ["machine_type"], "metric": "failure", "aggregation": "mean"}'

# Compare data
curl -X POST http://127.0.0.1:9500/analysis/compare \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "predictive_maintenance", "dimension": "machine_type", "metric": "failure", "aggregation": "mean"}'

# Predict
curl -X POST http://127.0.0.1:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{"model_id": "<MODEL_ID>", "entity_id": "TOOL_000", "timestamp": "2026-07-15T12:00:00"}'

# Batch predict
curl -X POST http://127.0.0.1:9500/predictions/batch \
  -H "Content-Type: application/json" \
  -d '[{"model_id": "<MODEL_ID>", "entity_id": "TOOL_000", "timestamp": "2026-07-15T12:00:00"},
       {"model_id": "<MODEL_ID>", "entity_id": "TOOL_001", "timestamp": "2026-07-15T12:00:00"}]'

# Explain prediction
curl -X POST http://127.0.0.1:9500/predictions/explain \
  -H "Content-Type: application/json" \
  -d '{"model_id": "<MODEL_ID>", "entity_id": "TOOL_000", "timestamp": "2026-07-15T12:00:00", "top_k": 5}'
```

> Replace `<MODEL_ID>` with an actual model ID from the list models output.

---

## 5. Test via the Browser UI

With uvicorn running, open **http://127.0.0.1:9500**:

### Datasets tab
1. Click **Register** (fields are pre-filled with the predictive-maintenance config)
2. Click **Profile** to see row counts, column distributions, event distribution

### Analysis tab
1. Click **Slice** to see failure rate aggregated by machine type
2. Click **Compare** to compare failure rates across machine types

### Training tab
1. Select `xgboost` or `logistic_regression`
2. Click **Train** — results show model ID and metrics

### Models tab
1. Click **Refresh** to list all trained models
2. Paste a model ID and click **Get Details** to see metadata, features, metrics

### Prediction tab
1. Paste a model ID
2. Click **Predict** to see failure probability and risk level
3. Click **Explain** to see top contributing features

---

## 6. Test the MCP Server

### Run the server

```bash
python -m backend.integrations.mcp.server
```

This runs an MCP server over stdio. Connect from any MCP-compatible client.

### Available MCP tools

| Tool | Description |
|------|-------------|
| `anistroph_list_datasets` | List all registered datasets |
| `anistroph_profile_dataset` | Profile a dataset by ID |
| `anistroph_slice_data` | Slice data by dimensions with aggregation |
| `anistroph_compare_data` | Compare a metric across dimension values |
| `anistroph_list_models` | List all trained models |
| `anistroph_get_model_metrics` | Get evaluation metrics for a model |
| `anistroph_predict` | Make a prediction (entity_id + timestamp or records) |
| `anistroph_explain_prediction` | Explain a prediction with top drivers |

### Connect from Claude Desktop

Add this to your Claude Desktop MCP config:

```json
{
  "mcpServers": {
    "anistroph": {
      "command": "python",
      "args": ["-m", "backend.integrations.mcp.server"],
      "cwd": "/Users/raj/Documents/Raj/Ainstroph"
    }
  }
}
```

Then ask Claude things like:
- "List all Anistroph datasets"
- "Profile the predictive_maintenance dataset"
- "What models are available in Anistroph?"
- "Predict failure probability for TOOL_000 at 2026-07-15T12:00:00 using model \<id\>"

---

## 7. Generate the Full-Size Dataset

Default: 50 machines × 60 days × 5-min intervals (~864K rows):

```bash
python scripts/generate_sensor_data.py
```

Custom parameters:

```bash
python scripts/generate_sensor_data.py --machines 50 --days 60 --interval 5 --seed 42
```

Output:
- CSV: `data/synthetic/predictive_maintenance.csv`
- Parquet: `data/raw/predictive_maintenance.parquet`

---

## 8. Verify REST and MCP Produce Identical Results

The end-to-end test verifies this automatically:

```bash
pytest tests/integration/test_e2e.py::TestEndToEnd::test_rest_and_mcp_same_services -v
```

This test trains a model, predicts through REST, predicts the same input
through MCP, and asserts the probabilities are identical (both call the same
core services).

---

## 9. Verify No Temporal Leakage

The feature leakage test verifies that a feature at time T never uses
observations after T:

```bash
pytest tests/unit/test_features.py::TestFeatureEngine::test_rolling_mean_leakage_safe -v
```

The target isolation test verifies that a failure on one entity never labels
another:

```bash
pytest tests/unit/test_targets.py::TestFutureEventTarget::test_entity_isolation -v
```

---

## 10. Test Adding a New Dataset (Extensibility)

To verify the architecture is domain-agnostic, create a new dataset config
and train without changing any core code:

```bash
python -c "
import yaml, tempfile, os
from backend.datasets.config import load_dataset_config
from backend.features.engine import FeatureEngine
import polars as pl

# Create a generic 'widget_quality' dataset config
config = {
    'dataset': {
        'dataset_id': 'widget_quality',
        'name': 'Widget Quality Demo',
        'entity_key': 'widget_id',
        'time_key': 'ts',
        'columns': {
            'ts': {'type': 'timestamp', 'role': 'identifier'},
            'widget_id': {'type': 'categorical', 'role': 'identifier'},
            'metric_x': {'type': 'numeric', 'role': 'feature'},
            'metric_y': {'type': 'numeric', 'role': 'feature'},
            'category': {'type': 'categorical', 'role': 'feature'},
            'defect': {'type': 'boolean', 'role': 'event'},
        },
    },
    'features': {
        'metric_x': {'column': 'metric_x', 'transforms': ['current', {'mean': {'windows': ['1h']}}]},
        'metric_y': {'column': 'metric_y', 'transforms': ['current']},
        'category': {'column': 'category', 'transforms': ['categorical']},
    },
    'target': {
        'name': 'defect_flag',
        'type': 'binary',
        'source_column': 'defect',
        'positive_class': 1,
    },
}

path = tempfile.mktemp(suffix='.yaml')
with open(path, 'w') as f:
    yaml.dump(config, f)

cfg = load_dataset_config(path)
print('Dataset:', cfg.dataset_spec.dataset_id)
print('Features:', list(cfg.feature_spec.features.keys()))
print('Target:', cfg.target_spec.name, cfg.target_spec.type)

# Build features on sample data
from datetime import datetime, timedelta
df = pl.DataFrame({
    'ts': [datetime(2026, 1, 1, i) for i in range(20)],
    'widget_id': ['W1']*20,
    'metric_x': [float(i) for i in range(20)],
    'metric_y': [float(i*2) for i in range(20)],
    'category': ['A']*10 + ['B']*10,
    'defect': [0]*18 + [1, 1],
})
engine = FeatureEngine()
feat_df, meta = engine.build_features(df, cfg.dataset_spec, cfg.feature_spec, fit=True)
print('Feature columns:', meta.feature_names)
print('No domain assumptions needed — architecture is generic!')
os.unlink(path)
"
```

---

## Quick Reference

| What | Command |
|------|---------|
| Full test suite | `pytest` |
| Unit tests | `pytest tests/unit/ -v` |
| Integration tests | `pytest tests/integration/ -v` |
| End-to-end test | `pytest tests/integration/test_e2e.py -v` |
| Start API + UI | `uvicorn backend.main:app --reload` |
| Generate data | `python scripts/generate_sensor_data.py` |
| MCP server | `python -m backend.integrations.mcp.server` |
| API docs (Swagger) | http://127.0.0.1:9500/docs |
| API docs (ReDoc) | http://127.0.0.1:9500/redoc |
| Web UI | http://127.0.0.1:9500 |
