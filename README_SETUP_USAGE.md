# Anistroph — Setup & Usage Guide

Step-by-step instructions for setting up Anistroph, generating data,
registering datasets, training models, making predictions, and querying
through REST, Python, MCP, and the web UI.

---

## 1. Installation

### Local (native)

```bash
cd /Users/raj/Documents/Raj/Anistroph

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

> **macOS note:** XGBoost requires `libomp`. If you see an XGBoost
> library loading error, run `brew install libomp`.

### Docker

```bash
make start          # builds image + starts container with hot reload
make stop           # stop containers
make rebuild        # rebuild after dependency changes
```

The app runs on **http://localhost:9500**.

---

## 2. Generate Synthetic Data

```bash
source .venv/bin/activate

# Default: 50 machines, 60 days, 5-min intervals (~864K rows)
python scripts/generate_sensor_data.py

# Custom parameters
python scripts/generate_sensor_data.py --machines 20 --days 30 --interval 10 --seed 42
```

Output:
- CSV: `data/synthetic/predictive_maintenance.csv`
- Parquet: `data/raw/predictive_maintenance.parquet`

---

## 3. Register a Dataset

Registration reads the dataset config (`datasets/predictive_maintenance/dataset.yaml`),
validates the data against the spec, converts CSV to Parquet, and stores
metadata in the dataset registry.

### Via Python

```python
from backend.services import get_services

svc = get_services()
meta = svc.register_dataset_from_config(
    "datasets/predictive_maintenance/dataset.yaml",
    "data/synthetic/predictive_maintenance.csv",
)
print(f"Registered: {meta.dataset_id}, {meta.row_count} rows")
```

### Via REST API

```bash
curl -X POST http://localhost:9500/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "datasets/predictive_maintenance/dataset.yaml",
    "source_path": "data/synthetic/predictive_maintenance.csv"
  }'
```

### Via web UI

1. Open http://localhost:9500
2. Go to the **Datasets** tab
3. Click **Register** (fields are pre-filled)
4. Click **Profile** to see dataset statistics

---

## 4. Profile a Dataset

Profiling generates statistics: row count, column count, types, missing
values, unique counts, numeric distributions, categorical distributions,
time range, entity count, and event distribution.

### Via Python

```python
from backend.services import get_services

prof = get_services().profile("predictive_maintenance")
print(f"Rows: {prof['row_count']}")
print(f"Entities: {prof['entity_count']}")
print(f"Failures: {prof['event_distribution']['failure']}")
```

### Via REST API

```bash
curl http://localhost:9500/datasets/predictive_maintenance/profile
```

### Via MCP (Claude Desktop)

> "Profile the predictive_maintenance dataset. What's the failure rate?"

---

## 5. Train a Model

Training builds features, constructs the target, splits chronologically,
fits the model, evaluates on held-out test data, persists the model
artifact, and registers it.

Available model types: `xgboost`, `logistic_regression`.

### Via Python

```python
from backend.services import get_services

svc = get_services()

# Train XGBoost with a custom model ID
result = svc.train(
    dataset_id="predictive_maintenance",
    target_name="failure_within_horizon",
    model_type="xgboost",
    model_id="pm-xgb",
)
print(f"Model ID: {result['model_id']}")
print(f"ROC-AUC: {result['metrics']['roc_auc']:.3f}")
print(f"PR-AUC: {result['metrics']['pr_auc']:.3f}")
print(f"Recall: {result['metrics']['recall']:.3f}")
print(f"Precision: {result['metrics']['precision']:.3f}")
print(f"F1: {result['metrics']['f1']:.3f}")
```

### Via REST API

```bash
curl -X POST http://localhost:9500/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "predictive_maintenance",
    "target_name": "failure_within_horizon",
    "model_type": "xgboost",
    "model_id": "pm-xgb"
  }'
```

### Via web UI

1. Open http://localhost:9500
2. Go to the **Training** tab
3. Select `xgboost` or `logistic_regression`
4. Click **Train**
5. Copy the model ID from the result

### Training is NOT available via MCP

Model training is intentionally not exposed through MCP. This is a
deliberate design decision — training is a heavyweight operation that
should be done explicitly via REST, Python, or the UI.

---

## 6. List Models

### Via Python

```python
from backend.services import get_services

models = get_services().list_models()
for m in models:
    print(f"{m.model_id}: {m.model_type}, dataset={m.dataset_id}")
```

### Via REST API

```bash
curl http://localhost:9500/models
```

### Via MCP (Claude Desktop)

> "What models are available in Anistroph?"

---

## 7. Get Model Metrics

### Via Python

```python
from backend.services import get_services

metrics = get_services().get_model_metrics("pm-xgb")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"PR-AUC: {metrics['pr_auc']:.3f}")
print(f"Confusion matrix: {metrics['confusion_matrix']}")
```

### Via REST API

```bash
curl http://localhost:9500/models/pm-xgb/metrics
```

### Via MCP (Claude Desktop)

> "Show me the metrics for model pm-xgb."

---

## 8. Predict

The caller provides a model ID, entity ID, and timestamp. Anistroph
retrieves historical observations, builds features using the same
Feature Engine + persisted metadata, and returns the prediction.

### Via Python

```python
from backend.services import get_services

pred = get_services().predict(
    model_id="pm-xgb",
    entity_id="TOOL_000",
    timestamp="2026-06-15T12:00:00",
)
print(f"Probability: {pred['probability']:.4f}")
print(f"Prediction: {pred['prediction']}")
```

### Via REST API

```bash
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "pm-xgb",
    "entity_id": "TOOL_000",
    "timestamp": "2026-06-15T12:00:00"
  }'
```

### Batch predict

```bash
curl -X POST http://localhost:9500/predictions/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"model_id": "pm-xgb", "entity_id": "TOOL_000", "timestamp": "2026-06-15T12:00:00"},
    {"model_id": "pm-xgb", "entity_id": "TOOL_001", "timestamp": "2026-06-15T12:00:00"}
  ]'
```

### Via MCP (Claude Desktop)

> "Predict the failure probability for TOOL_000 at 2026-06-15T12:00:00
> using model pm-xgb."

---

## 9. Explain a Prediction

Returns the top contributing features (model feature importance weighted
by instance values). Explanations are deterministic and model-derived —
no LLM fabrication.

### Via Python

```python
from backend.services import get_services

expl = get_services().explain(
    model_id="pm-xgb",
    entity_id="TOOL_000",
    timestamp="2026-06-15T12:00:00",
    top_k=10,
)
print(f"Probability: {expl['probability']:.4f}")
for d in expl["top_drivers"]:
    print(f"  {d['feature']}: {d['impact']:.1%}")
```

### Via REST API

```bash
curl -X POST http://localhost:9500/predictions/explain \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "pm-xgb",
    "entity_id": "TOOL_000",
    "timestamp": "2026-06-15T12:00:00",
    "top_k": 10
  }'
```

### Via MCP (Claude Desktop)

> "Explain why that prediction came out that way. What are the top drivers?"

---

## 10. Analytical Queries (Slice, Compare)

Analytical operations are independent of ML. They aggregate data by
dimensions without involving models.

### Slice

```python
from backend.services import get_services

result = get_services().slice(
    dataset_id="predictive_maintenance",
    dimensions=["machine_type"],
    metric="failure",
    aggregation="mean",
)
# [{"machine_type": "TYPE_A", "failure_mean": 0.0014}, ...]
```

### Compare

```python
from backend.services import get_services

result = get_services().compare(
    dataset_id="predictive_maintenance",
    dimension="machine_type",
    metric="vibration",
    aggregation="mean",
)
```

### Via REST API

```bash
# Slice
curl -X POST http://localhost:9500/analysis/slice \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "predictive_maintenance",
    "dimensions": ["machine_type"],
    "metric": "failure",
    "aggregation": "mean"
  }'

# Compare
curl -X POST http://localhost:9500/analysis/compare \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "predictive_maintenance",
    "dimension": "machine_type",
    "metric": "vibration",
    "aggregation": "mean"
  }'
```

### Via MCP (Claude Desktop)

> "Slice the predictive_maintenance dataset by machine_type and show the
> mean failure rate for each type."

---

## 11. Complete Workflow (Python)

```python
from backend.services import get_services

svc = get_services()

# 1. Register dataset
svc.register_dataset_from_config(
    "datasets/predictive_maintenance/dataset.yaml",
    "data/synthetic/predictive_maintenance.csv",
)

# 2. Profile
prof = svc.profile("predictive_maintenance")
print(f"Rows: {prof['row_count']}, Entities: {prof['entity_count']}")

# 3. Train
result = svc.train(
    "predictive_maintenance",
    "failure_within_horizon",
    "xgboost",
    model_id="pm-xgb",
)
print(f"ROC-AUC: {result['metrics']['roc_auc']:.3f}")

# 4. Predict
pred = svc.predict("pm-xgb", entity_id="TOOL_000", timestamp="2026-06-15T12:00:00")
print(f"Probability: {pred['probability']:.4f}")

# 5. Explain
expl = svc.explain("pm-xgb", entity_id="TOOL_000", timestamp="2026-06-15T12:00:00", top_k=5)
print(f"Top drivers: {[d['feature'] for d in expl['top_drivers']]}")

# 6. Analyze
sl = svc.slice("predictive_maintenance", ["machine_type"], "failure", "mean")
print(f"Failure rate by type: {sl}")
```

---

## 12. MCP Setup (Claude Desktop)

### Config file

Location: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "anistroph": {
      "command": "/Users/raj/Documents/Raj/Anistroph/.venv/bin/python",
      "args": ["-m", "backend.integrations.mcp.server"],
      "cwd": "/Users/raj/Documents/Raj/Anistroph"
    }
  }
}
```

> Use the absolute path to the venv Python so Claude Desktop picks up
> all installed dependencies.

### After saving the config

1. Fully quit Claude Desktop (`Cmd+Q`)
2. Reopen Claude Desktop
3. Start a new conversation
4. Verify Anistroph tools appear (hammer/tools icon)

### Available MCP tools

| Tool | Description |
|------|-------------|
| `anistroph_list_datasets` | List all registered datasets |
| `anistroph_profile_dataset` | Profile a dataset by ID |
| `anistroph_slice_data` | Slice data by dimensions with aggregation |
| `anistroph_compare_data` | Compare a metric across dimension values |
| `anistroph_list_models` | List all trained models |
| `anistroph_get_model_metrics` | Get evaluation metrics for a model |
| `anistroph_predict` | Make a prediction (entity_id + timestamp) |
| `anistroph_explain_prediction` | Explain a prediction with top drivers |

### Example Claude prompts

- "List all Anistroph datasets"
- "Profile the predictive_maintenance dataset"
- "Slice the data by machine_type and show mean failure rate"
- "What models are available?"
- "Show me the metrics for model pm-xgb"
- "Predict failure probability for TOOL_000 at 2026-06-15T12:00:00 using model pm-xgb"
- "Explain that prediction — what are the top drivers?"

---

## 13. Troubleshooting

### MCP tools not appearing in Claude Desktop

- Check config JSON is valid: `python3 -c "import json; json.load(open('~/Library/Application Support/Claude/claude_desktop_config.json'))"`
- Check logs: `tail -100 ~/Library/Logs/Claude/main.log | grep -i "mcp\|anistroph\|error"`
- Fully quit Claude Desktop (`Cmd+Q`), not just close the window
- Verify the venv Python path exists: `ls /Users/raj/Documents/Raj/Anistroph/.venv/bin/python`

### Predictions fail with "feature spec not found"

The model artifacts are missing. This happens if the model was trained
in a different environment (e.g., Docker) or artifacts were deleted.
Re-train the model:

```bash
python -c "from backend.services import get_services; get_services().train('predictive_maintenance', 'failure_within_horizon', 'xgboost', model_id='pm-xgb')"
```

### MCP returns empty lists

The MCP server process caches data at startup. Registries now reload
from disk on every read, but if you registered data after the MCP
server started, restart Claude Desktop to be safe.

### XGBoost library loading error (macOS)

```bash
brew install libomp
```

---

## Quick Reference

| Action | Command / Endpoint |
|--------|-------------------|
| Start app (native) | `uvicorn backend.main:app --reload --port 9500` |
| Start app (Docker) | `make start` |
| Generate data | `python scripts/generate_sensor_data.py` |
| Run tests | `pytest` |
| MCP server | `python -m backend.integrations.mcp.server` |
| Web UI | http://localhost:9500 |
| API docs (Swagger) | http://localhost:9500/docs |
| API docs (ReDoc) | http://localhost:9500/redoc |
| Health check | `curl http://localhost:9500/health` |
