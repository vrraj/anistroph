# Anistroph — Testing Guide

How to test Anistroph from automated tests to manual exploration of the REST
API, web UI, and MCP server.

---

## Prerequisites

```bash
cd /Users/raj/Documents/Raj/Anistroph
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

### How the MCP server works

The MCP server runs over **stdio** (standard input/output), not HTTP. There
is no port. The MCP client (e.g., Claude Desktop) spawns the server as a
subprocess and communicates via JSON-RPC messages piped through stdin/stdout.

```
Claude Desktop
    │
    ├── spawns subprocess: python -m backend.integrations.mcp.server
    │
    ├── stdin  → JSON-RPC requests (tools/list, tools/call)
    └── stdout ← JSON-RPC responses (tool results)
```

The server uses the low-level `mcp.server.Server` API (not `FastMCP`, which
is unavailable in MCP SDK 2.0.0). Each handler has the signature:

```python
async def handler(ctx: Any, params: RequestType) -> ResultType:
```

- **`ctx`** — the request context object passed by the MCP SDK. Contains
  session info, logging utilities, and progress reporting. Anistroph's tools
  are stateless and don't use it, but it must be accepted as the first
  parameter.
- **`params`** — the typed request parameters (e.g., `ListToolsRequest`,
  `CallToolRequest`).

All 8 tools call the same `AnistrophServices` as the REST API — no
duplicated logic, no separate analytical code, no arbitrary Python
execution. Training is intentionally not exposed through MCP.

### Run the server standalone

```bash
source .venv/bin/activate
python -m backend.integrations.mcp.server
```

The server will wait for JSON-RPC messages on stdin. It doesn't print
anything until it receives a request.

### Test the server manually via stdio

```bash
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}\n{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}\n{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n' | .venv/bin/python -m backend.integrations.mcp.server
```

Expected: two JSON responses — the `initialize` result and the `tools/list`
result containing all 8 Anistroph tools.

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
      "cwd": "/Users/raj/Documents/Raj/Anistroph"
    }
  }
}
```

Then ask Claude things like:
- "List all Anistroph datasets"
- "Profile the predictive_maintenance dataset"
- "What models are available in Anistroph?"
- "Predict failure probability for TOOL_000 at 2026-07-15T12:00:00 using model \<id\>"

### Test MCP with Claude Desktop (step-by-step)

This is a manual acceptance test to verify the MCP server works end-to-end
with a real AI client.

#### Prerequisites

1. Anistroph is set up and a dataset + model exist:
```bash
# Generate data and train a model first
source .venv/bin/activate
python scripts/generate_sensor_data.py
python -c "
from backend.services import get_services
svc = get_services()
svc.register_dataset_from_config(
    'datasets/predictive_maintenance/dataset.yaml',
    'data/synthetic/predictive_maintenance.csv',
)
result = svc.train('predictive_maintenance', 'failure_within_horizon', 'xgboost')
print('Model ID:', result['model_id'])
"
```

2. Claude Desktop is installed (download from https://claude.ai/download).

3. The MCP config file exists at:
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

#### Setup

Add the Anistroph MCP server to Claude Desktop's config:

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

> **Important:** Use the absolute path to the venv's Python so Claude
> Desktop picks up all installed dependencies. Using bare `python` may
> resolve to a system Python that doesn't have Anistroph installed.

Restart Claude Desktop after saving the config.

#### Verify the connection

1. Open Claude Desktop.
2. Start a new conversation.
3. You should see an **Anistroph** tool icon (hammer/tools menu) indicating
   the MCP server is connected.
4. If you don't see it, check Claude Desktop logs:
   ```bash
   # macOS — main.log contains MCP connection errors (mcp.log may be empty)
   tail -100 ~/Library/Logs/Claude/main.log | grep -i "mcp\|anistroph\|error"
   ```

#### Test prompts (run in order)

**Test 1 — List datasets:**
> Prompt: "List all Anistroph datasets and show their row counts."

Expected: Claude calls `anistroph_list_datasets` and reports the
`predictive_maintenance` dataset with its row count and entity count.

**Test 2 — Profile a dataset:**
> Prompt: "Profile the predictive_maintenance dataset. What's the failure
> rate?"

Expected: Claude calls `anistroph_profile_dataset` and reports row count,
entity count, time range, and the failure event distribution.

**Test 3 — Slice data:**
> Prompt: "Slice the predictive_maintenance dataset by machine_type and show
> the mean failure rate for each type."

Expected: Claude calls `anistroph_slice_data` with dimensions
`["machine_type"]`, metric `failure`, aggregation `mean`, and returns a
table of machine types with their failure rates.

**Test 4 — List models:**
> Prompt: "What models are available in Anistroph?"

Expected: Claude calls `anistroph_list_models` and lists all trained models
with their IDs, types, and dataset IDs.

**Test 5 — Get model metrics:**
> Prompt: "Show me the metrics for model \<MODEL_ID\>."

Expected: Claude calls `anistroph_get_model_metrics` and reports ROC-AUC,
PR-AUC, precision, recall, F1, and the confusion matrix.

**Test 6 — Predict:**
> Prompt: "Predict the failure probability for TOOL_000 at
> 2026-07-15T12:00:00 using model \<MODEL_ID\>."

Expected: Claude calls `anistroph_predict` with the entity_id and timestamp,
and reports the probability and binary prediction.

**Test 7 — Explain prediction:**
> Prompt: "Explain why that prediction came out that way. What are the top
> drivers?"

Expected: Claude calls `anistroph_explain_prediction` and reports the top
feature drivers (e.g., `vibration_mean_6h`, `temperature_slope_6h`) with
their impact values. The explanation comes from model feature importance,
not LLM fabrication.

**Test 8 — Training is blocked (negative test):**
> Prompt: "Train a new XGBoost model on predictive_maintenance."

Expected: Claude cannot do this. Training is not exposed through MCP by
design. Claude should explain that training must be done via the REST API
or Python directly.

#### What to verify

| Check | Expected |
|-------|----------|
| MCP server connects | Anistroph tools appear in Claude Desktop |
| Tool calls succeed | Claude returns real data, not hallucinated |
| Predictions match REST | Same model + entity + timestamp = same probability |
| Explanations are deterministic | Top drivers come from model, not LLM |
| Training is blocked | No `anistroph_train` tool exists |
| Same services as REST | MCP and REST produce identical results |

#### Troubleshooting

**Claude Desktop doesn't show Anistroph tools:**
- Check that the config JSON is valid (no trailing commas, proper braces).
- The config file must contain **only** the `mcpServers` key. If you mix in
  other keys (like `preferences`), Claude Desktop may overwrite the file and
  strip `mcpServers` on save. Use the minimal config shown above.
- Verify the Python path is correct: run
  `/Users/raj/Documents/Raj/Anistroph/.venv/bin/python -m backend.integrations.mcp.server`
  — it should start without errors.
- Check Claude Desktop logs for connection errors:
  ```bash
  # The main log file (not mcp.log, which may be empty)
  tail -100 ~/Library/Logs/Claude/main.log | grep -i "mcp\|anistroph\|error"
  ```
- Restart Claude Desktop completely (`Cmd+Q`, not just close window).

**Claude searches the web instead of using Anistroph tools:**
- This means the MCP server connected but `tools/list` failed silently.
- Test the server manually (see "Test the server manually via stdio" above).
- If you see `TypeError: takes 1 positional argument but 2 were given`,
  the handler is missing the `ctx` parameter. MCP SDK 2.0.0 passes a context
  object as the first argument to every handler:
  ```python
  # Wrong (MCP 1.x style):
  def handler(params) -> Result:

  # Right (MCP 2.0.0):
  async def handler(ctx: Any, params: RequestType) -> ResultType:
  ```
- If you see `TypeError: object ListToolsResult can't be used in 'await'
  expression`, the handler is sync but MCP awaits all handlers. Make it
  `async`.

**Config file keeps getting overwritten by Claude Desktop:**
- Claude Desktop writes to `claude_desktop_config.json` on startup. If the
  file contains keys it doesn't recognize (or mixes `mcpServers` with
  `preferences`), it may rewrite the file without `mcpServers`.
- Solution: keep the config file minimal — only the `mcpServers` key.

**Tool calls return errors:**
- Ensure a dataset is registered and a model is trained before calling
  prediction tools.
- Verify `data/` and `artifacts/` directories have content from the
  prerequisite setup step.
- Run `pytest tests/integration/test_mcp.py -v` to verify the MCP tools
  work programmatically.

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
