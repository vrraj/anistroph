# Anistroph — Setup & Usage Guide

Step-by-step instructions for setting up Anistroph, generating data,
registering datasets, training models, making predictions, and querying
through REST, Python, MCP, and the web UI.

---

## 0. AI Integration — Claude (MCP) and ChatGPT (GPT Actions)

Anistroph exposes runtime analysis and inference through two AI integration
protocols. Both call the same underlying Python services. Neither exposes
model training — training is an admin operation.

### Protocol Comparison

| | Claude (MCP) | ChatGPT (GPT Actions) |
|---|---|---|
| **Protocol** | Model Context Protocol (stdio) | OpenAPI / REST |
| **Transport** | Local subprocess (stdio) | Public HTTPS via ngrok tunnel |
| **Spec URL** | N/A (tools defined in Python) | `https://<ngrok-url>/openapi-gpt.json` |
| **Scope** | 9 runtime tools | 13 runtime REST endpoints |
| **Training exposed?** | No | No |
| **Setup** | Claude Desktop config JSON | Custom GPT → Actions → Import URL |
| **Best for** | Local analysis, IDE integration | Cloud-based conversational analysis |

### What is exposed (runtime only)

Both protocols expose the same capabilities:

| Capability | MCP Tool | REST Endpoint |
|-----------|----------|---------------|
| List datasets | `anistroph_list_datasets` | `GET /datasets` |
| Profile dataset | `anistroph_profile_dataset` | `GET /datasets/{id}/profile` |
| Slice data | `anistroph_slice_data` | `POST /analysis/slice` |
| Compare data | `anistroph_compare_data` | `POST /analysis/compare` |
| Find interesting slices | `anistroph_find_interesting_slices` | *(via /analysis/slice)* |
| List models | `anistroph_list_models` | `GET /models` |
| Get model metrics | `anistroph_get_model_metrics` | `GET /models/{id}/metrics` |
| Predict | `anistroph_predict` | `POST /predictions` |
| Explain (SHAP) | `anistroph_explain_prediction` | `POST /predictions/explain` |
| Evaluate on held-out set | `anistroph_evaluate_model` | `POST /evaluations/{model_id}` |
| Find error slices | `anistroph_find_evaluation_slices` | `POST /evaluations/{model_id}/slices` |

### Claude Desktop (MCP stdio)

MCP uses a local stdio subprocess — no public URL needed.

1. Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "anistroph": {
      "command": "/Users/raj/Documents/Raj/anistroph/.venv/bin/python",
      "args": ["-m", "backend.integrations.mcp.server"],
      "cwd": "/Users/raj/Documents/Raj/anistroph"
    }
  }
}
```

2. Restart Claude Desktop.

3. Ask Claude:
> "List all Anistroph datasets"
> "Predict wafer yield for WAFER_015000 using model wafer-yield-xgb-v001"
> "Explain that prediction — what pushed the yield up or down?"
> "Find the worst yield combinations in the semiconductor dataset"

### ChatGPT (GPT Actions via ngrok)

ChatGPT runs in the cloud and needs a public URL. Use ngrok to temporarily
tunnel your local server.

1. Install ngrok (one-time):
```bash
brew install ngrok
ngrok config add-authtoken <your-token>   # get a free token at ngrok.com
```

2. Start the server + tunnel:
```bash
make start-gpt
```

Output:
```
========================================================
  Anistroph is now public for ChatGPT GPT Actions
========================================================

  Public URL:        https://abc123.ngrok.app
  OpenAPI (GPT):     https://abc123.ngrok.app/openapi-gpt.json
  Health:            https://abc123.ngrok.app/health

  To configure ChatGPT:
    1. Go to https://chat.openai.com/gpts
    2. Create a new GPT -> Configure -> Actions
    3. Import from URL: https://abc123.ngrok.app/openapi-gpt.json
    4. No auth required

  To stop:  make stop-gpt
========================================================
```

3. In ChatGPT, create a Custom GPT:
   - Go to **https://chat.openai.com/gpts** → **Create**
   - **Configure** → **Actions** → **Import from URL**
   - Paste the OpenAPI URL from `make start-gpt` output
   - No authentication required
   - Save

4. Ask your GPT:
> "List all Anistroph datasets"
> "Predict wafer yield for WAFER_015000 using model wafer-yield-xgb-v001"
> "Explain that prediction with SHAP"
> "Slice the semiconductor dataset by etch tool and chamber"

5. Stop the tunnel when done (the public URL becomes inaccessible):
```bash
make stop-gpt       # stops both ngrok + server
make stop-ngrok     # stops only ngrok (server stays on localhost)
```

> **Security:** The ngrok URL is temporary and random. It only works while
> `make start-gpt` is running. Running `make stop-gpt` immediately closes
> the public tunnel. The filtered OpenAPI spec (`/openapi-gpt.json`)
> excludes training and dataset registration — only runtime prediction,
> explanation, and analysis are exposed.

---

## 1. Installation

### Local (native)

```bash
cd /Users/raj/Documents/Raj/anistroph

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

### Dataset partitioning

Every registered dataset is automatically partitioned into separate Parquet
files at registration time:

| File | Purpose | Used during training? |
|------|---------|----------------------|
| `{dataset_id}.train.parquet` | Model fitting | Yes (training loads only this file) |
| `{dataset_id}.evaluation.parquet` | Held-out evaluation | Never — used post-training via the Evaluation tab / `anistroph_evaluate_model` |
| `{dataset_id}.validate.parquet` | Validation during training (optional) | Yes (early stopping, threshold tuning) — only if `VALIDATE_DATASET_PCT > 0` |

The full dataset is also persisted at `{dataset_id}.parquet` for backward
compatibility (profiling, sample rows).

**Split percentages** — `.env` provides global defaults:

```bash
TRAIN_DATASET_PCT=0.80
EVAL_DATASET_PCT=0.20
VALIDATE_DATASET_PCT=0.0   # reserved for future validation-during-training
```

Per-dataset YAML `split:` sections override these defaults. The YAML uses
`train` / `validation` / `test` keys, where `test` maps to the evaluation
partition:

```yaml
split:
  strategy: chronological   # or "random" for non-temporal datasets
  train: 0.80
  validation: 0.0           # future validate partition
  test: 0.20                # evaluation partition
```

To skip partitioning entirely (single-file mode), set `train: 1.0` in the YAML.

**Temporal datasets** sort chronologically — oldest rows go to train, newest
to evaluation. **Non-temporal datasets** shuffle with a fixed seed before
splitting.

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

## 7b. Evaluate a Model on the Held-Out Set

Evaluation runs a trained model against the **held-out evaluation partition**
(`evaluation.parquet`) — the slice reserved at dataset registration time that
is never seen during training. This gives an unbiased estimate of model
performance on unseen data.

**How it works:**
1. Loads the persisted model artifact from disk (`.joblib`).
2. Loads `evaluation.parquet` for the model's dataset.
3. Builds features using the persisted FeatureMetadata (`fit=False` — no refit).
4. Runs inference (`predict` / `predict_proba`) on the full eval set.
5. Computes aggregate metrics and returns a sample of prediction-vs-actual rows.

**Important:** Evaluation is a runtime operation — it does **not** retrain the
model. Training uses only `train.parquet`; evaluation uses only
`evaluation.parquet`. There is no overlap. Results are computed fresh each
time (not cached), so they always reflect the current model artifact and eval
partition. Safe to re-run; produces the same numbers as long as neither has
changed.

### Metrics by target type

**Classification:**

| Metric | Description |
|--------|-------------|
| `roc_auc` | Area under the ROC curve |
| `pr_auc` | Area under the precision-recall curve |
| `precision` | Precision at the decision threshold |
| `recall` | Recall at the decision threshold |
| `f1` | F1 score at the decision threshold |
| `confusion_matrix` | TN / FP / FN / TP counts |
| `threshold` | Decision threshold used (optimized for F1 on validation set) |

**Regression:**

| Metric | Description |
|--------|-------------|
| `mae` | Mean Absolute Error (same units as target) |
| `mse` | Mean Squared Error (squared units) |
| `rmse` | Root Mean Squared Error (same units as target) |
| `r2` | Coefficient of determination (0–1, higher is better) |
| `mape` | Mean Absolute Percentage Error (% — error relative to actual value) |
| `max_error` | Worst single prediction in the eval set |
| `median_abs_error` | Median of absolute errors (robust to outliers) |
| `p95_abs_error` | 95th percentile of absolute errors |
| `mean_prediction_error` | Mean of (predicted − actual) — bias direction |
| `baseline` | Same metrics for a constant mean predictor (for comparison) |

**MAPE** is particularly useful for price forecasting: a $50K error means
different things for a $600K home vs a $4M home. MAPE normalizes by the actual
value, giving the average error as a percentage.

### Slice-level evaluation (filtered metrics)

Aggregate metrics can hide important performance gaps across segments. For
example, a home-price model might have 3.25% overall MAPE but 1.2% in Saratoga
and 5.5% in San Jose — the aggregate hides the San Jose weakness.

Pass ``filters`` to evaluate on a subset of the evaluation partition. The
response includes **both** overall metrics (all eval rows) and
``filtered_metrics`` (matching rows only), so you can compare side by side.
Filters support equality (``{"city": "Saratoga"}``) and IN-style lists
(``{"lot_id": ["LOT_001", "LOT_002"]}``), same as ``sample_rows``.

When filters are applied, the ``predictions_sample`` is drawn from the filtered
subset (not the full set).

### Via Python

```python
from backend.services import get_services

# Overall evaluation (all eval rows)
result = get_services().evaluate_model("wafer-yield-xgb-v001", sample_size=50)
metrics = result["metrics"]
print(f"R²: {metrics['r2']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"Eval rows: {result['eval_row_count']}")
print(f"Sample: {result['predictions_sample'][:3]}")

# Slice-level evaluation (filtered to a single city)
result = get_services().evaluate_model(
    "home_prices-xgb-v001",
    sample_size=50,
    filters={"city": "Saratoga"},
)
print(f"Overall MAPE: {result['metrics']['mape']:.2f}%")
print(f"Saratoga MAPE: {result['filtered_metrics']['mape']:.2f}%  (n={result['filtered_row_count']})")
```

### Via REST API

```bash
# Overall evaluation
curl -X POST http://localhost:9500/evaluations/wafer-yield-xgb-v001 \
  -H "Content-Type: application/json" \
  -d '{"sample_size": 50}'

# Slice-level evaluation (filtered to Saratoga)
curl -X POST http://localhost:9500/evaluations/home_prices-xgb-v001 \
  -H "Content-Type: application/json" \
  -d '{"sample_size": 50, "filters": {"city": "Saratoga"}}'
```

Response includes `model_id`, `dataset_id`, `target_name`, `target_type`,
`eval_row_count`, `metrics`, and `predictions_sample` (list of
`{entity_id, actual, predicted, error, abs_error}` rows for regression, or
`{entity_id, actual, predicted, predicted_label}` for classification).

When `filters` is provided, the response also includes:
- `filtered_metrics`: aggregate metrics for the matching rows only
- `filtered_row_count`: number of rows matching the filters
- `filters`: echo of the filters applied

### Via MCP (Claude Desktop)

> "Evaluate model wafer-yield-xgb-v001 on the held-out set"
> "What's the MAPE for the home price model?"
> "Show me the worst predictions in the evaluation set"
> "Evaluate the home price model filtered to San Jose only"
> "Compare MAPE for Saratoga vs Los Gatos vs San Jose"

### Via web UI

Open the **Evaluation** tab (`http://localhost:9500/#evaluation`):
1. Select a trained model from the dropdown. Categorical filter dropdowns
   (e.g. city, zip_code, lot_id) are populated automatically from the model's
   dataset profile.
2. Set the sample size (number of prediction-vs-actual rows to display, capped at 1000).
3. Optionally select values from the filter dropdowns to evaluate on a subset.
   Leave all filters on "All" to evaluate on the full eval set.
4. Click **Evaluate**.
5. The results show **Overall Metrics** (all eval rows) and, if filters are
   applied, **Filtered Metrics** (matching rows only) side by side, followed by
   a prediction-vs-actual sample table from the filtered subset.

### Evaluation vs. Get Model Metrics

| | Get Model Metrics (§7) | Evaluate Model (§7b) |
|---|---|---|
| **Data used** | Validation set (carved during training) | Held-out evaluation partition |
| **When computed** | At training time, stored in model registry | On demand, each call |
| **Cached?** | Yes — stored in model metadata | No — recomputed every time |
| **Endpoint** | `GET /models/{id}/metrics` | `POST /evaluations/{id}` |
| **Purpose** | Training-time performance snapshot | Unbiased post-training performance |

### Error Slice Discovery (§7c)

Aggregate evaluation metrics can hide populations where the model performs
materially better or worse than average. **Error slice discovery** runs
inference on the held-out evaluation partition, computes per-row error, and
searches 1/2/3-dimensional combinations of categorical columns for slices
where the error metric deviates most from the overall baseline.

This is the model-evaluation analogue of `find_interesting_slices`:
- `find_interesting_slices` finds populations where the **target** deviates
  (e.g. "which wafer combinations have unusually low yield?")
- `find_evaluation_slices` finds populations where the **prediction error**
  deviates (e.g. "which wafer combinations does the model predict worst?")

Both apply minimum sample-size thresholds when ranking slices, and both work
across any registered dataset — the dimensions are auto-detected from the
dataset spec's categorical feature columns.

**Supported error metrics:**

| Metric | Target type | Description |
|--------|-------------|-------------|
| `abs_error` | Regression | Absolute error (default) — identifies populations with highest absolute prediction error |
| `error` | Regression | Signed error — shows bias direction (positive = over-predicting, negative = under-predicting) |
| `pct_error` | Regression | Percentage error — relative to actual value, useful when target ranges vary widely (e.g. home prices) |
| `log_loss` | Classification | Per-row log loss — identifies populations where predicted probabilities are most wrong |

**Example output (home_prices model, abs_error):**

```
Overall baseline MAE: $94,912

Top error slices:
  zip_code=95071:                    n=177, MAE=$126,022, diff=+$31,110  (1.33x — worst)
  city=Saratoga + zip_code=95071:    n=177, MAE=$126,022, diff=+$31,110  (1.33x)
  zip_code=95122:                    n=180, MAE=$65,337,  diff=-$29,575  (0.69x — best)
  city=San Jose + zip_code=95122:    n=180, MAE=$65,337,  diff=-$29,575  (0.69x)
  city=Saratoga:                     n=366, MAE=$122,513, diff=+$27,602  (1.29x)
```

This tells you the model struggles most with Saratoga 95071 (expensive homes,
wide price range) and does best with San Jose 95122 (affordable, tight band).
Pair with filtered evaluation (§7b) to drill into the full metrics for any
slice.

### Via Python

```python
from backend.services import get_services

# Find populations where absolute error is worst
slices = get_services().find_evaluation_slices(
    "home_prices-xgb-v001",
    metric="abs_error",
    min_sample_size=50,
    top_k=20,
)
for s in slices:
    vals = " + ".join(f"{k}={v}" for k, v in s["values"].items())
    print(f"  {vals}: n={s['row_count']}, MAE={s['metric_value']:.0f}, diff={s['difference']:+.0f}")

# Use percentage error for price datasets (relative, not absolute)
slices = get_services().find_evaluation_slices(
    "home_prices-xgb-v001",
    metric="pct_error",
    top_k=10,
)

# Then drill into a specific slice with filtered evaluation
result = get_services().evaluate_model(
    "home_prices-xgb-v001",
    filters={"zip_code": "95071"},
)
print(f"Overall MAPE: {result['metrics']['mape']:.2f}%")
print(f"95071 MAPE:   {result['filtered_metrics']['mape']:.2f}%  (n={result['filtered_row_count']})")
```

### Via REST API

```bash
curl -X POST http://localhost:9500/evaluations/home_prices-xgb-v001/slices \
  -H "Content-Type: application/json" \
  -d '{"metric": "abs_error", "min_sample_size": 50, "top_k": 20}'
```

Response is a list of slices sorted by `abs_difference` descending, each with:
`dimensions`, `values`, `row_count`, `metric_value`, `metric_median`,
`metric_std`, `overall_baseline`, `difference`, `abs_difference`.

### Via MCP (Claude Desktop)

> "Find the populations where the home price model has the worst prediction error"
> "Which wafer combinations does the semiconductor model struggle with most?"
> "Show me error slices by percentage error for the home price model"
> "Where is the model over-predicting vs under-predicting?" *(uses `error` metric)*
> "Find slices where the model's log loss is highest" *(classification)*

### Via web UI

On the **Evaluation** tab, click **Find Error Slices** (next to Evaluate).
The results show a table of ranked slices with dimensions, values, row count,
error metric, difference from baseline, and ratio (e.g. 1.33x = 33% worse
than average). Red = worse than average, green = better.

### find_interesting_slices vs find_evaluation_slices

| | `find_interesting_slices` | `find_evaluation_slices` |
|---|---|---|
| **What it finds** | Slices where the *target* deviates | Slices where the *prediction error* deviates |
| **Input** | Dataset ID + metric column | Model ID (runs inference first) |
| **Question answered** | "Which populations have unusual outcomes?" | "Which populations does the model predict worst?" |
| **MCP tool** | `anistroph_find_interesting_slices` | `anistroph_find_evaluation_slices` |
| **REST endpoint** | `POST /analysis/slice` (manual) | `POST /evaluations/{model_id}/slices` |
| **Requires trained model** | No | Yes |

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
      "command": "/Users/raj/Documents/Raj/anistroph/.venv/bin/python",
      "args": ["-m", "backend.integrations.mcp.server"],
      "cwd": "/Users/raj/Documents/Raj/anistroph"
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
| `anistroph_find_interesting_slices` | Discover slices with the largest deviation from baseline |
| `anistroph_sample_rows` | Return up to N raw rows, optionally filtered by column values |
| `anistroph_list_models` | List all trained models |
| `anistroph_get_model_metrics` | Get evaluation metrics for a model |
| `anistroph_predict` | Make a prediction (entity_id + timestamp) |
| `anistroph_explain_prediction` | Explain a prediction with top drivers |
| `anistroph_evaluate_model` | Evaluate a model against the held-out evaluation partition |

### Example MCP prompts

These prompts work with **any stdio MCP client** — Claude Desktop, Claude CLI,
Cursor, Cline, etc. The MCP server uses stdio transport, so the client is
interchangeable; "Claude Desktop" appears elsewhere in this doc only as the
example client. The LLM in the client maps the natural-language prompt to the
appropriate `anistroph_*` tool call.

**Discovery & profiling**
- "List all Anistroph datasets"
- "Profile the predictive_maintenance dataset"
- "Profile the semiconductor_yield dataset"
- "What columns and types are in semiconductor_yield?"
- "What's the time range of the semiconductor data?"

**Slicing & comparison**
- "Slice predictive_maintenance by machine_type and show mean failure rate"
- "Slice semiconductor_yield by etch_tool and etch_chamber, metric wafer_yield, aggregation mean"
- "Compare wafer_yield across fab_id values"
- "Show yield by etch tool and chamber, filtered to product_id = PROD_A"
- "Slice semiconductor_yield by process_route, metric wafer_yield, aggregation min"

**Interesting slice discovery**
- "Find the worst yield combinations in the semiconductor dataset"
- "Find the most unusual slices in predictive_maintenance by failure_within_horizon"
- "What combinations have the lowest wafer yield?"
- "Show me the top 10 interesting slices for wafer_yield"

**Raw row inspection**
- "Show me 20 rows from semiconductor_yield"
- "Show me the row for wafer_id WAFER_015000"
- "Show me 10 semiconductor_yield rows where etch_tool is ETCH_02 and etch_chamber is CH_B"
- "Show me the 5 lowest-yield wafers — just wafer_id, etch_tool, etch_chamber, and wafer_yield"
- "Show me rows for etch_tool ETCH_02 or ETCH_03, sorted by wafer_yield descending, top 20"

**Models & metrics**
- "What models are available?"
- "Show me the metrics for model pm-xgb"
- "List models trained on semiconductor_yield"

**Prediction & explanation**
- "Predict failure probability for TOOL_000 at 2026-06-15T12:00:00 using model pm-xgb"
- "Predict wafer yield for WAFER_015000 using model wafer-yield-xgb-v001"
- "Explain that prediction — what are the top drivers?"
- "Explain the prediction for WAFER_015000 with top_k=15"

**Held-out evaluation**
- "Evaluate model pm-xgb against the held-out evaluation set"
- "Run evaluation on wafer-yield-xgb-v001 and show me 20 prediction-vs-actual rows"
- "What's the MAE and R² for model wafer-yield-xgb-v001 on the evaluation partition?"
- "Evaluate model pm-xgb — how does it compare to the baseline?"
- "Evaluate the home price model filtered to San Jose only"
- "Compare MAPE for Saratoga vs Los Gatos vs San Jose in the home price model"

**Error slice discovery**
- "Find the populations where the home price model has the worst prediction error"
- "Which wafer combinations does the semiconductor model struggle with most?"
- "Show me error slices by percentage error for the home price model"
- "Where is the model over-predicting vs under-predicting?"
- "Find slices where the model's log loss is highest"

**Not available via MCP** (use REST, Python, or the Web UI instead):
- Dataset registration, model training, and deletion — these are admin
  operations exposed only through REST and Python services.

---

## 13. Troubleshooting

### MCP tools not appearing in Claude Desktop

- Check config JSON is valid: `python3 -c "import json; json.load(open('~/Library/Application Support/Claude/claude_desktop_config.json'))"`
- Check logs: `tail -100 ~/Library/Logs/Claude/main.log | grep -i "mcp\|anistroph\|error"`
- Fully quit Claude Desktop (`Cmd+Q`), not just close the window
- Verify the venv Python path exists: `ls /Users/raj/Documents/Raj/anistroph/.venv/bin/python`

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

---

## 14. Reference Model: anistroph-sentinel-v1

The reference model trained on the synthetic predictive-maintenance
dataset. Use this to test predictions, explanations, and MCP tool calls.

### Overview

| Property | Value |
|----------|-------|
| Model ID | `anistroph-sentinel-v1` |
| Model type | XGBoost (gradient-boosted trees) |
| Dataset | `predictive_maintenance` |
| Target | `failure_within_horizon` (binary: will this machine fail within 24h?) |
| Decision threshold | 0.199 (optimized for F1) |
| Training data | 20 machines, 30 days, 10-min intervals (86,400 rows) |
| Split | Chronological — 70% train / 15% validation / 15% test |

### Evaluation metrics (test set)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ROC-AUC | 0.767 | Good discrimination — well above 0.5 (random) |
| PR-AUC | 0.381 | Reasonable for imbalanced data (~0.14% failure rate) |
| Precision | 0.394 | ~40% of predicted failures are true failures |
| Recall | 0.701 | Catches ~70% of actual failures |
| F1 Score | 0.505 | Harmonic mean of precision and recall |
| True positives | 1,957 | Correctly predicted failures |
| False positives | 3,009 | False alarms |
| False negatives | 835 | Missed failures |
| True negatives | 7,159 | Correctly predicted non-failures |

### Training data details

**Entities:** 20 machines (`TOOL_000` through `TOOL_019`)

**Machine types:**
- `TYPE_A` — 30,240 rows (failure rate: 0.136%)
- `TYPE_B` — 30,240 rows (failure rate: 0.152%)
- `TYPE_C` — 25,920 rows (failure rate: 0.147%)

**Time range:** June 1 – June 30, 2026 (10-minute intervals)

**Sensor columns:**

| Column | Min | Mean | Max | Description |
|--------|-----|------|-----|-------------|
| temperature | 60.3 | 74.7 | 115.7 | Operating temperature (°C) |
| vibration | 1.09 | 2.19 | 8.10 | Vibration intensity (g) |
| pressure | 77.9 | 99.6 | 121.8 | System pressure (bar) |
| current | 8.65 | 10.16 | 11.74 | Electrical current (A) |
| voltage | 222.8 | 230.0 | 237.5 | Voltage (V) |
| rpm | 1,699.8 | 1,793.6 | 1,884.1 | Rotational speed (RPM) |
| flow_rate | 57.0 | 60.0 | 62.9 | Flow rate (L/min) |
| maintenance_age_hours | -23.5 | 47.6 | 323.7 | Hours since last maintenance |
| operating_hours | 33.7 | 2,802.0 | 5,686.0 | Total operating hours |

**Event columns:**
- `failure` (boolean): 125 failures out of 86,400 rows (~0.14%)
- `failure_type`: PRESSURE (82), THERMAL (39), MECHANICAL (4)

### Engineered features (26 total)

The Feature Engine transforms raw sensor readings into 26 model-ready
features. All rolling windows are computed per-entity (per machine) and
are leakage-safe (no future data used).

**Temperature features (6):**

| Feature | Transform | Window | What it captures |
|---------|-----------|--------|------------------|
| `temperature_current` | current | — | Instantaneous temperature |
| `temperature_mean_1h` | rolling mean | 1 hour | Short-term thermal trend |
| `temperature_mean_6h` | rolling mean | 6 hours | Long-term thermal baseline |
| `temperature_std_1h` | rolling std | 1 hour | Short-term thermal volatility |
| `temperature_std_6h` | rolling std | 6 hours | Long-term thermal instability |
| `temperature_slope_6h` | rolling slope | 6 hours | Temperature trend direction |

**Vibration features (7):**

| Feature | Transform | Window | What it captures |
|---------|-----------|--------|------------------|
| `vibration_current` | current | — | Instantaneous vibration |
| `vibration_mean_1h` | rolling mean | 1 hour | Short-term vibration level |
| `vibration_mean_6h` | rolling mean | 6 hours | Long-term vibration baseline |
| `vibration_max_6h` | rolling max | 6 hours | Peak vibration in last 6h |
| `vibration_std_1h` | rolling std | 1 hour | Short-term vibration volatility |
| `vibration_std_6h` | rolling std | 6 hours | Long-term vibration instability |
| `vibration_slope_6h` | rolling slope | 6 hours | Vibration trend direction |

**Pressure features (3):**

| Feature | Transform | Window | What it captures |
|---------|-----------|--------|------------------|
| `pressure_current` | current | — | Instantaneous pressure |
| `pressure_mean_1h` | rolling mean | 1 hour | Short-term pressure level |
| `pressure_std_6h` | rolling std | 6 hours | Long-term pressure instability |

**RPM features (2):**

| Feature | Transform | Window | What it captures |
|---------|-----------|--------|------------------|
| `rpm_current` | current | — | Instantaneous rotational speed |
| `rpm_mean_1h` | rolling mean | 1 hour | Short-term RPM level |

**Single-value features (5):**

| Feature | What it captures |
|---------|------------------|
| `current_current` | Electrical current draw |
| `voltage_current` | Operating voltage |
| `flow_rate_current` | Fluid flow rate |
| `maintenance_age_hours_current` | Hours since last maintenance |
| `operating_hours_current` | Cumulative operating hours |

**Categorical features (3 — one-hot encoded):**

| Feature | What it captures |
|---------|------------------|
| `machine_type__TYPE_A` | Machine is type A |
| `machine_type__TYPE_B` | Machine is type B |
| `machine_type__TYPE_C` | Machine is type C |

### Target construction

The target `failure_within_horizon` is a **future_event** target:

- **Source column:** `failure` (boolean — did a failure occur at this timestamp?)
- **Horizon:** 24 hours forward
- **Logic:** For each observation at time T, the target is 1 if the
  machine will experience a failure at any point between T and T+24h.
- **Entity isolation:** The horizon is computed per-machine — machine A's
  future failures never affect machine B's target.
- **Leakage prevention:** The target looks forward in time; features look
  backward. Training/inference never see future target information.

### Sample predictions

Predictions return a probability (0.0–1.0) and a binary prediction
(0 or 1, based on the 0.199 threshold).

| Machine | Timestamp | Probability | Prediction | Interpretation |
|---------|-----------|-------------|------------|----------------|
| TOOL_000 | 2026-06-15T12:00 | 0.189 | 0 | Low risk — no failure expected in 24h |
| TOOL_000 | 2026-06-28T12:00 | 0.384 | 1 | Elevated risk — failure likely within 24h |
| TOOL_005 | 2026-06-28T12:00 | 0.000 | 0 | Very low risk — healthy machine |
| TOOL_010 | 2026-06-28T12:00 | 0.986 | 1 | Very high risk — failure imminent |

### Insights

- **ROC-AUC 0.767** means the model ranks at-risk machines correctly
  ~77% of the time — significantly better than random.
- **Recall 70%** means the model catches 7 out of 10 actual failures.
  The 30% missed failures (false negatives) are the most costly in a
  real maintenance setting — consider lowering the threshold to catch
  more, at the cost of more false alarms.
- **Precision 39%** means ~4 in 10 predicted failures are real. The
  ~6 in 10 false alarms could be filtered by a secondary review process.
- **Failure type imbalance:** PRESSURE failures (82) dominate, followed
  by THERMAL (39) and MECHANICAL (4). The model may perform better on
  PRESSURE failures than MECHANICAL ones due to sample size.
- **Machine type differences are small** (0.136%–0.152% failure rates),
  suggesting the synthetic generator creates roughly balanced types.
- **Late-month predictions are higher** because machines deteriorate
  over time — `maintenance_age_hours` and `operating_hours` increase,
  and rolling statistics capture the degradation trend.

### How to use this model

**Via Claude Desktop (MCP):**
> "List all Anistroph models"
> "Show me the metrics for model anistroph-sentinel-v1"
> "Predict failure probability for TOOL_010 at 2026-06-28T12:00:00 using model anistroph-sentinel-v1"
> "Explain that prediction — what are the top drivers?"

**Via REST API:**
```bash
# Predict
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{"model_id": "anistroph-sentinel-v1", "entity_id": "TOOL_010", "timestamp": "2026-06-28T12:00:00"}'

# Explain
curl -X POST http://localhost:9500/predictions/explain \
  -H "Content-Type: application/json" \
  -d '{"model_id": "anistroph-sentinel-v1", "entity_id": "TOOL_010", "timestamp": "2026-06-28T12:00:00", "top_k": 10}'
```

**Via Python:**
```python
from backend.services import get_services
svc = get_services()

# Predict
pred = svc.predict("anistroph-sentinel-v1", entity_id="TOOL_010", timestamp="2026-06-28T12:00:00")
print(f"Probability: {pred['probability']:.4f}, Prediction: {pred['prediction']}")

# Explain
expl = svc.explain("anistroph-sentinel-v1", entity_id="TOOL_010", timestamp="2026-06-28T12:00:00", top_k=10)
for d in expl["top_drivers"]:
    print(f"  {d['feature']}: {d['impact']:.4f}")
```

---

## 15. All Tools & API Reference

### REST API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — returns `{"status": "ok", "version": "0.1.0"}` |
| GET | `/datasets` | List all registered datasets |
| POST | `/datasets` | Register a new dataset from config + source file |
| GET | `/datasets/{dataset_id}` | Get metadata for a specific dataset |
| GET | `/datasets/{dataset_id}/profile` | Profile a dataset (row count, distributions, time range) |
| POST | `/datasets/{dataset_id}/rows` | Sample raw rows (optional filters, columns, sort; capped at 1000) |
| DELETE | `/datasets/{dataset_id}` | Remove a dataset from the registry |
| GET | `/models` | List all trained models |
| GET | `/models/types` | List available model types (`xgboost`, `logistic_regression`) |
| POST | `/models/train` | Train a new model |
| GET | `/models/{model_id}` | Get metadata for a specific model |
| GET | `/models/{model_id}/metrics` | Get training-time metrics (stored in model registry) |
| POST | `/evaluations/{model_id}` | Evaluate a model on the held-out eval partition (recomputed each call) |
| POST | `/evaluations/{model_id}/slices` | Find populations where model error deviates from overall baseline |
| DELETE | `/models/{model_id}` | Remove a model from the registry |
| POST | `/predictions` | Make a single prediction (entity_id + timestamp or records) |
| POST | `/predictions/batch` | Make multiple predictions in one request |
| POST | `/predictions/explain` | Explain a prediction with top feature drivers |
| POST | `/analysis/slice` | Slice data by dimensions with aggregation |
| POST | `/analysis/compare` | Compare a metric across dimension values |
| GET | `/` | Web UI (static HTML) |
| GET | `/docs` | Swagger UI (interactive API docs) |
| GET | `/redoc` | ReDoc (alternative API docs) |

### MCP tools (Claude Desktop)

| Tool | Arguments | Description |
|------|-----------|-------------|
| `anistroph_list_datasets` | *(none)* | List all registered datasets in Anistroph. Returns dataset IDs, names, row counts, entity keys, time ranges. |
| `anistroph_profile_dataset` | `dataset_id` (string) | Profile a registered dataset. Returns row count, column count, column types, missing values, unique counts, numeric distributions (min/mean/max/std), categorical distributions, time range, entity count, and event distribution. |
| `anistroph_slice_data` | `dataset_id` (string), `dimensions` (array of strings), `metric` (string), `aggregation` (string, default "mean"), `filters` (object, optional), `limit` (integer, optional) | Slice a dataset by dimensions with an aggregation over a metric. Returns one row per dimension combination with the aggregated metric value. |
| `anistroph_compare_data` | `dataset_id` (string), `dimension` (string), `metric` (string), `aggregation` (string, default "mean"), `filters` (object, optional) | Compare a metric across values of a single dimension. Returns one row per dimension value with the aggregated metric. |
| `anistroph_list_models` | *(none)* | List all registered trained models. Returns model IDs, types, dataset IDs, and creation timestamps. |
| `anistroph_get_model_metrics` | `model_id` (string) | Get the evaluation metrics for a trained model. Returns ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, and decision threshold. |
| `anistroph_predict` | `model_id` (string), `entity_id` (string, optional), `timestamp` (string, optional), `records` (array of objects, optional) | Make a prediction using a trained model. For temporal datasets, provide `entity_id` and `timestamp` — the server retrieves historical observations and builds features. For non-temporal datasets, provide `records` (raw feature dicts). Returns probability and binary prediction. |
| `anistroph_explain_prediction` | `model_id` (string), `entity_id` (string, optional), `timestamp` (string, optional), `records` (array of objects, optional), `top_k` (integer, default 10) | Explain a prediction by returning the top contributing features. Returns the same probability plus a list of top drivers with feature names and impact values. Explanations are deterministic and model-derived — no LLM fabrication. |
| `anistroph_find_interesting_slices` | `dataset_id` (string), `metric` (string), `dimensions` (array of strings, optional), `min_sample_size` (integer, default 100), `max_dimensions` (integer, default 3), `aggregation` (string, default "mean"), `filters` (object, optional), `top_k` (integer, default 20) | Find slices with the largest deviation from the overall metric baseline. Searches 1, 2, and 3-dimensional combinations of categorical columns. Returns ranked slices with dimension values, row count, metric value, and difference from baseline. |
| `anistroph_sample_rows` | `dataset_id` (string), `n` (integer, default 10), `filters` (object, optional), `columns` (array of strings, optional), `sort_by` (string, optional), `descending` (boolean, default false) | Return up to `n` raw rows (capped at 1000) from a registered dataset, optionally filtered by column values, with an optional column subset and sort. Use to inspect individual records (e.g. a specific `wafer_id`) rather than aggregations. `filters` supports equality (`{"col": "value"}`) and IN-style (`{"col": ["a", "b"]}`). Returns `dataset_id`, `row_count` (after filtering), `returned`, `columns`, and `rows` (list of dicts). |
| `anistroph_evaluate_model` | `model_id` (string), `sample_size` (integer, default 50), `filters` (object, optional) | Evaluate a trained model against the dataset's held-out evaluation partition. Loads `evaluation.parquet`, runs inference using the persisted model, and compares predictions against known actual target values. Returns aggregate metrics (MAE/MSE/RMSE/R²/MAPE/max_error for regression, AUC/precision/recall/F1 for classification) and a `predictions_sample` list of `{entity_id, actual, predicted, error}` rows. Optional `filters` enable slice-level evaluation (returns both overall and filtered metrics). The evaluation set is never used during training. |
| `anistroph_find_evaluation_slices` | `model_id` (string), `metric` (string, default "abs_error"), `min_sample_size` (integer, default 50), `max_dimensions` (integer, default 3), `top_k` (integer, default 20) | Find populations where model prediction error deviates most from the overall average. Runs inference on the held-out evaluation partition, computes per-row error, and searches 1/2/3-dimensional combinations of categorical columns for slices where the error metric differs materially from the overall baseline. Regression metrics: `abs_error`, `error` (signed), `pct_error`. Classification: `log_loss`. Returns ranked slices with dimension values, row count, error value, overall baseline, and difference. |

### Python service methods

| Method | Arguments | Returns | Description |
|--------|----------|---------|-------------|
| `get_services().list_datasets()` | *(none)* | `list[DatasetMeta]` | List all registered datasets |
| `get_services().register_dataset_from_config(config_path, source_path, parquet_path?)` | config_path (str/Path), source_path (str/Path), parquet_path (optional) | `DatasetMeta` | Register a dataset from YAML config + source data |
| `get_services().get_dataset(dataset_id)` | dataset_id (str) | `DatasetMeta \| None` | Get metadata for a dataset |
| `get_services().profile(dataset_id)` | dataset_id (str) | `dict` | Profile a dataset |
| `get_services().train(dataset_id, target_name, model_type, model_id?)` | dataset_id, target_name, model_type, model_id (optional) | `dict` (model_id + metrics) | Train a new model |
| `get_services().list_models()` | *(none)* | `list[ModelMetadata]` | List all trained models |
| `get_services().get_model_metrics(model_id)` | model_id (str) | `dict` | Get model evaluation metrics |
| `get_services().predict(model_id, entity_id?, timestamp?, records?)` | model_id, entity_id, timestamp, records (all optional except model_id) | `dict` (probability + prediction) | Make a prediction |
| `get_services().explain(model_id, entity_id?, timestamp?, records?, top_k?)` | model_id, entity_id, timestamp, records, top_k (default 10) | `dict` (probability + top_drivers) | Explain a prediction |
| `get_services().slice(dataset_id, dimensions, metric, aggregation?, filters?, limit?)` | dataset_id, dimensions, metric, aggregation, filters, limit | `list[dict]` | Slice data by dimensions |
| `get_services().compare(dataset_id, dimension, metric, aggregation?, filters?)` | dataset_id, dimension, metric, aggregation, filters | `list[dict]` | Compare a metric across dimension values |
| `get_services().find_interesting_slices(dataset_id, metric, dimensions?, min_sample_size?, max_dimensions?, aggregation?, filters?, top_k?)` | dataset_id, metric, dimensions, min_sample_size (default 100), max_dimensions (default 3), aggregation, filters, top_k (default 20) | `list[dict]` | Find slices with the largest deviation from baseline |
| `get_services().sample_rows(dataset_id, n?, filters?, columns?, sort_by?, descending?)` | dataset_id, n (default 10, max 1000), filters, columns, sort_by, descending (default False) | `dict` (dataset_id, row_count, returned, columns, rows) | Return up to N raw rows, optionally filtered |
| `get_services().evaluate_model(model_id, sample_size?, filters?)` | model_id, sample_size (default 50, max 1000), filters (optional dict) | `dict` (model_id, dataset_id, eval_row_count, metrics, filtered_metrics?, predictions_sample) | Evaluate a model against the held-out evaluation partition. Optional filters return both overall and filtered metrics. |
| `get_services().find_evaluation_slices(model_id, metric?, dimensions?, min_sample_size?, max_dimensions?, top_k?)` | model_id, metric (default "abs_error"), dimensions (auto-detected if None), min_sample_size (default 50), max_dimensions (default 3), top_k (default 20) | `list[dict]` (dimensions, values, row_count, metric_value, overall_baseline, difference) | Find populations where model error deviates most from the overall average |

### What's NOT exposed

| Capability | Available via | NOT available via |
|------------|--------------|-------------------|
| Train a model | REST, Python, Web UI | MCP (by design — training is heavyweight) |
| Register a dataset | REST, Python, Web UI | MCP (by design — registration is an admin operation) |
| Delete a dataset | REST, Python | MCP |
| Delete a model | REST, Python | MCP |
| Arbitrary Python execution | *(nowhere)* | MCP (by design — only the 12 defined tools) |

---

## 16. Semiconductor Yield Dataset & Models

The second reference dataset in Anistroph — a wafer-level yield regression
problem from synthetic semiconductor manufacturing data.

### Overview

| Property | Value |
|----------|-------|
| Dataset ID | `semiconductor_yield` |
| Dataset name | Semiconductor Wafer Yield |
| Entity key | `wafer_id` |
| Time key | `timestamp` (used for chronological splitting) |
| Row count | 30,000 wafers |
| Columns | 29 |
| Target | `wafer_yield` (regression, 0.0-1.0) |
| Split | Chronological - 70% train / 15% validation / 15% test |

### Data generation

```bash
python scripts/generate_semiconductor_yield_data.py --wafers 30000
```

Output: `data/semiconductor_yield/data.parquet`

Each row represents one completed wafer with:
- **Hierarchy:** lot_id -> wafer_id
- **Categorical context:** product_id, fab_id, process_route
- **Etch process:** etch_tool, etch_chamber, etch_recipe + 7 numeric measurements
- **Deposition process:** deposition_tool, deposition_chamber, deposition_recipe + 5 numeric measurements
- **Lithography:** exposure_dose, focus_offset
- **Maintenance:** maintenance_age_etch, maintenance_age_deposition
- **Target:** wafer_yield (good_dies / tested_dies)

### Injected yield relationships

The synthetic generator injects learnable but imperfect relationships:

| Condition | Yield Effect |
|-----------|-------------|
| Baseline | ~96-98% |
| ETCH_02 | Small reduction (~1%) |
| CH_B | Small reduction (~0.6%) |
| High etch_temperature_std (>2.5) | Small reduction (~0.5%) |
| ETCH_02 + CH_B | Larger reduction (~3.5%) |
| ETCH_02 + CH_B + high temp_std | Large reduction (~7-9%) |
| PROD_A + RECIPE_B | Small reduction |
| DEP_03 + high pressure_std | Small reduction |
| ETCH_02 + old maintenance (>350h) | Small reduction |
| ROUTE_3 + high temp_std | Small reduction |

No single feature perfectly determines yield - the model must learn
combinations and interactions.

### Trained models

Two regression models are trained on this dataset:

#### wafer-yield-xgb-v001 (primary)

| Property | Value |
|----------|-------|
| Model ID | `wafer-yield-xgb-v001` |
| Model type | XGBoost Regressor |
| Features | 39 (23 one-hot categorical + 16 numeric) |

**Evaluation metrics (test set):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | 0.0065 | Average error ~0.65 percentage points |
| RMSE | 0.0082 | Root mean square error ~0.82 pp |
| R2 | 0.816 | Explains ~82% of yield variance |
| Median abs error | 0.0054 | Half of predictions are within 0.54 pp |
| 95th pct abs error | 0.016 | 95% of predictions within 1.6 pp |
| Baseline MAE | 0.0139 | Mean-predictor MAE (model beats this 2x) |
| Baseline R2 | ~0.0 | Mean predictor explains nothing |

#### wafer-yield-linear-v001 (baseline)

| Property | Value |
|----------|-------|
| Model ID | `wafer-yield-linear-v001` |
| Model type | Linear Regression (Ridge) |

**Evaluation metrics (test set):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | 0.0090 | Average error ~0.90 pp |
| RMSE | 0.0119 | Root mean square error ~1.19 pp |
| R2 | 0.610 | Explains ~61% of yield variance |

XGBoost significantly outperforms the linear baseline (R2 0.82 vs 0.61),
confirming the injected nonlinear interactions are learnable.

### Engineered features (39 total)

**Categorical (one-hot encoded, 23 features):**

| Source column | Encoded features |
|---------------|-----------------|
| product_id | PROD_A, PROD_B, PROD_C |
| fab_id | FAB_01, FAB_02 |
| process_route | ROUTE_1, ROUTE_2, ROUTE_3 |
| etch_tool | ETCH_01, ETCH_02, ETCH_03 |
| etch_chamber | CH_A, CH_B |
| etch_recipe | RECIPE_A, RECIPE_B, RECIPE_C |
| deposition_tool | DEP_01, DEP_02, DEP_03 |
| deposition_chamber | DCH_A, DCH_B |
| deposition_recipe | DEP_RECIPE_A, DEP_RECIPE_B |

**Numeric (16 features, all "current" - no rolling windows):**

| Feature | Description |
|---------|-------------|
| etch_temperature_mean_current | Etch process mean temperature |
| etch_temperature_std_current | Etch temperature variability (key driver) |
| etch_pressure_mean_current | Etch process mean pressure |
| etch_pressure_std_current | Etch pressure variability |
| etch_gas_flow_mean_current | Etch gas flow rate |
| etch_rf_power_mean_current | Etch RF power |
| etch_process_time_current | Etch process duration |
| deposition_temperature_mean_current | Deposition mean temperature |
| deposition_temperature_std_current | Deposition temperature variability |
| deposition_pressure_mean_current | Deposition mean pressure |
| deposition_pressure_std_current | Deposition pressure variability |
| deposition_process_time_current | Deposition duration |
| exposure_dose_current | Lithography exposure dose |
| focus_offset_current | Lithography focus offset |
| maintenance_age_etch_current | Hours since etch tool maintenance |
| maintenance_age_deposition_current | Hours since deposition tool maintenance |

### Sample predictions

| Wafer | Predicted Yield | Actual Yield | Error |
|-------|----------------|-------------|-------|
| WAFER_000001 | 0.9616 | 0.9526 | +0.009 |
| WAFER_005000 | 0.9688 | 0.9867 | -0.018 |
| WAFER_010000 | 0.9711 | 0.9803 | -0.009 |
| WAFER_015000 | 0.9312 | 0.9317 | -0.0005 |

### Interesting slice discovery

`find_interesting_slices` searches 1, 2, and 3-dimensional combinations
of categorical columns and ranks by deviation from the overall yield
baseline (minimum 100 rows per slice).

Top slices for semiconductor yield:

| Dimensions | Values | Rows | Mean Yield | Diff |
|-----------|--------|------|-----------|------|
| etch_tool, etch_chamber, etch_recipe | ETCH_02, CH_B, RECIPE_B | 1,669 | 92.4% | -3.5pp |
| etch_tool, etch_chamber, deposition_tool | ETCH_02, CH_B, DEP_03 | 1,654 | 92.4% | -3.4pp |
| product_id, etch_tool, etch_chamber | PROD_A, ETCH_02, CH_B | 1,676 | 92.5% | -3.4pp |

The ETCH_02 + CH_B interaction is correctly identified as the worst yield
combination - matching the injected hidden relationship.

### How to use

**Generate data and train (admin):**
```bash
# Generate synthetic data
python scripts/generate_semiconductor_yield_data.py --wafers 30000

# Register dataset
python -c "
from backend.services import get_services
svc = get_services()
svc.register_dataset_from_config(
    'datasets/semiconductor_yield/dataset.yaml',
    'data/semiconductor_yield/data.parquet',
)
"

# Train XGBoost regressor
python scripts/train_model.py --dataset semiconductor_yield \
  --model-type xgboost_regressor --model-id wafer-yield-xgb-v001

# Train linear baseline
python scripts/train_model.py --dataset semiconductor_yield \
  --model-type linear_regression --model-id wafer-yield-linear-v001
```

**Predict via MCP (Claude Desktop, Claude CLI, or any stdio MCP client):**
> "List all Anistroph models"
> "Predict wafer yield for WAFER_015000 using model wafer-yield-xgb-v001"
> "Explain that prediction - what are the top drivers?"
> "Find the worst yield combinations in the semiconductor dataset"
> "Show yield by etch tool and chamber"

**Predict via REST:**
```bash
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{"model_id": "wafer-yield-xgb-v001", "entity_id": "WAFER_015000"}'
```

**Predict via Python:**
```python
from backend.services import get_services
svc = get_services()

# Predict
pred = svc.predict("wafer-yield-xgb-v001", entity_id="WAFER_015000")
print(f"Predicted: {pred['predicted_yield']:.4f}, Actual: {pred['actual_yield']:.4f}")

# Find interesting slices
slices = svc.find_interesting_slices("semiconductor_yield", "wafer_yield", top_k=10)
for s in slices:
    print(f"  {s['values']} rows={s['row_count']} yield={s['metric_value']:.4f} diff={s['difference']:.4f}")
```

### Insights

- **R2 0.82** means the XGBoost model explains 82% of yield variance -
  strong for synthetic manufacturing data with injected noise.
- **MAE 0.65 pp** means average prediction error is less than 1 percentage
  point - useful for screening wafers or flagging at-risk lots.
- **XGBoost >> Linear** (R2 0.82 vs 0.61) confirms the yield relationships
  are nonlinear interactions, not simple additive effects.
- **ETCH_02 + CH_B** is the dominant yield driver - the interesting slice
  finder correctly identifies this as the worst combination.
- **Maintenance age** has a gradual continuous effect - older maintenance
  on ETCH_02 compounds the chamber effect.
- **Focus offset** has a small but consistent effect - absolute offset
  matters regardless of direction.

### Limitations

- Synthetic data - real semiconductor yield data has more complex
  spatial, temporal, and equipment interactions.
- Wafer-level only - die-level spatial patterns are not modeled in v0.1.
- No SHAP - explanation uses XGBoost feature importance x feature value,
  not SHAP values (SHAP can be added later).
- Predictions are model estimates - actual yield depends on many factors
  not captured in the synthetic features.

## 17. Bay Area Home Prices Dataset

The third reference dataset in Anistroph — a home-price regression problem
from synthetic Bay Area listing data. Price is driven primarily by square
footage, with city / zip code as the dominant price driver. The same
slicing, profiling, training, and prediction tools apply unchanged.

### Overview

| Property | Value |
|----------|-------|
| Dataset ID | `home_prices` |
| Dataset name | Bay Area Home Prices |
| Entity key | `property_id` |
| Time key | `timestamp` (used for chronological splitting) |
| Row count | 20,000 listings |
| Columns | 11 |
| Target | `price` (regression, USD) |
| Split | Chronological — 70% train / 15% validation / 15% test |

### Data generation

```bash
python scripts/generate_home_prices_data.py --homes 20000
```

Output: `data/home_prices/data.parquet`

Each row represents one home listing with:
- **Identifiers:** timestamp, property_id
- **Location:** city (San Jose / Los Gatos / Saratoga), zip_code (15 zips total)
- **Property:** sqft (1500–3800), bedrooms (2–6), bathrooms (1.0–5.0 in half-baths)
- **Land:** lot_size_sqft
- **Age:** year_built (1950–2024)
- **Parking:** garage (0–3 stalls)
- **Target:** price (USD)

### Pricing hierarchy

The generator injects a clear city-level price hierarchy, calibrated so
San Jose ~1600 sqft homes median ~$1.8MM:

| City | Median $/sqft | Median Price | Share of Rows |
|------|---------------|--------------|---------------|
| Saratoga | ~$1,600 | ~$4.08M | 12% |
| Los Gatos | ~$1,320 | ~$3.34M | 18% |
| San Jose | ~$1,030 | ~$2.62M | 70% |

Within San Jose, 11 zip codes span ~$980–$1,250/sqft (95122 lowest,
95129 highest). Per-zip variation means no single categorical perfectly
determines price.

### Injected price relationships

| Driver | Effect |
|--------|--------|
| City / zip_code | Dominant — sets base $/sqft |
| sqft | Primary continuous driver, with diminishing $/sqft at larger sizes |
| Bedrooms (>3) | ~1.5% premium per extra bedroom |
| Bathrooms (>2) | ~1% premium per extra full bath |
| year_built (≥2000) | Up to ~6% premium for newer homes |
| year_built (<1960) | Up to ~3% discount for older homes |
| lot_size_sqft (>6000) | ~0.4% premium per extra 1,000 sqft of lot |
| garage (2–3 stalls) | ~1–2.5% premium over 1-stall |
| Zip-level noise | ±4% so the relationship is learnable but imperfect |

### How to use

**Generate data and register (admin):**
```bash
# Generate synthetic data
python scripts/generate_home_prices_data.py --homes 20000

# Register dataset (partitions into train/eval/validate automatically)
python -c "
from backend.services import get_services
svc = get_services()
svc.register_dataset_from_config(
    'datasets/home_prices/dataset.yaml',
    'data/home_prices/data.parquet',
)
"

# Train XGBoost regressor
python scripts/train_model.py --dataset home_prices \
  --model-type xgboost_regressor --model-id home-price-xgb-v001
```

**Analyze via MCP (Claude Desktop, Claude CLI, or any stdio MCP client):**
> "List all Anistroph datasets"
> "Show median price by city in the home_prices dataset"
> "Find the most expensive zip codes in home_prices"
> "Slice home_prices by city and bedrooms, median price"
> "Find interesting slices in home_prices for price"

**Analyze via REST:**
```bash
# Slice by city
curl -X POST http://localhost:9500/analysis/slice \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "home_prices", "dimensions": ["city"], "metric": "price", "agg": "median"}'

# Sample rows
curl -X POST http://localhost:9500/datasets/home_prices/rows \
  -H "Content-Type: application/json" \
  -d '{"n": 5, "filters": {"city": "Saratoga"}, "sort_by": "price", "descending": true}'
```

**Analyze via Python:**
```python
from backend.services import get_services
svc = get_services()

# Slice by city
slices = svc.slice_data("home_prices", dimensions=["city"], metric="price", agg="median")
for s in slices:
    print(f"  {s['city']}: median=${s['price_median']:,.0f} (n={s['row_count']})")

# Find interesting slices
slices = svc.find_interesting_slices("home_prices", "price", top_k=10)
for s in slices:
    print(f"  {s['values']} rows={s['row_count']} price=${s['metric_value']:,.0f} diff=${s['difference']:,.0f}")
```

### Insights

- **City is the dominant driver** — Saratoga commands ~55% higher $/sqft
  than San Jose, with Los Gatos in between.
- **Sqft has diminishing returns** — larger homes have slightly lower
  $/sqft, reflecting real-world market behavior.
- **Zip-level variation** within San Jose (95122 vs 95129) gives the
  interesting-slice finder categorical signal to discover.
- **No single feature perfectly determines price** — the zip-level noise
  and interaction of bedrooms/baths/lot/year_built make this a realistic
  regression problem.

### Limitations

- Synthetic data — real Bay Area prices depend on schools, views, lot
  shape, condition, market timing, and many other factors not captured.
- No macroeconomic dynamics — listing timestamps span ~18 months but
  prices are not modeled as a time series.
- Zip code boundaries are illustrative, not authoritative.
