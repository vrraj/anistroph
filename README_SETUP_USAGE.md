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
| **Scope** | 13 runtime tools | 13 runtime REST endpoints |
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
> "Predict wafer yield for WAFER_015000 using model wafer-yield-xgboost"
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
> "Predict wafer yield for WAFER_015000 using model wafer-yield-xgboost"
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

> **The generation script and `dataset.yaml` are independent files.** The
> script hard-codes the columns it writes (it builds a `pl.DataFrame({...})`
> with whatever columns the author chose). The YAML *describes* the schema of
> that output so the ML pipeline knows how to interpret it. If you add a
> column to a generator script, you must also add it to the YAML's `columns:`
> block — otherwise the pipeline will ignore it. The YAML is never read by
> the generator.

---

## 2a. Author the `dataset.yaml` Configuration

Before registering a dataset, you author a `dataset.yaml` that declares the
schema, the model inputs, and the target. This is the single source of truth
the generic ML pipeline consults — no domain knowledge (e.g. "temperature
means degrees Celsius") lives in the engine itself.

A YAML has four required blocks and one optional block:

```yaml
dataset:        # schema + identifiers + split strategy
  ...
columns:        # per-column type and role (inside `dataset:`)
  ...
features:       # the actual model inputs (top-level)
  ...
target:         # what to predict (top-level)
split:          # train/eval partitioning (inside `dataset:`, optional)
  ...
```

### The `dataset:` block

```yaml
dataset:
  dataset_id: home_prices          # unique id, used in API calls and model names
  name: Bay Area Home Prices       # human-readable name
  entity_key: property_id          # column that uniquely identifies one entity
  time_key: timestamp              # optional — set only for temporal datasets
```

- **`entity_key`** (required): the column identifying one entity (one wafer,
  one machine, one property). Used for grouping rolling windows and for
  `entity_id` lookups at inference time.
- **`time_key`** (optional): if present, the dataset is treated as *temporal*
  — rolling-window transforms become available, splits are chronological by
  default, and inference can be called with `(entity_id, timestamp)` to
  predict from history. If absent, the dataset is *non-temporal* — only
  `current` and `categorical` transforms are meaningful, splits are random,
  and inference requires either `entity_id` (single-row lookup) or `records`
  (raw values supplied by the caller).

### The `columns:` block (inside `dataset:`)

Declares every column the source data contains. Each entry has a `type` and a
`role`. Columns present in the data but missing from this block are silently
ignored by the pipeline.

```yaml
columns:
  timestamp:
    type: timestamp
    role: identifier
  property_id:
    type: categorical
    role: identifier
  city:
    type: categorical
    role: feature
  sqft:
    type: numeric
    role: feature
  price:
    type: numeric
    role: target
```

**Column types** (`type:`):

| Type | Meaning |
|------|---------|
| `numeric` | Continuous or discrete number (temperature, price, sqft) |
| `categorical` | String label with finite categories (city, tool_id, machine_type) |
| `boolean` | True/false flag (failure event) |
| `timestamp` | Datetime column used for time-based operations |
| `string` | Free-form text — rarely used as a feature |

**Column roles** (`role:`):

| Role | Meaning | Used by training? |
|------|---------|-------------------|
| `identifier` | Entity key, time key, or other IDs (lot_id, wafer_id) | No — used for joins and lookups only |
| `feature` | A candidate input column | Only if also listed in `features:` (see below) |
| `target` | The column the model predicts (or its source) | Yes — as the label |
| `event` | A boolean event flag used to construct a `future_event` target | Yes — as the target source |
| `metadata` | Informational only (e.g. `failure_mode`) | No — available in `sample_rows` but never fed to the model |
| `ignore` | Explicitly excluded | No |

> **`role: feature` is necessary but not sufficient.** A column with
> `role: feature` is *eligible* to be a model input, but it only becomes one
> if it is also listed in the top-level `features:` block. The `columns:`
> block declares schema; the `features:` block declares model inputs. This
> separation lets you keep a column in the schema (for profiling, slicing,
> and `sample_rows`) without feeding it to the model.

### The `features:` block (top-level)

This is the contract between training and inference. Every entry here becomes
a model input. Each entry maps a feature name to a source column plus one or
more **transforms** that turn the raw column into engineered feature(s).

```yaml
features:
  city:                            # feature name (also the output column base name)
    column: city                   # source column to read from
    transforms:
      - categorical                # one-hot encode the categories
  sqft:
    column: sqft
    transforms:
      - current                    # pass through the raw value unchanged
  temperature:                     # temporal example — multiple transforms
    column: temperature
    transforms:
      - current
      - mean:
          windows: [1h, 6h]
      - std:
          windows: [1h, 6h]
      - slope:
          windows: [6h]
```

**Available transforms:**

| Transform | Output | Applies to | Description |
|-----------|--------|------------|-------------|
| `current` | `{column}_current` | numeric, categorical | Pass through the raw value at the current row. For numeric columns this is just the value; the `_current` suffix is added unless the feature name already matches. This is the only transform that makes sense for non-temporal datasets. |
| `categorical` | `{column}__{category}` (one column per learned category) | categorical | One-hot encode the column. Categories are **fit on training data only** and persisted in `FeatureMetadata`, so inference applies the identical encoding. Unseen categories at inference become all-zeros. Supports `min_frequency` (default 1) to drop rare categories. |
| `mean` / `min` / `max` / `std` / `median` | `{column}_{op}_{window}` | numeric, temporal only | Rolling aggregate over a trailing time window, grouped by `entity_key`. Leakage-safe: only uses rows up to and including the current timestamp. Requires `windows: ["1h", "6h", ...]`. |
| `slope` | `{column}_slope_{window}` | numeric, temporal only | Rolling linear-regression slope (cov(t,y)/var(t)) over the trailing window. Captures trend direction. |
| `delta` | `{column}_delta_{window}` | numeric, temporal only | Current value minus the minimum value in the trailing window. Captures recent deviation. |
| `hour_of_day` | `hour_of_day` | timestamp | Extract hour-of-day (0–23) from the `time_key`. Useful for capturing diurnal cycles. |
| `day_of_week` | `day_of_week` | timestamp | Extract day-of-week (0–6) from the `time_key`. |
| `elapsed_time` | `elapsed_time` | timestamp | Seconds elapsed since the entity's first observation. Captures aging / cumulative usage. |

**Transform syntax:** a transform can be written as a bare string
(`- current`) or as a mapping with parameters (`- mean: {windows: [1h, 6h]}`).
The bare form is shorthand for `{op: <name>}`.

> **The `features:` block is the inference contract.** For non-temporal
> datasets, a caller sending `records` to `/predictions` must include every
> source column listed here (e.g. `city`, `sqft`, `bedrooms`, ...). The
> Feature Engine then applies the same transforms using the persisted
> `FeatureMetadata` from training. The caller never constructs one-hot
> vectors or rolling aggregates — only raw source values.

### The `target:` block (top-level)

Declares what the model predicts. Drives automatic model selection and the
evaluation metrics reported after training.

```yaml
target:
  name: price                      # internal target name (returned in predictions)
  type: regression                 # task type — see table below
  source_column: price             # the raw column to use as the label
```

For `future_event` targets (binary classification from a boolean event
column over a time horizon):

```yaml
target:
  name: failure_within_horizon
  type: future_event
  source_column: failure           # boolean event column
  horizon: 24h                     # predict whether event occurs within 24h
  positive_class: 1                # value indicating the positive class
```

**Target types:**

| `type` | Meaning | Default model | Evaluation metrics |
|--------|---------|---------------|--------------------|
| `regression` | Predict a continuous number | `xgboost_regressor` | MAE, RMSE, R², MSE, MAPE, max_error |
| `classification` | Binary classification (canonical name) | `xgboost` | ROC-AUC, PR-AUC, F1, precision, recall |
| `binary` | Alias for `classification` (legacy) | `xgboost` | same as classification |
| `future_event` | Classification with a time horizon (legacy) | `xgboost` | same as classification |

`classification` is the canonical name in v0.1. `binary` and `future_event`
are kept as aliases for backward compatibility with existing configs.

### The `split:` block (inside `dataset:`, optional)

Controls how the data is partitioned at registration time. Already covered in
§3 ("Split configuration"), but included here for completeness:

```yaml
split:
  strategy: chronological   # "chronological" (temporal) or "random" (non-temporal)
  train: 0.80
  validation: 0.0           # reserved for validate-during-training partition
  test: 0.20                # becomes the held-out evaluation partition
```

If omitted, defaults are read from `.env` (`TRAIN_DATASET_PCT`,
`EVAL_DATASET_PCT`, `VALIDATE_DATASET_PCT`). Set `train: 1.0` to skip
partitioning entirely (single-file mode).

### Worked example: non-temporal regression

`datasets/home_prices/dataset.yaml` — one row per home listing, predict
`price` from raw attributes (no rolling windows):

```yaml
dataset:
  dataset_id: home_prices
  name: Bay Area Home Prices
  entity_key: property_id
  time_key: timestamp              # present for chronological splitting, but
                                   # no rolling transforms used → effectively
                                   # non-temporal for feature purposes
  columns:
    timestamp:     {type: timestamp, role: identifier}
    property_id:   {type: categorical, role: identifier}
    city:          {type: categorical, role: feature}
    zip_code:      {type: categorical, role: feature}
    sqft:          {type: numeric, role: feature}
    bedrooms:      {type: numeric, role: feature}
    bathrooms:     {type: numeric, role: feature}
    lot_size_sqft: {type: numeric, role: feature}
    year_built:    {type: numeric, role: feature}
    garage:        {type: numeric, role: feature}
    price:         {type: numeric, role: target}
  split:
    strategy: chronological
    train: 0.70
    validation: 0.15
    test: 0.15

features:
  city:          {column: city, transforms: [categorical]}
  zip_code:      {column: zip_code, transforms: [categorical]}
  sqft:          {column: sqft, transforms: [current]}
  bedrooms:      {column: bedrooms, transforms: [current]}
  bathrooms:     {column: bathrooms, transforms: [current]}
  lot_size_sqft: {column: lot_size_sqft, transforms: [current]}
  year_built:    {column: year_built, transforms: [current]}
  garage:        {column: garage, transforms: [current]}

target:
  name: price
  type: regression
  source_column: price
```

Inference contract: send `records` with `city`, `zip_code`, `sqft`,
`bedrooms`, `bathrooms`, `lot_size_sqft`, `year_built`, `garage`. The engine
one-hot encodes the two categoricals and passes the numerics through.

### Worked example: temporal classification with rolling windows

`datasets/predictive_maintenance/dataset.yaml` — one row per machine
observation at 5-min intervals, predict failure within 24h:

```yaml
dataset:
  dataset_id: predictive_maintenance
  entity_key: machine_id
  time_key: timestamp              # temporal → rolling transforms available
  columns: ...
  split: {strategy: chronological, train: 0.70, validation: 0.15, test: 0.15}

features:
  temperature:
    column: temperature
    transforms:
      - current                    # raw value at this observation
      - mean: {windows: [1h, 6h]}  # trailing 1h and 6h averages
      - std:  {windows: [1h, 6h]}  # trailing volatility
      - slope: {windows: [6h]}     # is it trending up/down?
  vibration:
    column: vibration
    transforms:
      - current
      - mean: {windows: [1h, 6h]}
      - max:  {windows: [6h]}
      - std:  {windows: [1h, 6h]}
      - slope: {windows: [6h]}
  # ... pressure, current, voltage, rpm, flow_rate, maintenance_age_hours,
  #     operating_hours (all `current`), machine_type (`categorical`)

target:
  name: failure_within_horizon
  type: future_event
  source_column: failure           # boolean event column
  horizon: 24h
  positive_class: 1
```

Inference contract: send `(entity_id, timestamp)` — e.g.
`{"model_id": "...", "entity_id": "MACHINE_0042", "timestamp":
"2025-06-01T00:00:00"}`. Anistroph loads that machine's history up to the
timestamp from the parquet, builds the rolling features using the persisted
`FeatureMetadata`, and predicts. The caller supplies no feature values.

### Multi-target datasets

A single source parquet can back multiple targets by creating separate YAML
files that all point at the same data. Each YAML has a different `target:`
block and may have a different `features:` block. See §14 (predictive
maintenance) and §16 (semiconductor) for the reference multi-target setups.

---

## 3. Register a Dataset

Registration reads the dataset config (`datasets/predictive_maintenance/dataset.yaml`),
validates the data against the spec, converts CSV to Parquet, partitions the
data into train/evaluation/validate sets, and stores metadata in the dataset
registry.

### What happens during registration

```
register_dataset_from_config()
    │
    ├── 1. Load YAML config
    │      → DatasetSpec (entity_key, time_key, split strategy)
    │      → FeatureSpec (which columns are features, their types)
    │      → TargetSpec (target column, task type: regression/classification)
    │
    ├── 2. Ingest source data
    │      → Read CSV or Parquet
    │      → Validate columns against spec
    │      → Write full Parquet: data/processed/{dataset_id}.parquet
    │
    ├── 3. Profile the full dataset
    │      → Column types, unique counts, top values, time range
    │
    ├── 4. Partition into train / evaluation / validate
    │      → Resolve split percentages (YAML overrides .env defaults)
    │      → Temporal dataset → chronological split (oldest→train, newest→eval)
    │      → Non-temporal dataset → random split with fixed seed
    │      → Write partition files:
    │         data/processed/{dataset_id}.train.parquet
    │         data/processed/{dataset_id}.evaluation.parquet
    │         data/processed/{dataset_id}.validate.parquet
    │
    └── 5. Register in dataset registry
           → Store all paths + metadata in artifacts/dataset_registry.json
           → partitioned=True
           → train_parquet_path, eval_parquet_path, validate_parquet_path
```

### How evaluation finds the right dataset

When you select a model in the Evaluation tab and click Evaluate, the system
resolves the evaluation data through a 3-step lookup:

```
1. Model metadata (stored at training time)
   → model.dataset_id = "semiconductor_yield"

2. Dataset registry (stored at registration time)
   → dataset.eval_parquet_path = "data/processed/semiconductor_yield.evaluation.parquet"

3. Evaluation runner loads that eval parquet
   → Builds features using the model's persisted FeatureMetadata (no refit)
   → Runs inference with the persisted model
   → Computes metrics against known actuals
```

This is why evaluation "just works" — the model stores which dataset it was
trained on, and the dataset registry stores where the eval partition lives.
No manual configuration needed. The same chain applies to `predict`,
`find_evaluation_slices`, and `explain_prediction`.

**Key guarantee:** Training loads only `train.parquet`; evaluation loads only
`evaluation.parquet`. The two never overlap — the partition is done
chronologically (temporal) or with a fixed random seed (non-temporal) at
registration time, before any model is trained.

### Via Python

```python
from backend.services import get_services

svc = get_services()
meta = svc.register_dataset_from_config(
    "datasets/predictive_maintenance/dataset.yaml",
    "data/synthetic/predictive_maintenance.csv",
)
print(f"Registered: {meta.dataset_id}, {meta.row_count} rows")
print(f"Train:  {meta.train_parquet_path}")
print(f"Eval:   {meta.eval_parquet_path}")
print(f"Validate: {meta.validate_parquet_path}")
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

Response includes `partitioned: true` and the paths to all three partition
files (`train_parquet_path`, `eval_parquet_path`, `validate_parquet_path`).

### Via web UI

1. Open http://localhost:9500
2. Go to the **Datasets** tab
3. Click **Register** (fields are pre-filled)
4. Click **Profile** to see dataset statistics

### Partition files

Every registered dataset is automatically partitioned into separate Parquet
files at registration time:

| File | Purpose | Used during training? |
|------|---------|----------------------|
| `{dataset_id}.train.parquet` | Model fitting | Yes — training loads only this file |
| `{dataset_id}.evaluation.parquet` | Held-out evaluation | Never during training — used post-training via Evaluation tab / `anistroph_evaluate_model` / `anistroph_find_evaluation_slices` |
| `{dataset_id}.validate.parquet` | Validation during training (threshold tuning, early stopping) | Yes — only if `VALIDATE_DATASET_PCT > 0` |
| `{dataset_id}.parquet` | Full dataset (all rows) | No — used for profiling, slicing, and `sample_rows` |

The partition files are created at registration time and never modified
afterward. Re-registering a dataset (e.g. after regenerating data) overwrites
them with the new data.

### Split configuration

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
to evaluation. This prevents time leakage: the model never sees future data
during training. **Non-temporal datasets** shuffle with a fixed seed before
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

The model type is **auto-selected from the dataset's task type** when
`model_type` is omitted:

| Task type (`target.type` in YAML) | Default model | Evaluation metrics |
|-----------------------------------|---------------|-------------------|
| `regression` | `xgboost_regressor` | MAE, MSE, RMSE, R², MAPE, max error |
| `classification` | `xgboost` | ROC-AUC, PR-AUC, precision, recall, F1 |
| `binary` (alias) | `xgboost` | same as classification |
| `future_event` (alias) | `xgboost` | same as classification |

Available model types (specify explicitly to override the default):
`xgboost`, `logistic_regression`, `xgboost_regressor`, `linear_regression`.

### Via Python

```python
from backend.services import get_services

svc = get_services()

# Auto-select model from task type (recommended)
# predictive_maintenance is classification → xgboost
result = svc.train(
    dataset_id="predictive_maintenance",
    target_name="failure_within_horizon",
    model_id="predictive-maintenance-xgboost",
)
print(f"Model type: {result['model_type']}")  # xgboost
print(f"ROC-AUC: {result['metrics']['roc_auc']:.3f}")

# Explicit model type (overrides auto-selection)
result = svc.train(
    dataset_id="semiconductor_yield",
    target_name="wafer_yield",
    model_type="linear_regression",
    model_id="semi-lr",
)
print(f"Model type: {result['model_type']}")  # linear_regression
```

### Via REST API

```bash
# Auto-select from task type (model_type omitted)
curl -X POST http://localhost:9500/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "predictive_maintenance",
    "target_name": "failure_within_horizon",
    "model_id": "predictive-maintenance-xgboost"
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

## 6. List & Delete Models

### Via Python

```python
from backend.services import get_services

models = get_services().list_models()
for m in models:
    print(f"{m.model_id}: {m.model_type}, dataset={m.dataset_id}")

# Delete a model (removes from registry + deletes artifact files)
get_services().delete_model("old-model-id")
```

### Via REST API

```bash
# List all models
curl http://localhost:9500/models

# Delete a model
curl -X DELETE http://localhost:9500/models/old-model-id
```

### Via web UI

Open the **Models** tab (`http://localhost:9500/#models`):
1. Click **Refresh** to list all registered models.
2. Each model card shows the model ID, type, task type, and dataset.
3. Click **Delete** to remove a model (with confirmation prompt).
4. Use the Model Details and Model Metrics cards to inspect individual models.

### Via MCP (Claude Desktop)

> "What models are available in Anistroph?"

---

## 7. Get Model Metrics

### Via Python

```python
from backend.services import get_services

metrics = get_services().get_model_metrics("predictive-maintenance-xgboost")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"PR-AUC: {metrics['pr_auc']:.3f}")
print(f"Confusion matrix: {metrics['confusion_matrix']}")
```

### Via REST API

```bash
curl http://localhost:9500/models/predictive-maintenance-xgboost/metrics
```

### Via MCP (Claude Desktop)

> "Show me the metrics for model predictive-maintenance-xgboost."

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
result = get_services().evaluate_model("wafer-yield-xgboost", sample_size=50)
metrics = result["metrics"]
print(f"R²: {metrics['r2']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"Eval rows: {result['eval_row_count']}")
print(f"Sample: {result['predictions_sample'][:3]}")

# Slice-level evaluation (filtered to a single city)
result = get_services().evaluate_model(
    "home-prices-xgboost",
    sample_size=50,
    filters={"city": "Saratoga"},
)
print(f"Overall MAPE: {result['metrics']['mape']:.2f}%")
print(f"Saratoga MAPE: {result['filtered_metrics']['mape']:.2f}%  (n={result['filtered_row_count']})")
```

### Via REST API

```bash
# Overall evaluation
curl -X POST http://localhost:9500/evaluations/wafer-yield-xgboost \
  -H "Content-Type: application/json" \
  -d '{"sample_size": 50}'

# Slice-level evaluation (filtered to Saratoga)
curl -X POST http://localhost:9500/evaluations/home-prices-xgboost \
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

> "Evaluate model wafer-yield-xgboost on the held-out set"
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
    "home-prices-xgboost",
    metric="abs_error",
    min_sample_size=50,
    top_k=20,
)
for s in slices:
    vals = " + ".join(f"{k}={v}" for k, v in s["values"].items())
    print(f"  {vals}: n={s['row_count']}, MAE={s['metric_value']:.0f}, diff={s['difference']:+.0f}")

# Use percentage error for price datasets (relative, not absolute)
slices = get_services().find_evaluation_slices(
    "home-prices-xgboost",
    metric="pct_error",
    top_k=10,
)

# Then drill into a specific slice with filtered evaluation
result = get_services().evaluate_model(
    "home-prices-xgboost",
    filters={"zip_code": "95071"},
)
print(f"Overall MAPE: {result['metrics']['mape']:.2f}%")
print(f"95071 MAPE:   {result['filtered_metrics']['mape']:.2f}%  (n={result['filtered_row_count']})")
```

### Via REST API

```bash
curl -X POST http://localhost:9500/evaluations/home-prices-xgboost/slices \
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

There are two prediction modes depending on whether the entity already
exists in the dataset or you're predicting for a brand-new entity:

**A. Entity lookup** (existing entity in the data): provide `model_id` +
`entity_id` (+ `timestamp` for temporal datasets). Anistroph loads the
entity's row(s) from the parquet, builds features using the persisted
`FeatureMetadata`, and returns the prediction. The caller supplies no
feature values — just an identifier.

**B. Records** (brand-new entity, not in the data): provide `model_id` +
`records` — a list of dicts with raw source-column values. The caller must
include every column listed in the model's `features:` block (see §2a).
The Feature Engine transforms them using the persisted metadata. The caller
never constructs engineered features (no one-hot vectors, no rolling
aggregates) — only raw source values.

### Via Python

```python
from backend.services import get_services

# A. Entity lookup (temporal dataset — entity + timestamp)
pred = get_services().predict(
    model_id="predictive-maintenance-xgboost",
    entity_id="TOOL_000",
    timestamp="2026-06-15T12:00:00",
)
print(f"Probability: {pred['probability']:.4f}")
print(f"Prediction: {pred['prediction']}")

# A. Entity lookup (non-temporal dataset — entity only)
pred = get_services().predict(
    model_id="wafer-yield-xgboost",
    entity_id="WAFER_015000",
)
print(f"Predicted yield: {pred['predicted_yield']:.4f}")
print(f"Actual yield:    {pred.get('actual_yield', 'N/A')}")

# B. Records (new wafer — raw source values for every feature column)
pred = get_services().predict(
    model_id="wafer-yield-xgboost",
    records=[{
        "product_id": "PROD_A", "fab_id": "FAB_01", "process_route": "ROUTE_1",
        "etch_tool": "ETCH_01", "etch_chamber": "CH_A", "etch_recipe": "RECIPE_A",
        "deposition_tool": "DEP_01", "deposition_chamber": "DEP_CH_A",
        "deposition_recipe": "DEP_RECIPE_A",
        "etch_temperature_mean": 85.0, "etch_temperature_std": 1.0,
        "etch_pressure_mean": 4.0, "etch_pressure_std": 0.1,
        "etch_gas_flow_mean": 100.0, "etch_rf_power_mean": 500.0,
        "etch_process_time": 60.0,
        "deposition_temperature_mean": 300.0, "deposition_temperature_std": 2.0,
        "deposition_pressure_mean": 2.0, "deposition_pressure_std": 0.05,
        "deposition_process_time": 120.0,
        "exposure_dose": 32.5, "focus_offset": 0.12,
        "maintenance_age_etch": 100.0, "maintenance_age_deposition": 50.0,
    }],
)
print(f"Predicted yields: {pred['predictions']}")

# B. Records (Stage A model — only 7 pre-etch features needed)
pred = get_services().predict(
    model_id="semiconductor_yield_stage_a-xgboost_regressor-20260817002238",
    records=[{
        "product_id": "PROD_A", "fab_id": "FAB_01", "process_route": "ROUTE_1",
        "etch_recipe": "RECIPE_A", "deposition_recipe": "DEP_RECIPE_A",
        "exposure_dose": 32.5, "focus_offset": 0.12,
    }],
)
print(f"Predicted yields: {pred['predictions']}")
```

### Via REST API

```bash
# A. Entity lookup (temporal)
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "predictive-maintenance-xgboost",
    "entity_id": "TOOL_000",
    "timestamp": "2026-06-15T12:00:00"
  }'

# A. Entity lookup (non-temporal)
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "wafer-yield-xgboost",
    "entity_id": "WAFER_015000"
  }'

# B. Records (new wafer — full feature set)
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "wafer-yield-xgboost",
    "records": [{
      "product_id": "PROD_A", "fab_id": "FAB_01", "process_route": "ROUTE_1",
      "etch_tool": "ETCH_01", "etch_chamber": "CH_A", "etch_recipe": "RECIPE_A",
      "deposition_tool": "DEP_01", "deposition_chamber": "DEP_CH_A",
      "deposition_recipe": "DEP_RECIPE_A",
      "etch_temperature_mean": 85.0, "etch_temperature_std": 1.0,
      "etch_pressure_mean": 4.0, "etch_pressure_std": 0.1,
      "etch_gas_flow_mean": 100.0, "etch_rf_power_mean": 500.0,
      "etch_process_time": 60.0,
      "deposition_temperature_mean": 300.0, "deposition_temperature_std": 2.0,
      "deposition_pressure_mean": 2.0, "deposition_pressure_std": 0.05,
      "deposition_process_time": 120.0,
      "exposure_dose": 32.5, "focus_offset": 0.12,
      "maintenance_age_etch": 100.0, "maintenance_age_deposition": 50.0
    }]
  }'

# B. Records (Stage A — only 7 pre-etch features)
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "semiconductor_yield_stage_a-xgboost_regressor-20260817002238",
    "records": [{
      "product_id": "PROD_A", "fab_id": "FAB_01", "process_route": "ROUTE_1",
      "etch_recipe": "RECIPE_A", "deposition_recipe": "DEP_RECIPE_A",
      "exposure_dose": 32.5, "focus_offset": 0.12
    }]
  }'
```

### Batch predict

```bash
curl -X POST http://localhost:9500/predictions/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"model_id": "predictive-maintenance-xgboost", "entity_id": "TOOL_000", "timestamp": "2026-06-15T12:00:00"},
    {"model_id": "predictive-maintenance-xgboost", "entity_id": "TOOL_001", "timestamp": "2026-06-15T12:00:00"}
  ]'
```

### Via MCP (Claude Desktop)

**Entity lookup** (wafer already in the data):

> "Predict wafer yield for WAFER_015000 using model wafer-yield-xgboost"

**Records** (brand-new wafer — supply raw values for every feature):

> "Predict wafer yield for a new wafer with: product_id=PROD_A,
> fab_id=FAB_01, process_route=ROUTE_1, etch_tool=ETCH_01,
> etch_chamber=CH_A, etch_recipe=RECIPE_A, deposition_tool=DEP_01,
> deposition_chamber=DEP_CH_A, deposition_recipe=DEP_RECIPE_A,
> etch_temperature_mean=85, etch_pressure_mean=4.0, exposure_dose=32.5,
> focus_offset=0.1, and all other features at typical values"

**Records** (Stage A model — only 7 pre-etch features):

> "Predict wafer yield for a new wafer using the stage A model with:
> product_id=PROD_A, fab_id=FAB_01, process_route=ROUTE_1,
> etch_recipe=RECIPE_A, deposition_recipe=DEP_RECIPE_A,
> exposure_dose=32.5, focus_offset=0.12"

**Discovery** (not sure which model or features to use):

> "What models are available for semiconductor yield prediction?"

Claude will call `anistroph_list_models` and show you the model IDs,
their datasets, and target names. Each model's `features:` block (in its
dataset YAML) defines exactly which source columns you must supply when
using `records` — see §2a for how to author the YAML.

---

## 9. Explain a Prediction

Returns the top contributing features using SHAP TreeExplainer (for XGBoost
models) or importance-weighted contributions (fallback). Explanations are
deterministic and model-derived — no LLM fabrication.

**SHAP explanation normalization:** When one-hot encoding expands a source
feature into multiple model features, the explanation layer aggregates SHAP
contributions back to the original source feature before returning them.
This means the caller sees `etch_tool = ETCH_02, impact = +0.0024` rather
than separate `etch_tool__ETCH_01 = 0`, `etch_tool__ETCH_02 = 1` entries.
See [SHAP explanation normalization](#shap-explanation-normalization) below
for details.

### Via Python

```python
from backend.services import get_services

# Entity lookup (existing wafer)
expl = get_services().explain(
    model_id="wafer-yield-xgboost",
    entity_id="WAFER_015000",
    top_k=10,
)
print(f"Predicted yield: {expl['predicted_yield']:.4f}")
for d in expl["top_drivers"]:
    print(f"  {d['feature']:25s} = {str(d['value']):15s} impact={d['impact']:+.6f}")

# Records-based (new wafer — works for models without rolling transforms)
expl = get_services().explain(
    model_id="semiconductor_yield_stage_a-xgboost_regressor-20260817002238",
    records=[{
        "product_id": "PROD_B", "fab_id": "FAB_02", "process_route": "ROUTE_2",
        "etch_recipe": "RECIPE_B", "deposition_recipe": "DEP_RECIPE_A",
        "exposure_dose": 24.9, "focus_offset": 0.035,
    }],
    top_k=10,
)
for d in expl["top_drivers"]:
    print(f"  {d['feature']:25s} = {str(d['value']):15s} impact={d['impact']:+.6f}")
```

### Via REST API

```bash
# Entity lookup
curl -X POST http://localhost:9500/predictions/explain \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "wafer-yield-xgboost",
    "entity_id": "WAFER_015000",
    "top_k": 10
  }'

# Records-based
curl -X POST http://localhost:9500/predictions/explain \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "semiconductor_yield_stage_a-xgboost_regressor-20260817002238",
    "records": [{
      "product_id": "PROD_B", "fab_id": "FAB_02", "process_route": "ROUTE_2",
      "etch_recipe": "RECIPE_B", "deposition_recipe": "DEP_RECIPE_A",
      "exposure_dose": 24.9, "focus_offset": 0.035
    }],
    "top_k": 10
  }'
```

### Via MCP (Claude Desktop)

**Entity lookup:**

> "Explain the wafer yield prediction for WAFER_015000 using model
> wafer-yield-xgboost. What are the top drivers?"

**Records-based:**

> "Explain the yield prediction for a new wafer with: product_id=PROD_B,
> fab_id=FAB_02, process_route=ROUTE_2, etch_recipe=RECIPE_B,
> deposition_recipe=DEP_RECIPE_A, exposure_dose=24.9, focus_offset=0.035
> using the stage A model"

### SHAP explanation normalization

When a categorical source column is one-hot encoded, the FeatureEngine
expands it into N binary model features using the naming convention
`{source}__{category}`. SHAP returns a separate impact value for each.
The explanation layer groups these back to the original source feature
so the caller sees a single human-readable entry per source column.

**How the mapping works:**

```
Source column:  etch_tool
Input value:    ETCH_02
                    ↓ (FeatureEngine one-hot encodes)
Model columns:  etch_tool__ETCH_01 = 0
                etch_tool__ETCH_02 = 1    ← active (value=1)
                etch_tool__ETCH_03 = 0
                    ↓ (SHAP computes per-column impacts)
SHAP values:    etch_tool__ETCH_01 → -0.0010
                etch_tool__ETCH_02 → +0.0024
                etch_tool__ETCH_03 → -0.0005
                    ↓ (grouping: split on "__", sum impacts, find value=1)
Explanation:    feature = "etch_tool"
                value   = "ETCH_02"        ← the active category
                impact  = +0.0009          ← sum of all three
```

**Response format:**

```json
{
  "feature": "etch_tool",
  "value": "ETCH_02",
  "impact": 0.0009,
  "detail": {
    "active_category": "ETCH_02",
    "categories": {
      "ETCH_01": {"value": 0.0, "impact": -0.0010},
      "ETCH_02": {"value": 1.0, "impact": 0.0024},
      "ETCH_03": {"value": 0.0, "impact": -0.0005}
    }
  }
}
```

The `detail` field retains the raw per-category SHAP values for debugging
but is not needed for interpretation. The top-level `feature`, `value`,
and `impact` are what the caller (or Claude) should use.

**Naming convention rules (to avoid issues):**

- One-hot columns use `{source}__{category}` with a double-underscore
  separator (e.g. `etch_tool__ETCH_02`)
- The grouping logic splits on `__` from the right (`rsplit("__", 1)`), so
  category values may contain `__` but **source column names must not**
- Passthrough (`current` transform) columns are named `{source}_current`
  in the model and displayed as `{source}` (suffix stripped) in the
  explanation
- Rolling-window columns are named `{source}_{op}_{window}` (e.g.
  `temperature_mean_6h`) and are not grouped (each is a distinct feature)
- Source column names in the YAML `columns:` block should avoid `__` to
  prevent ambiguity in the grouping logic

**Why this matters:** Without normalization, an LLM trying to explain SHAP
values sees separate entries for `etch_tool__ETCH_01 = 0`, `etch_tool__ETCH_02
= 1`, `etch_tool__ETCH_03 = 0` and must interpret the statistical effect of
each one-hot zero — leading to confusing statements like "not being ETCH_01
pushed yield up." With normalization, it sees `etch_tool = ETCH_02,
impact = +0.0009` — a single, unambiguous statement.

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
    model_id="predictive-maintenance-xgboost",
)
print(f"ROC-AUC: {result['metrics']['roc_auc']:.3f}")

# 4. Predict
pred = svc.predict("predictive-maintenance-xgboost", entity_id="TOOL_000", timestamp="2026-06-15T12:00:00")
print(f"Probability: {pred['probability']:.4f}")

# 5. Explain
expl = svc.explain("predictive-maintenance-xgboost", entity_id="TOOL_000", timestamp="2026-06-15T12:00:00", top_k=5)
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
- "Show me the metrics for model wafer-yield-xgboost"
- "Show me the metrics for model critical-dimension-xgboost"
- "Show me the metrics for model film-thickness-xgboost"
- "Show me the metrics for model predictive-maintenance-xgboost"
- "Show me the metrics for model rul-xgboost"
- "Show me the metrics for model maintenance-required-xgboost"
- "List models trained on semiconductor_yield"

**Prediction & explanation**
- "Predict failure probability for TOOL_000 at 2026-06-15T12:00:00 using model predictive-maintenance-xgboost"
- "Predict wafer yield for WAFER_015000 using model wafer-yield-xgboost"
- "Predict critical dimension for WAFER_015000 using model critical-dimension-xgboost"
- "Predict film thickness for WAFER_015000 using model film-thickness-xgboost"
- "Predict remaining useful life for TOOL_010 at 2026-07-15T12:00:00 using model rul-xgboost"
- "Predict maintenance required for TOOL_010 at 2026-07-15T12:00:00 using model maintenance-required-xgboost"
- "Predict wafer yield for a new wafer with: product_id=PROD_A, fab_id=FAB_01, process_route=ROUTE_1, etch_tool=ETCH_01, etch_chamber=CH_A, etch_recipe=RECIPE_A, deposition_tool=DEP_01, deposition_chamber=DEP_CH_A, deposition_recipe=DEP_RECIPE_A, etch_temperature_mean=85, etch_pressure_mean=4.0, exposure_dose=32.5, focus_offset=0.1, and all other features at typical values"
- "Predict wafer yield for a new wafer using the stage A model with: product_id=PROD_A, fab_id=FAB_01, process_route=ROUTE_1, etch_recipe=RECIPE_A, deposition_recipe=DEP_RECIPE_A, exposure_dose=32.5, focus_offset=0.12"
- "What models are available for semiconductor yield prediction?"
- "What inputs does the wafer-yield-xgboost model need for prediction?"
- "What inputs does the stage A model need for prediction?"
- "Explain that prediction — what are the top drivers?"
- "Explain the critical dimension prediction for WAFER_015000 with top_k=10"
- "Explain the film thickness prediction for WAFER_015000 using model film-thickness-xgboost"
- "Explain the RUL prediction for TOOL_010 at 2026-07-15T12:00:00 using model rul-xgboost"

**Held-out evaluation**
- "Evaluate model predictive-maintenance-xgboost against the held-out evaluation set"
- "Run evaluation on wafer-yield-xgboost and show me 20 prediction-vs-actual rows"
- "What's the MAE and R² for model critical-dimension-xgboost on the evaluation partition?"
- "Evaluate model film-thickness-xgboost — how does it compare to the baseline?"
- "Evaluate the home price model filtered to San Jose only"
- "Compare MAPE for Saratoga vs Los Gatos vs San Jose in the home price model"
- "Evaluate the maintenance-required-xgboost model filtered to machine_type=TYPE_B"

**Error slice discovery**
- "Find the populations where the home price model has the worst prediction error"
- "Which wafer combinations does the semiconductor model struggle with most?"
- "Find the worst prediction-error slices for the critical-dimension-xgboost model by etch_tool and etch_recipe"
- "Find evaluation slices for the film-thickness-xgboost model by deposition_tool and deposition_recipe, with at least 100 rows"
- "Compare overall model error with the worst deposition_tool/recipe combinations for film-thickness-xgboost"
- "Find the top 10 evaluation slices for the rul-xgboost model ranked by MAE deviation"
- "Show me error slices by percentage error for the home price model"
- "Where is the model over-predicting vs under-predicting?"
- "Find slices where the model's log loss is highest"

**Multidimensional analysis across targets**
- "Slice the semiconductor_cd dataset by etch_tool and show the mean critical_dimension_nm for each"
- "Slice the predictive_maintenance_rul dataset by machine_type and show the mean remaining_useful_life_hours"
- "Compare the predictive_maintenance_maint dataset: maintenance_required=1 vs maintenance_required=0. Show temperature, vibration, and maintenance_age_hours"
- "Find interesting slices in the semiconductor_film_thickness dataset by deposition_tool. Which deposition tools have the most unusual film thickness?"
- "Find the most interesting slices in the semiconductor_cd dataset by etch_tool, etch_recipe, and product_id"
- "Slice the predictive_maintenance dataset by machine_type and failure_mode. How many failures of each mode does each machine type have?"

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
python -c "from backend.services import get_services; get_services().train('predictive_maintenance', 'failure_within_horizon', 'xgboost', model_id='predictive-maintenance-xgboost')"
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

## 14. Predictive Maintenance Datasets & Models

The predictive maintenance reference data is a tool sensor dataset with
**three targets**, each with its own dataset config and model:

| Dataset ID | Target | Type | What it predicts |
|------------|--------|------|-----------------|
| `predictive_maintenance` | `failure_within_horizon` | classification (future_event) | Will the tool fail within 24h? |
| `predictive_maintenance_rul` | `remaining_useful_life_hours` | regression | Hours until next failure |
| `predictive_maintenance_maint` | `maintenance_required` | classification | Does the tool need maintenance now? |

All three configs point to the same source Parquet file but define different
targets. The `failure_mode` column (NONE/THERMAL/PRESSURE/VIBRATION/POWER) is
stored as metadata for future multiclass classification support.

### Trained models

| Model ID | Target | Type | Key Metrics |
|----------|--------|------|-------------|
| `predictive-maintenance-xgboost` | failure_within_horizon | classification | ROC-AUC=0.85, F1=0.61 |
| `rul-xgboost` | remaining_useful_life_hours | regression | MAE=27.9h (R² low — RUL is hard from current state) |
| `maintenance-required-xgboost` | maintenance_required | classification | ROC-AUC=1.00, F1=0.94 |

### Training data details

**Source data:** `data/raw/predictive_maintenance.parquet`

| Property | Value |
|----------|-------|
| Machines | 50 (`TOOL_000` through `TOOL_049`) |
| Time range | June 1 – July 30, 2026 (5-minute intervals) |
| Row count | 864,000 |
| Failures | ~722 (~0.08% failure rate) |
| Split | Chronological — 70% train / 15% validation / 15% test |

**Machine types:**
- `TYPE_A` — baseline deterioration
- `TYPE_B` — faster deterioration (1.3x)
- `TYPE_C` — slower deterioration (0.8x)

**Sensor columns:**

| Column | Description |
|--------|-------------|
| temperature | Operating temperature (°C) |
| vibration | Vibration intensity (g) |
| pressure | System pressure (bar) |
| current | Electrical current (A) |
| voltage | Voltage (V) |
| rpm | Rotational speed (RPM) |
| flow_rate | Flow rate (L/min) |
| maintenance_age_hours | Hours since last maintenance |
| operating_hours | Total operating hours |

**Target/event columns:**
- `failure` (boolean): 1 if the tool failed at this timestamp
- `failure_mode` (categorical): NONE, THERMAL, PRESSURE, VIBRATION, or POWER
- `remaining_useful_life_hours` (numeric): hours until next failure (0 at failure, 9999 if no future failure)
- `maintenance_required` (boolean): 1 if maintenance age is high or risk is elevated

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
> "Show me the metrics for model predictive-maintenance-xgboost"
> "Predict failure probability for TOOL_010 at 2026-06-28T12:00:00 using model predictive-maintenance-xgboost"
> "Explain that prediction — what are the top drivers?"

**Via REST API:**
```bash
# Predict
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{"model_id": "predictive-maintenance-xgboost", "entity_id": "TOOL_010", "timestamp": "2026-06-28T12:00:00"}'

# Explain
curl -X POST http://localhost:9500/predictions/explain \
  -H "Content-Type: application/json" \
  -d '{"model_id": "predictive-maintenance-xgboost", "entity_id": "TOOL_010", "timestamp": "2026-06-28T12:00:00", "top_k": 10}'
```

**Via Python:**
```python
from backend.services import get_services
svc = get_services()

# Predict
pred = svc.predict("predictive-maintenance-xgboost", entity_id="TOOL_010", timestamp="2026-06-28T12:00:00")
print(f"Probability: {pred['probability']:.4f}, Prediction: {pred['prediction']}")

# Explain
expl = svc.explain("predictive-maintenance-xgboost", entity_id="TOOL_010", timestamp="2026-06-28T12:00:00", top_k=10)
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
| `anistroph_get_model_inputs` | `model_id` (string) | Get the prediction input schema for a model — what the caller must supply to predict. Returns the prediction mode (`entity_lookup` vs `records_or_entity_lookup`), the `entity_key`, whether `timestamp` is required, and the list of required source columns with their types and transforms. Use this before calling `anistroph_predict` to discover what inputs a model expects. Column order does not matter — columns are matched by name. |
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
| `get_services().train(dataset_id, target_name, model_type?, model_id?)` | dataset_id, target_name, model_type (optional — auto-selected from task type), model_id (optional) | `dict` (model_id, model_type, metrics) | Train a new model. Model type auto-selected from `target.type` if omitted. |
| `get_services().list_models()` | *(none)* | `list[ModelMetadata]` | List all trained models |
| `get_services().get_model(model_id)` | model_id (str) | `ModelMetadata` | Get a model's metadata |
| `get_services().get_model_metrics(model_id)` | model_id (str) | `dict` | Get model evaluation metrics |
| `get_services().get_model_inputs(model_id)` | model_id (str) | `dict` (model_id, dataset_id, target_name, target_type, prediction_mode, entity_key, requires_timestamp, required_columns, note) | Get the prediction input schema for a model — what the caller must supply to predict |
| `get_services().delete_model(model_id)` | model_id (str) | `bool` | Delete a model and its artifacts |
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

## 16. Semiconductor Datasets & Models

The semiconductor reference data is a wafer-level manufacturing dataset with
**three regression targets**, each with its own dataset config and model:

| Dataset ID | Target | Type | What it predicts |
|------------|--------|------|-----------------|
| `semiconductor_yield` | `wafer_yield` | regression | Overall wafer yield (0.0–1.0) |
| `semiconductor_cd` | `critical_dimension_nm` | regression | Measured CD after lithography/etch (~38 nm) |
| `semiconductor_film_thickness` | `film_thickness_nm` | regression | Measured deposited film thickness (~510 nm) |

All three configs point to the same source Parquet file but define different
targets. This demonstrates Anistroph's multi-target architecture: same data,
same features, different prediction goals.

### Overview

| Property | Value |
|----------|-------|
| Source data | `data/semiconductor_yield/data.parquet` |
| Dataset configs | `datasets/semiconductor_yield/`, `datasets/semiconductor_cd/`, `datasets/semiconductor_film_thickness/` |
| Entity key | `wafer_id` |
| Time key | `timestamp` (used for chronological splitting) |
| Row count | 50,000 wafers |
| Columns | 31 (28 features + 3 targets) |
| Split | Chronological — 70% train / 15% validation / 15% test |

### Data generation

```bash
python scripts/generate_semiconductor_yield_data.py --wafers 50000
```

Output: `data/semiconductor_yield/data.parquet` (shared by all three dataset configs)

Each row represents one completed wafer with:
- **Hierarchy:** lot_id -> wafer_id
- **Categorical context:** product_id, fab_id, process_route
- **Etch process:** etch_tool, etch_chamber, etch_recipe + 7 numeric measurements
- **Deposition process:** deposition_tool, deposition_chamber, deposition_recipe + 5 numeric measurements
- **Lithography:** exposure_dose, focus_offset
- **Maintenance:** maintenance_age_etch, maintenance_age_deposition
- **Targets:** wafer_yield, critical_dimension_nm, film_thickness_nm

### Injected relationships

The synthetic generator injects learnable but imperfect relationships for each
target:

**Wafer yield:**

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

**Critical dimension (CD):**

| Condition | CD Effect |
|-----------|----------|
| Nominal | ~38.0 nm |
| Higher exposure_dose | Smaller CD (-0.8 nm per unit dose) |
| Larger |focus_offset| | Wider CD |
| ETCH_02 | Over-etch → smaller CD (-0.5 nm) |
| RECIPE_C | Aggressive etch → smaller CD (-0.4 nm) |
| ETCH_02 + RECIPE_C | Interaction → significantly smaller CD |
| Higher etch_temperature | Faster etch → smaller CD |

**Film thickness:**

| Condition | Film Thickness Effect |
|-----------|----------------------|
| Nominal | ~510.0 nm |
| Longer deposition_process_time | Thicker (+0.8 nm per minute) |
| DEP_02 | Thinner (-8.0 nm) |
| DEP_03 | Thicker (+6.0 nm) |
| DEP_RECIPE_B | Thicker (+5.0 nm) |
| Higher deposition_pressure | Slightly thicker |
| DEP_02 + DEP_RECIPE_A | Interaction → significantly thinner |

No single feature perfectly determines any target — the model must learn
combinations and interactions.

### Trained models

Three XGBoost regression models, one per target:

| Model ID | Target | R² | MAE | MAPE |
|----------|--------|-----|-----|------|
| `wafer-yield-xgboost` | wafer_yield | 0.81 | 0.0065 | 0.68% |
| `critical-dimension-xgboost` | critical_dimension_nm | 0.89 | 0.24 nm | 0.64% |
| `film-thickness-xgboost` | film_thickness_nm | 0.98 | 1.63 nm | 0.32% |

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
python scripts/generate_semiconductor_yield_data.py --wafers 50000

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
  --model-type xgboost_regressor --model-id wafer-yield-xgboost

# Train linear baseline
python scripts/train_model.py --dataset semiconductor_yield \
  --model-type linear_regression --model-id wafer-yield-linear-v001
```

**Predict via MCP (Claude Desktop, Claude CLI, or any stdio MCP client):**
> "List all Anistroph models"
> "Predict wafer yield for WAFER_015000 using model wafer-yield-xgboost"
> "Explain that prediction - what are the top drivers?"
> "Find the worst yield combinations in the semiconductor dataset"
> "Show yield by etch tool and chamber"

**Predict via REST:**
```bash
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{"model_id": "wafer-yield-xgboost", "entity_id": "WAFER_015000"}'
```

**Predict via Python:**
```python
from backend.services import get_services
svc = get_services()

# Predict
pred = svc.predict("wafer-yield-xgboost", entity_id="WAFER_015000")
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
| Row count | 40,000 listings |
| Columns | 11 |
| Target | `price` (regression, USD) |
| Split | Chronological — 70% train / 15% validation / 15% test |

### Data generation

```bash
python scripts/generate_home_prices_data.py --homes 40000
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
python scripts/generate_home_prices_data.py --homes 40000

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
  --model-type xgboost_regressor --model-id home-prices-xgboost
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

---

## 18. Staged Prediction — Same Target, Progressive Feature Sets

This section documents an architectural pattern Anistroph supports **today,
with no code changes**: training multiple models against the *same target*
on the *same source data*, where each model is restricted to the features
available at a particular stage of a physical process. The motivating
example is wafer fabrication, where yield can be predicted at four points
in the line with progressively more information:

```
                           Target = WAFER YIELD

Before Etch          After Etch          After Deposition       Before Test
    │                    │                      │                    │
    ▼                    ▼                      ▼                    ▼
 Model A              Model B                Model C              Model D
    │                    │                      │                    │
Product              + Etch actuals        + Deposition         + Litho
Route                + Chamber               actuals              actuals
Recipe               + Temp/Pressure       + Film thickness    + CD
Setpoints            + RF/etc.
```

### Why this works without engine changes

The mechanism is the same one already used for multi-target datasets
(§16: `semiconductor_yield`, `semiconductor_cd`, `semiconductor_film_thickness`
share one parquet but differ in their `target:` block). The staged pattern
is the mirror image: **same target, different `features:` blocks**.

The key guarantee is in the Feature Engine — it only reads columns listed
in the YAML's `features:` block, even if the source parquet contains many
more columns. Data leakage is prevented by the YAML, not by the engine:

```python
# backend/features/engine.py — the engine iterates ONLY over feature_spec.features
for feat_name, col_spec in feature_spec.features.items():
    source_col = col_spec.column
    transforms = normalize_transforms(col_spec.transforms)
    ...
```

A column present in the parquet but absent from a model's `features:` block
is never read during training or inference for that model. So Model A can
be trained on the same parquet that contains etch actuals, and those
actuals will not leak into Model A — they simply aren't listed in Model A's
`features:` block.

### Configuration pattern

Create one dataset config per stage. All configs share the same source
parquet, the same `target:` block, and the same `split:` block. They differ
only in their `features:` block (and, for clarity, in the `role:` of
columns that are targets in one config but features in another).

| Config | Stage | Features included (cumulative) |
|--------|-------|-------------------------------|
| `semiconductor_yield_stage_a` | Before Etch | Product route, recipes, setpoints (litho exposure dose, focus offset) |
| `semiconductor_yield_stage_b` | After Etch | + Etch tool/chamber/temperature/pressure/RF/gas flow/process time |
| `semiconductor_yield_stage_c` | After Deposition | + Deposition tool/chamber/temperature/pressure/process time + `film_thickness_nm` |
| `semiconductor_yield_stage_d` | Before Test | + `critical_dimension_nm` (CD) |

Each config trains a separate model with its own persisted `FeatureSpec`
and `FeatureMetadata`. The data already contains every column needed —
`film_thickness_nm` and `critical_dimension_nm` are already in the parquet
(see §16).

### Worked example: Stage A (Before Etch) YAML

```yaml
# datasets/semiconductor_yield_stage_a/dataset.yaml
#
# Predicts wafer_yield using ONLY information available before etch begins.
# Etch and deposition actuals exist in the parquet but are deliberately
# omitted from `features:` so they cannot leak into this model.

dataset:
  dataset_id: semiconductor_yield_stage_a
  name: Semiconductor Yield — Stage A (Before Etch)
  entity_key: wafer_id
  time_key: timestamp
  columns:
    timestamp:        {type: timestamp, role: identifier}
    lot_id:           {type: categorical, role: identifier}
    wafer_id:         {type: categorical, role: identifier}
    product_id:       {type: categorical, role: feature}
    fab_id:           {type: categorical, role: feature}
    process_route:    {type: categorical, role: feature}
    etch_recipe:      {type: categorical, role: feature}     # recipe only, not actuals
    deposition_recipe: {type: categorical, role: feature}    # recipe only, not actuals
    exposure_dose:    {type: numeric, role: feature}         # litho setpoint
    focus_offset:     {type: numeric, role: feature}         # litho setpoint
    # Etch/deposition actuals are declared here for schema completeness
    # but their role is `metadata` so they are never used as features.
    etch_tool:                {type: categorical, role: metadata}
    etch_chamber:             {type: categorical, role: metadata}
    etch_temperature_mean:    {type: numeric, role: metadata}
    # ... (other etch/deposition actuals as metadata)
    film_thickness_nm:        {type: numeric, role: metadata}
    critical_dimension_nm:    {type: numeric, role: metadata}
    wafer_yield:              {type: numeric, role: target}
  split:
    strategy: chronological
    train: 0.70
    validation: 0.15
    test: 0.15

features:
  product_id:        {column: product_id, transforms: [categorical]}
  fab_id:            {column: fab_id, transforms: [categorical]}
  process_route:     {column: process_route, transforms: [categorical]}
  etch_recipe:       {column: etch_recipe, transforms: [categorical]}
  deposition_recipe: {column: deposition_recipe, transforms: [categorical]}
  exposure_dose:     {column: exposure_dose, transforms: [current]}
  focus_offset:      {column: focus_offset, transforms: [current]}

target:
  name: wafer_yield
  type: regression
  source_column: wafer_yield
```

Stage B, C, and D YAMLs are identical except that each adds more entries to
the `features:` block (and flips the corresponding `columns:` entries from
`role: metadata` to `role: feature`). The `target:` block is the same in
all four.

### Why `role: metadata` for unused actuals

Columns in the parquet that should be available for `sample_rows`, slicing,
and profiling — but must never be model inputs for a given stage — should
be declared with `role: metadata`. This makes the exclusion explicit and
auditable. A column with `role: feature` that is *not* listed in `features:`
would also be excluded from the model, but `role: metadata` documents the
intent: "this column exists in the data but is not a model input for this
stage."

### Training and evaluation

Register and train each stage independently:

```python
from backend.services import get_services

svc = get_services()

# Register all four stage configs (all point at the same parquet)
for stage in ["a", "b", "c", "d"]:
    svc.register_dataset_from_config(
        f"datasets/semiconductor_yield_stage_{stage}/dataset.yaml",
        parquet_path="data/semiconductor_yield/data.parquet",
    )

# Train one model per stage
for stage in ["a", "b", "c", "d"]:
    svc.train_model(
        dataset_id=f"semiconductor_yield_stage_{stage}",
        target_name="wafer_yield",
        model_type="xgboost_regressor",
    )
```

Because all four configs use the same chronological split on the same
parquet, they are evaluated on the **same held-out wafers**. Their metrics
(MAE, RMSE, R²) are directly comparable — you'd expect R² to increase from
Stage A → D as more information becomes available.

### Inference contract per stage

Each model's persisted `FeatureSpec` determines what a caller must supply.
For non-temporal datasets, send `records` with raw source values matching
that model's `features:` block:

```bash
# Stage A — only pre-etch information
curl -X POST http://localhost:9500/predictions -H "Content-Type: application/json" -d '{
  "model_id": "semiconductor_yield_stage_a-xgboost_regressor-...",
  "records": [{
    "product_id": "PROD_A",
    "fab_id": "FAB_1",
    "process_route": "ROUTE_X",
    "etch_recipe": "ETCH_02",
    "deposition_recipe": "DEP_01",
    "exposure_dose": 32.5,
    "focus_offset": 0.12
  }]
}'

# Stage D — full process history
curl -X POST http://localhost:9500/predictions -H "Content-Type: application/json" -d '{
  "model_id": "semiconductor_yield_stage_d-xgboost_regressor-...",
  "records": [{
    "product_id": "PROD_A", "fab_id": "FAB_1", "process_route": "ROUTE_X",
    "etch_recipe": "ETCH_02", "deposition_recipe": "DEP_01",
    "exposure_dose": 32.5, "focus_offset": 0.12,
    "etch_tool": "ETCH_01", "etch_chamber": "CH_B",
    "etch_temperature_mean": 85.2, "etch_pressure_mean": 4.1,
    # ... all etch and deposition actuals ...
    "film_thickness_nm": 511.3,
    "critical_dimension_nm": 38.2
  }]
}'
```

Columns not in a model's `features:` block are silently ignored — so even
if a caller sends etch actuals to a Stage A model, they do not leak in.

> **Use `records`, not `entity_id`, for staged predictions in a real
> deployment.** `entity_id` lookup loads the full row from the parquet,
> which includes later-stage actuals that would not exist yet at the point
> in the process where Stage A/B/C are called. The model still only builds
> features from its own `features:` list, so the prediction is correct —
> but in a live fab, those later-stage columns would not yet be populated.
> Sending `records` with only the columns available at that stage is the
> faithful representation.

### What is NOT supported (and workarounds)

| Limitation | Workaround |
|------------|------------|
| No first-class "stage" abstraction — the engine doesn't know Stage B follows Stage A | Represent stages via `dataset_id` naming (`_stage_a`, `_stage_b`, ...) and compare models individually via the Models list or Evaluation tab |
| No UI view that plots the 4 models' R² on a single progression chart | Compare metrics manually via `anistroph_get_model_metrics` or the Evaluation tab, one model at a time |
| No cascading model outputs as features (Model A's *prediction* cannot be an input feature to Model B) | Pre-compute Model A's predictions, write them as a new column in the parquet, then list that column in Model B's `features:` block. Requires a one-time offline scoring pass. |
| No automatic stage comparison endpoint | Use `anistroph_evaluate_model` on each of the 4 models and compare the returned `metrics` dicts. All four are evaluated on the same held-out wafers, so the comparison is apples-to-apples. |
| **Registration duplicates the parquet on disk** (known quirk — to be fixed) | Each `register_dataset_from_config` call calls `ingest()` → `persist_parquet()`, which writes a new copy of the source data plus new train/eval/validate partition files — even when the source parquet is identical to an existing dataset's. For the staged pattern (4 stages × same data) and the multi-target pattern (§16: 3 configs × same data), this means multiple copies of the same rows on disk. With the current semiconductor data (~7 MB × 4 configs), this is ~50 MB of duplicates. The duplication is wasteful but not incorrect — the model trains on the same rows, just from a duplicate file. A future "share parquet" registration mode that reuses an existing dataset's parquet + partitions and stores only a new config/feature-spec/target-spec would eliminate this. |

### Relationship to multi-target datasets (§16)

The staged pattern and the multi-target pattern are duals:

| | Multi-target (§16) | Staged (this section) |
|---|---|---|
| Same | Source parquet, `features:` block, `split:` block | Source parquet, `target:` block, `split:` block |
| Differs | `target:` block (different prediction goals) | `features:` block (different available information) |
| Use case | One dataset, multiple things to predict | One thing to predict, multiple points in a process to predict it from |

Both patterns rely on the same underlying property: a dataset config is a
*view* over the source data, and the `features:` + `target:` blocks define
that view. Multiple views can share one parquet.
