---
layout: default
title: "Setup & Usage | Anistroph"
description: "Set up Anistroph, connect it to Claude through MCP, and explore dataset configuration, prediction, explanation, evaluation, and multidimensional analysis."
---

# Anistroph — Setup & Usage Guide

This guide shows how to set up Anistroph, connect it to Claude through MCP, and use its prediction, explanation, evaluation, and multidimensional analysis capabilities.

Start with the quick-start section to get Anistroph running and try it with Claude. The sections that follow explain dataset configuration, temporal prediction, operations, interfaces, and the underlying APIs in more detail.

> **New to Anistroph?** See the [Anistroph project overview](https://vrraj.github.io/anistroph/) for the architecture, capabilities, and design goals.

> **AI Agent Analysis & Validation**
>
> **Discover → Understand → Execute → Validate**
>
> Claude and AI agents first discover the available datasets and models, then inspect the selected model's input contract to determine required inputs, temporal requirements such as `as_of`, and required inference history.
>
> The agent can then orchestrate prediction, explanation, evaluation, and analysis through MCP. These operations are executed by Anistroph's shared services and can be independently reproduced through the Web UI or REST API when validation is required.

## Contents

- [Get Anistroph Running](#get-anistroph-running)
- [Use Anistroph with Claude](#use-anistroph-with-claude)
- [What Just Happened?](#what-just-happened)
- [Reference Datasets](#reference-datasets)
- [Configure Your Own Dataset](#configure-your-own-dataset)
- [Temporal Prediction](#temporal-prediction)
- [Operations](#operations)
- [Other Interfaces](#other-interfaces)
- [Example Queries](#example-queries)
- [Testing & Troubleshooting](#testing--troubleshooting)
- [API Reference](#api-reference)
- [Web UI](#web-ui)

---

## Get Anistroph Running

For installation prerequisites and the architecture overview, see [README.md](../README.md).

```bash
# Generate and register the reference datasets
make setup

# Start Anistroph natively
make start-native

# Or start with Docker
make start
```

Once Anistroph is running, the MCP Streamable HTTP endpoint is available at `http://localhost:9500/mcp`. The Web UI and Swagger documentation are also available, but the quickest way to experience Anistroph is through Claude and MCP.

---

## Use Anistroph with Claude

The fastest way to explore Anistroph is to connect Claude Desktop directly to its MCP server.

### Claude Desktop (MCP stdio)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "anistroph": {
      "command": "/path/to/anistroph/.venv/bin/python",
      "args": ["-m", "backend.integrations.mcp.server"],
      "cwd": "/path/to/anistroph"
    }
  }
}
```

> Use the absolute path to the venv Python so Claude Desktop picks up all installed dependencies.

After saving: fully quit Claude Desktop (`Cmd+Q`), reopen, start a new conversation, and verify Anistroph tools appear (hammer/tools icon).

### Try Anistroph

After Claude can see the Anistroph tools, start with a short workflow:

1. **Discover** — "What datasets and models are available in Anistroph?"
2. **Predict** — "Predict 4-week material demand for FAB_A__MAT_0001."
3. **Explain** — "Explain that prediction — what are the top drivers?"
4. **Analyze** — "Which suppliers are associated with the greatest shortage risk?"
5. **Evaluate** — "Where is forecast error highest by fab × material category?"

These queries exercise the same shared Anistroph services used by REST, Python, and the Web UI.

---

## What Just Happened?

Claude is acting as the conversational client. MCP exposes Anistroph's prediction and analytical capabilities as tools, while Anistroph owns the datasets, feature construction, trained models, evaluation, and explanation logic.

```text
Claude / AI Agent
       ↓
      MCP
       ↓
    Anistroph
       ↓
Dataset + Feature Engine + Model
       ↓
Predict • Explain • Analyze • Evaluate
```

Claude does not need to construct engineered model features itself. It can discover the model's required inputs, invoke prediction or explanation, and use Anistroph's analytical tools against the same registered datasets and models.

Administrative operations such as dataset registration, model training, and deletion remain outside MCP by design.

---

## Reference Datasets

Anistroph ships with synthetic reference datasets across four domains. Each can be explored via Claude/MCP using the prompts in [Example Queries](#example-queries).

| Dataset | Domain | Target | Task |
|---------|--------|--------|------|
| **Semiconductor Procurement — Demand** | Supply chain | `material_demand_next_4w` | Regression |
| **Semiconductor Procurement — Shortage Risk** | Supply chain | `shortage_risk_next_4w` | Classification |
| **Semiconductor Yield** | Manufacturing | `wafer_yield` | Regression |
| **Semiconductor Critical Dimension** | Manufacturing | `critical_dimension_nm` | Regression |
| **Semiconductor Film Thickness** | Manufacturing | `film_thickness_nm` | Regression |
| **Semiconductor Staged Prediction (A–D)** | Manufacturing | `wafer_yield` (4 stages) | Regression |
| **Predictive Maintenance — Failure** | Equipment health | `failure_within_horizon` | Classification |
| **Predictive Maintenance — RUL** | Equipment health | `remaining_useful_life_hours` | Regression |
| **Predictive Maintenance — Maintenance Required** | Equipment health | `maintenance_required` | Classification |
| **Bay Area Home Prices** | Real estate | `price` | Regression |

> **Temporal datasets & rolling forecasts:** Anistroph supports temporal prediction where rolling features are dynamically reconstructed from entity history at prediction time — the model is static, the features are not. See [Temporal Prediction](#temporal-prediction) for the full explanation.

---

## Configure Your Own Dataset

Before registering a dataset, you author a `dataset.yaml` that declares the schema, the model inputs, and the target. This is the single source of truth the generic ML pipeline consults — no domain knowledge lives in the engine itself.

A YAML has a `dataset:` block containing schema and split configuration, plus top-level `features:` and `target:` blocks:

```yaml
dataset:        # identifiers + schema + split strategy
  ...
  columns:      # per-column type and role
    ...
  split:        # train/evaluation partitioning (optional)
    ...

features:       # actual model inputs
  ...

target:         # what to predict
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

- **`entity_key`** (required): the column identifying one entity (one wafer, one machine, one property). Used for grouping rolling windows and for `entity_id` lookups at inference time.
- **`time_key`** (optional): if present, the dataset is treated as *temporal* — rolling-window transforms become available, splits are chronological by default, and inference can be called with `(entity_id, timestamp)` to predict from history. If absent, the dataset is *non-temporal* — only `current` and `categorical` transforms are meaningful, splits are random, and inference requires either `entity_id` (single-row lookup) or `records` (raw values supplied by the caller).

### The `columns:` block (inside `dataset:`)

Declares every column the source data contains. Each entry has a `type` and a `role`. Columns present in the data but missing from this block are silently ignored by the pipeline.

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

> **`role: feature` is necessary but not sufficient.** A column with `role: feature` is *eligible* to be a model input, but it only becomes one if it is also listed in the top-level `features:` block. The `columns:` block declares schema; the `features:` block declares model inputs.

### The `features:` block (top-level)

This is the contract between training and inference. Every entry here becomes a model input. Each entry maps a feature name to a source column plus one or more **transforms** that turn the raw column into engineered feature(s).

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
| `current` | `{column}_current` | numeric, categorical | Pass through the raw value at the current row. The only transform that makes sense for non-temporal datasets. |
| `categorical` | `{column}__{category}` (one column per learned category) | categorical | One-hot encode the column. Categories are fit on training data only and persisted in `FeatureMetadata`. Unseen categories at inference become all-zeros. Supports `min_frequency` (default 1) to drop rare categories. |
| `mean` / `min` / `max` / `std` / `median` | `{column}_{op}_{window}` | numeric, temporal only | Rolling aggregate over a trailing time window, grouped by `entity_key`. Leakage-safe: only uses rows up to and including the current timestamp. Requires `windows: ["1h", "6h", ...]`. |
| `slope` | `{column}_slope_{window}` | numeric, temporal only | Rolling linear-regression slope over the trailing window. Captures trend direction. |
| `delta` | `{column}_delta_{window}` | numeric, temporal only | Current value minus the minimum value in the trailing window. Captures recent deviation. |
| `hour_of_day` | `hour_of_day` | timestamp | Extract hour-of-day (0–23) from the `time_key`. |
| `day_of_week` | `day_of_week` | timestamp | Extract day-of-week (0–6) from the `time_key`. |
| `elapsed_time` | `elapsed_time` | timestamp | Seconds elapsed since the entity's first observation. |

**Transform syntax:** a transform can be written as a bare string (`- current`) or as a mapping with parameters (`- mean: {windows: [1h, 6h]}`). The bare form is shorthand for `{op: <name>}`.

> **The `features:` block is the inference contract.** For non-temporal datasets, a caller sending `records` to `/predictions` must include every source column listed here. The Feature Engine applies the same transforms using the persisted `FeatureMetadata` from training. The caller never constructs one-hot vectors or rolling aggregates — only raw source values.

### The `target:` block (top-level)

Declares what the model predicts. Drives automatic model selection and the evaluation metrics reported after training.

```yaml
target:
  name: price                      # internal target name (returned in predictions)
  type: regression                 # task type — see table below
  source_column: price             # the raw column to use as the label
```

For `future_event` targets (binary classification from a boolean event column over a time horizon):

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

### The `split:` block (inside `dataset:`, optional)

Controls how the data is partitioned at registration time:

```yaml
split:
  strategy: chronological   # "chronological" (temporal) or "random" (non-temporal)
  train: 0.80
  validation: 0.0           # reserved for validate-during-training partition
  eval: 0.20                # becomes the held-out evaluation partition
```

If omitted, defaults are read from `.env` (`TRAIN_DATASET_PCT`, `EVAL_DATASET_PCT`, `VALIDATE_DATASET_PCT`). Set `train: 1.0` to skip partitioning entirely (single-file mode).

**Temporal datasets** sort chronologically — oldest rows go to train, newest to evaluation. **Non-temporal datasets** shuffle with a fixed seed before splitting.

### Worked example: non-temporal regression

`datasets/home_prices/dataset.yaml` — one row per home listing, predict `price` from raw attributes:

```yaml
dataset:
  dataset_id: home_prices
  name: Bay Area Home Prices
  entity_key: property_id
  time_key: timestamp
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
    train: 0.80
    eval: 0.20

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

Inference contract: send `records` with `city`, `zip_code`, `sqft`, `bedrooms`, `bathrooms`, `lot_size_sqft`, `year_built`, `garage`. The engine one-hot encodes the two categoricals and passes the numerics through.

### Worked example: temporal classification with rolling windows

`datasets/predictive_maintenance/dataset.yaml` — one row per machine observation at 5-min intervals, predict failure within 24h:

```yaml
dataset:
  dataset_id: predictive_maintenance
  entity_key: machine_id
  time_key: timestamp              # temporal → rolling transforms available
  columns: ...
  split: {strategy: chronological, train: 0.80, eval: 0.20}

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
  source_column: failure
  horizon: 24h
  positive_class: 1
```

### Multi-target datasets

A single source parquet can support multiple targets. Each target gets its own dataset config (e.g. `semiconductor_yield`, `semiconductor_cd`, `semiconductor_film_thickness`) pointing at the same source data but defining different `target:` sections. Each receives independent partitions, training, evaluation, and model artifacts.

---

## Temporal Prediction

Temporal prediction introduces an important distinction between the **trained model**, the **current model inputs**, and the **future period being predicted**.

### 1. The trained model does not contain current history

A trained model learns relationships between features and outcomes.

For example, a material-demand model may learn relationships involving:

```text
recent 4-week consumption
recent 8-week consumption
planned wafer starts
inventory on hand
supplier lead time
scheduled receipts
        ↓
material demand over the next 4 weeks
```

The model retains the learned relationship, but values such as **recent 4-week consumption** are not permanently stored as the current state of each fab/material combination.

Those values change whenever a new observation becomes available.

Therefore, a new prediction does **not** normally require retraining. It requires the latest feature values.

---

### 2. History-dependent features are calculated at prediction time

Consider a temporal feature configuration such as:

```yaml
material_consumption_qty:
  column: material_consumption_qty
  transforms:
    - current
    - mean:
        windows: [4w, 8w, 13w]
```

The trained model expects features such as:

```text
material_consumption_qty_mean_4w
material_consumption_qty_mean_8w
material_consumption_qty_mean_13w
```

For a prediction made **as of Week 20**, Anistroph retrieves the entity's observations through Week 20 and calculates the appropriate rolling values.

Conceptually:

```text
Entity history through Week 20
             ↓
4w / 8w / 13w rolling features
             ↓
current operational features
             ↓
existing trained model
             ↓
prediction
```

The historical observations are therefore used to **construct the current model inputs**. They are not being used to retrain the model.

#### Inference history window is derived from the model configuration

The amount of inference history required is determined from the temporal features configured for the model — not specified by the caller.

```text
Configured windows: 4w, 8w, 13w
              ↓
Required inference history: 13w
```

Anistroph scans only the required entity history (bounded by the longest configured window) rather than loading the entire dataset. The `anistroph_get_model_inputs` tool exposes this as `inference_history_window` (e.g. `"13w"`, `"6h"`, or `null` for non-temporal models).

The caller does not need to understand the temporal feature implementation. The MCP prediction contract remains simple:

```text
anistroph_predict(
    model_id,
    entity_id,
    as_of
)
```

---

### 3. The prediction point is an `as_of` boundary

Temporal prediction requires a point separating known history from the future.

Conceptually, this is the prediction **`as_of`** time.

For example:

```text
entity: FAB_A__MAT_0001
as_of: 2025-06-09
```

means:

> Make a prediction for this entity using only information available through June 9, 2025.

Anistroph can then:

1. Select the entity's history through the `as_of` point.
2. Calculate required rolling/history-based features.
3. Use current values from the latest available observation.
4. Pass the resulting feature vector to the existing trained model.
5. Return the prediction.

The `as_of` point is **not the future date being predicted**. It represents the end of known information.

---

### 4. The target determines the future horizon

Consider:

```yaml
target:
  name: material_demand_next_4w
  type: regression
  source_column: material_demand_next_4w
```

For each historical observation, the target represents demand during the four weeks following that observation.

```text
Observation Week 10 → actual demand Weeks 11–14
Observation Week 11 → actual demand Weeks 12–15
Observation Week 12 → actual demand Weeks 13–16
```

The trained regression model therefore learns to predict the **next four weeks** from the state available at each observation point.

At inference time the same interpretation applies:

```text
as_of Week 20 → predict Weeks 21–24
as_of Week 21 → predict Weeks 22–25
as_of Week 22 → predict Weeks 23–26
```

The forecast horizon rolls forward with the prediction point.

---

### 5. A rolling forecast does not require retraining

Suppose a model was trained through Week 100.

When Week 101 observations become available:

```text
New Week 101 observation
        ↓
history now includes Week 101
        ↓
rolling features recalculated
        ↓
existing trained model
        ↓
new next-4-week prediction
```

Week 102 repeats the same process using the updated history.

The model itself remains unchanged.

Retraining is appropriate when the **learned relationship** needs to change — for example because of model-performance degradation, concept drift, significant changes to the underlying process, or changes to the feature definition.

It is not required simply because another week of observations has arrived.

#### Training history vs inference history

Training and inference use history for different purposes:

```text
Training history
      ↓
learn relationships between features and outcomes
      ↓
trained model


Recent entity history
      ↓
construct current temporal features
      ↓
trained model
      ↓
new prediction
```

For example:

```text
Training history     2–3 years
Inference history    13 weeks
                     (derived from configured features)
Forecast horizon     4 weeks
```

These values are independent. Training history determines the operating patterns and relationships the model learns. Inference history provides the recent state required to construct the current model inputs. The forecast horizon determines how far forward the target represents.

---

### 6. Temporal features can also be supplied precomputed

Anistroph can conceptually support two patterns.

**History-derived features**

```text
entity + as_of
      ↓
Anistroph retrieves history
      ↓
Anistroph calculates rolling features
      ↓
model prediction
```

This keeps temporal feature construction within the prediction pipeline and ensures that features are calculated using information available at the requested prediction point.

**Precomputed features**

```text
upstream data pipeline
      ↓
fully prepared feature row
      ↓
model prediction
```

Here, an upstream process calculates rolling and lag features before invoking prediction.

The trained model ultimately receives the same type of feature vector in either case. The difference is **where temporal feature computation occurs**.

---

### 7. Temporal prediction vs. model retraining

| Event | New prediction? | Retraining required? |
|-------|----------------:|---------------------:|
| New weekly observation | Yes | No |
| Rolling window advances | Yes | No |
| Inventory or supplier metrics change | Yes | No |
| Forecast `as_of` point changes | Yes | No |
| Model performance degrades | — | Potentially |
| Underlying relationships materially change | — | Potentially |
| Feature definitions change | — | Yes |

The key architectural distinction is:

> **Temporal history determines the current feature values. Training determines how the model interprets those values. The forecast target determines what future outcome is predicted.**

These three concerns remain independent.

---

## Operations

Anistroph exposes operational capabilities across Python, REST, MCP, and the Web UI. Administrative operations such as training, registration, and deletion are intentionally excluded from MCP. Examples below show Python and MCP; REST endpoints are listed in the [API Reference](#api-reference).

### Register a Dataset

Registration reads the YAML config, validates the data, converts CSV to Parquet, partitions into train/evaluation/validate sets, and stores metadata in the dataset registry.

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
```

**Partition files created at registration:**

| File | Purpose | Used during training? |
|------|---------|----------------------|
| `{dataset_id}.train.parquet` | Model fitting | Yes — training loads only this file |
| `{dataset_id}.evaluation.parquet` | Held-out evaluation | Never during training |
| `{dataset_id}.validate.parquet` | Validation during training | Only if `VALIDATE_DATASET_PCT > 0` |
| `{dataset_id}.parquet` | Full dataset | No — used for profiling, slicing, `sample_rows` |

> Training loads only `train.parquet`; evaluation loads only `evaluation.parquet`. The two never overlap.

### Profile a Dataset

```python
prof = get_services().profile("predictive_maintenance")
print(f"Rows: {prof['row_count']}, Entities: {prof['entity_count']}")
```

Via MCP:
> "Profile the predictive_maintenance dataset. What's the failure rate?"

### Train a Model

The model type is auto-selected from the dataset's task type when `model_type` is omitted:

| Task type | Default model | Evaluation metrics |
|-----------|---------------|-------------------|
| `regression` | `xgboost_regressor` | MAE, MSE, RMSE, R², MAPE, max error |
| `classification` / `binary` / `future_event` | `xgboost` | ROC-AUC, PR-AUC, precision, recall, F1 |

```python
from backend.services import get_services

svc = get_services()

# Auto-select model from task type (recommended)
result = svc.train(
    dataset_id="predictive_maintenance",
    target_name="failure_within_horizon",
    model_id="my-pm-model",
)
print(f"Model type: {result['model_type']}, ROC-AUC: {result['metrics']['roc_auc']:.3f}")

# Explicit model type (overrides auto-selection)
result = svc.train(
    dataset_id="semiconductor_yield",
    target_name="wafer_yield",
    model_type="linear_regression",
    model_id="my-semi-lr",
)
```

Or via CLI:
```bash
python scripts/train_model.py --dataset semiconductor_yield \
  --model-type xgboost_regressor --model-id my-wafer-yield-model
```

> Training is NOT available via MCP — it is an admin operation available through REST, Python, CLI, and the Web UI only.

### List & Delete Models

```python
models = get_services().list_models()
for m in models:
    print(f"{m.model_id}: {m.model_type}, dataset={m.dataset_id}")

get_services().delete_model("old-model-id")
```

Via MCP:
> "What models are available in Anistroph?"

### Get Model Metrics

```python
metrics = get_services().get_model_metrics("my-pm-model")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}, F1: {metrics['f1']:.3f}")
```

Via MCP:
> "Show me the metrics for model my-pm-model."

### Evaluate a Model on the Held-Out Set

Evaluation runs a trained model against the **held-out evaluation partition** — the data reserved at registration that is never seen during training. This gives an unbiased estimate of model performance on unseen data.

**Classification metrics:** ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, threshold.

**Regression metrics:** MAE, MSE, RMSE, R², MAPE, max_error, median_abs_error, p95_abs_error, mean_prediction_error, baseline comparison.

```python
# Overall evaluation
result = get_services().evaluate_model("my-wafer-yield-model", sample_size=50)
print(f"R²: {result['metrics']['r2']:.4f}, MAE: {result['metrics']['mae']:.4f}")

# Slice-level evaluation (filtered to a subset)
result = get_services().evaluate_model(
    "my-home-prices-model",
    sample_size=50,
    filters={"city": "Saratoga"},
)
print(f"Overall MAPE: {result['metrics']['mape']:.2f}%")
print(f"Saratoga MAPE: {result['filtered_metrics']['mape']:.2f}%  (n={result['filtered_row_count']})")
```

Via MCP:
> "Evaluate model my-wafer-yield-model on the held-out set"
> "Evaluate the home price model filtered to San Jose only"

### Error Slice Discovery

Finds populations where the **prediction error** deviates most from the overall baseline. Searches 1/2/3-dimensional combinations of categorical columns.

**Supported error metrics:**

| Metric | Target type | Description |
|--------|-------------|-------------|
| `abs_error` | Regression | Absolute error (default) |
| `error` | Regression | Signed error — shows bias direction |
| `pct_error` | Regression | Percentage error — relative to actual value |
| `log_loss` | Classification | Per-row log loss |

```python
slices = get_services().find_evaluation_slices(
    "my-home-prices-model",
    metric="abs_error",
    min_sample_size=50,
    top_k=20,
)
for s in slices:
    vals = " + ".join(f"{k}={v}" for k, v in s["values"].items())
    print(f"  {vals}: n={s['row_count']}, MAE={s['metric_value']:.0f}, diff={s['difference']:+.0f}")
```

Via MCP:
> "Find the populations where the home price model has the worst prediction error"
> "Where is the model over-predicting vs under-predicting?"

### Predict

Two prediction modes:

**A. Entity lookup** (existing entity in the data): provide `model_id` + `entity_id` (+ `timestamp` for temporal models). Anistroph loads the entity's row(s) from the parquet, builds features, and returns the prediction.

**B. Records** (new or hypothetical entity): provide `model_id` + `records` — a list of dicts with raw source-column values matching the model's `features:` block. The caller never constructs engineered features.

> For temporal models, `timestamp` is the **`as_of` date** — the last known point in history, not the date being predicted. See [Temporal Prediction](#temporal-prediction) for the full explanation.

```python
from backend.services import get_services

# A. Entity lookup (temporal dataset — as_of date required)
pred = get_services().predict(
    model_id="my-pm-model",
    entity_id="TOOL_000",
    timestamp="2026-06-15T12:00:00",   # as_of: predict using history through this point
)
print(f"Probability: {pred['probability']:.4f}")

# A. Entity lookup (non-temporal dataset — no timestamp needed)
pred = get_services().predict(
    model_id="my-wafer-yield-model",
    entity_id="WAFER_015000",
)
print(f"Predicted yield: {pred['predicted_yield']:.4f}")

# B. Records (new wafer — raw source values for every feature column)
pred = get_services().predict(
    model_id="my-wafer-yield-model",
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
```

**Discovering model inputs** — before predicting with `records`, discover which source columns the model expects:

```python
schema = get_services().get_model_inputs("my-wafer-yield-model")
print(f"Mode: {schema['prediction_mode']}")
print(f"Requires timestamp: {schema['requires_timestamp']}")
print(f"Inference history window: {schema['inference_history_window']}")
for c in schema["required_columns"]:
    print(f"  {c['column']:30s} type={c['type']:12s} transforms={c['transforms']}")
```

Via MCP:
> "What inputs does the my-wafer-yield-model need for prediction?"

The response includes:
- `prediction_mode` — `entity_lookup_or_records` (both modes work; no history needed) or `entity_lookup` (rolling-window transforms require history)
- `requires_timestamp` — whether the `as_of` timestamp is required
- `inference_history_window` — the required history duration derived from the model's feature config (e.g. `"13w"`, `"6h"`, or `null`)

Via MCP:
> "Predict wafer yield for WAFER_015000 using model my-wafer-yield-model"
> "Predict wafer yield for a new wafer with: product_id=PROD_A, fab_id=FAB_01, ..."

### Explain a Prediction

Returns the top contributing features using SHAP TreeExplainer (for XGBoost) or importance-weighted contributions (fallback). Explanations are deterministic and model-derived.

```python
expl = get_services().explain(
    model_id="my-wafer-yield-model",
    entity_id="WAFER_015000",
    top_k=10,
)
print(f"Predicted yield: {expl['predicted_yield']:.4f}")
for d in expl["top_drivers"]:
    print(f"  {d['feature']:25s} = {str(d['value']):15s} impact={d['impact']:+.6f}")
```

Via MCP:
> "Explain the wafer yield prediction for WAFER_015000 — what are the top drivers?"

### SHAP Explanation Normalization

When a categorical source column is one-hot encoded, the FeatureEngine expands it into N binary model features using `{source}__{category}` naming. SHAP returns a separate impact for each. The explanation layer groups these back to the original source feature:

```text
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

The `detail` field retains raw per-category SHAP values for debugging. The top-level `feature`, `value`, and `impact` are what the caller (or Claude) should use.

**Naming convention rules:**
- One-hot columns use `{source}__{category}` with double-underscore separator
- Grouping splits on `__` from the right, so category values may contain `__` but source column names must not
- Passthrough (`current`) columns are named `{source}_current` in the model, displayed as `{source}` (suffix stripped) in explanations
- Rolling-window columns are named `{source}_{op}_{window}` and are not grouped (each is a distinct feature)

### Analytical Queries (Slice, Compare)

Analytical operations are independent of ML — they aggregate data by dimensions without involving models.

```python
# Slice by dimensions with aggregation
result = get_services().slice(
    dataset_id="predictive_maintenance",
    dimensions=["machine_type"],
    metric="failure",
    aggregation="mean",
)
# [{"machine_type": "TYPE_A", "failure_mean": 0.0014}, ...]

# Compare a metric across values of a single dimension
result = get_services().compare(
    dataset_id="predictive_maintenance",
    dimension="machine_type",
    metric="vibration",
    aggregation="mean",
)
```

Via MCP:
> "Slice predictive_maintenance by machine_type and show mean failure rate"
> "Find the worst yield combinations in the semiconductor dataset"

---

## Other Interfaces

Claude Desktop over MCP stdio is the simplest interactive starting point, but Anistroph exposes the same runtime capabilities through additional interfaces.

### MCP Streamable HTTP

For remote MCP clients (Axiolex, custom agents), the server exposes the same tools at `http://localhost:9500/mcp`. Start the server with `make start-native` or `make start`, then point your MCP client at the URL.

### ChatGPT (GPT Actions via ngrok)

ChatGPT runs in the cloud and needs a public URL. Use ngrok to temporarily tunnel your local server:

```bash
make start-gpt    # starts server + ngrok tunnel, prints public URL
make stop-gpt     # closes the tunnel when done
```

Paste the OpenAPI URL into ChatGPT: GPTs → Create → Configure → Actions → Import from URL. The filtered spec at `/openapi-gpt.json` excludes training and dataset registration — only runtime prediction, explanation, and analysis are exposed.

### REST / OpenAPI

The REST API exposes dataset, model, prediction, explanation, evaluation, and analysis endpoints. Once the server is running, the OpenAPI documentation is available at `/docs` and `/redoc`.

The complete endpoint list is in [API Reference](#api-reference).

---

## Example Queries

Claude prompts grouped by dataset. These work with any MCP client — Claude Desktop, Claude CLI, Cursor, etc. Replace model IDs with your own (check with "What models are available?").

### Discover (cross-dataset)

> "What datasets and models are available in Anistroph?"
> "Profile the semiconductor_procurement_demand dataset"
> "What columns and types are in semiconductor_yield?"

### Semiconductor Procurement — Demand

> "Predict 4-week material demand for FAB_A__MAT_0001 as of 2025-06-09"
> "What inputs does the demand model need?"
> "Explain that prediction — what are the top drivers?"
> "Which features are pushing the demand forecast up or down?"
> "Which material categories have the highest forecast demand at FAB_A?"
> "Evaluate the demand model on the held-out set"
> "Where is forecast error highest by fab × material category?"

### Semiconductor Procurement — Shortage Risk

> "Predict shortage risk for FAB_A__MAT_0001 as of 2025-06-09"
> "Which suppliers are associated with the greatest shortage risk?"
> "Evaluate the shortage risk model on the held-out set"
> "Where is shortage risk mispredicted — which fab × material category combinations?"

### Semiconductor Yield

> "Predict wafer yield for WAFER_015000"
> "Explain the wafer yield prediction for WAFER_015000 — what are the top drivers?"
> "Show yield by etch tool and chamber"
> "Find the worst yield combinations in the semiconductor dataset"
> "Compare wafer yield across fab_id values"
> "Evaluate the wafer yield model on the held-out set"
> "Where is the model over-predicting vs under-predicting wafer yield?"

### Semiconductor Critical Dimension

> "Predict critical dimension for WAFER_015000"
> "Explain that CD prediction — which process settings are driving it?"
> "Show critical dimension by etch tool and recipe"
> "Evaluate the CD model on the held-out set"

### Semiconductor Film Thickness

> "Predict film thickness for WAFER_015000"
> "Explain that film thickness prediction — what are the top drivers?"
> "Show film thickness by deposition tool and chamber"
> "Evaluate the film thickness model on the held-out set"

### Semiconductor Staged Prediction (A–D)

> "What inputs does the stage A model need?"
> "Predict yield for WAFER_015000 using stage A vs stage D — how much does accuracy improve?"
> "Compare predictions from all four staged models for the same wafer"
> "Show the metrics for all four staged models side by side"

### Predictive Maintenance — Failure

> "Predict failure probability for TOOL_010 as of 2026-06-02T05:30:00"
> "What's driving the failure risk for TOOL_010?"
> "Slice predictive_maintenance by machine_type and show mean failure rate"
> "Evaluate the failure model on the held-out set"
> "Which machine type has the worst prediction error?"

### Predictive Maintenance — RUL

> "Predict remaining useful life for TOOL_010 as of 2026-06-02T05:30:00"
> "Explain that RUL prediction — what are the top drivers?"
> "Show mean remaining useful life by machine_type"
> "Evaluate the RUL model on the held-out set"

### Predictive Maintenance — Maintenance Required

> "Predict maintenance required for TOOL_010 as of 2026-06-02T05:30:00"
> "Which machines are flagged for maintenance in the latest week?"
> "Evaluate the maintenance required model on the held-out set"

### Bay Area Home Prices

> "Predict the price of a 2000 sqft, 4-bedroom home in Saratoga"
> "Explain that price prediction — what are the top drivers?"
> "Show me 5 homes in Saratoga sorted by price descending"
> "Compare average price across city values"
> "Evaluate the home price model filtered to San Jose only"
> "Find the populations where the home price model has the worst prediction error"

> Training, dataset registration, and deletion are NOT available via MCP — use REST, Python, CLI, or the Web UI for admin operations.

---

## Testing & Troubleshooting

### Full test suite

```bash
pytest
```

147 tests covering datasets, features, targets, partitioning, training, inference, explanation, MCP, REST API, and SHAP grouping.

### Unit tests only

```bash
pytest tests/unit/ -v
```

### Integration tests only

```bash
pytest tests/integration/ -v
```

### Leakage and parity checks

```bash
# Rolling-window features never use observations after time T
pytest tests/unit/test_features.py::TestFeatureEngine::test_rolling_mean_leakage_safe -v

# A failure on one entity never labels another entity
pytest tests/unit/test_targets.py::TestFutureEventTarget::test_entity_isolation -v

# Train and inference use the same feature metadata
pytest tests/unit/test_features.py::TestFeatureEngine::test_inference_uses_same_metadata -v

# REST and MCP produce identical predictions (same service layer)
pytest tests/integration/test_e2e.py::TestEndToEnd::test_rest_and_mcp_same_services -v
```

### MCP tools not appearing in Claude Desktop

- Check config JSON is valid: `python3 -c "import json; json.load(open('~/Library/Application Support/Claude/claude_desktop_config.json'))"`
- Check logs: `tail -100 ~/Library/Logs/Claude/main.log | grep -i "mcp\|anistroph\|error"`
- Fully quit Claude Desktop (`Cmd+Q`), not just close the window
- Verify the venv Python path exists

### Predictions fail with "feature spec not found"

The model artifacts are missing. Re-train the model:
```bash
python scripts/train_model.py --dataset predictive_maintenance --model-type xgboost --model-id my-pm-model
```

### MCP returns empty lists

Restart Claude Desktop after registering new datasets or training new models.

### XGBoost library loading error (macOS)

```bash
brew install libomp
```

---

## API Reference

### REST API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/datasets` | List all registered datasets |
| POST | `/datasets` | Register a new dataset from config + source file |
| GET | `/datasets/{dataset_id}` | Get metadata for a specific dataset |
| GET | `/datasets/{dataset_id}/profile` | Profile a dataset |
| POST | `/datasets/{dataset_id}/rows` | Sample raw rows (optional filters, columns, sort) |
| DELETE | `/datasets/{dataset_id}` | Remove a dataset |
| GET | `/models` | List all trained models |
| GET | `/models/types` | List available model types |
| POST | `/models/train` | Train a new model |
| GET | `/models/{model_id}` | Get model metadata |
| GET | `/models/{model_id}/metrics` | Get training-time metrics |
| GET | `/models/{model_id}/inputs` | Get prediction input schema |
| POST | `/evaluations/{model_id}` | Evaluate a model on held-out eval partition |
| POST | `/evaluations/{model_id}/slices` | Find error slice populations |
| DELETE | `/models/{model_id}` | Remove a model |
| POST | `/predictions` | Make a single prediction (entity_id or records) |
| POST | `/predictions/batch` | Make multiple predictions |
| POST | `/predictions/explain` | Explain a prediction with top feature drivers |
| POST | `/analysis/slice` | Slice data by dimensions with aggregation |
| POST | `/analysis/compare` | Compare a metric across dimension values |
| POST | `/analysis/interesting-slices` | Find slices with largest deviation from baseline |
| GET | `/` | Web UI (static HTML) |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc API docs |

### MCP tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `anistroph_list_datasets` | *(none)* | List all registered datasets |
| `anistroph_profile_dataset` | `dataset_id` | Profile a dataset (schema, distributions, time range) |
| `anistroph_slice_data` | `dataset_id`, `dimensions`, `metric`, `aggregation?`, `filters?` | Slice by dimensions with aggregation |
| `anistroph_compare_data` | `dataset_id`, `dimension`, `metric`, `aggregation?`, `filters?` | Compare a metric across dimension values |
| `anistroph_find_interesting_slices` | `dataset_id`, `metric`, `dimensions?`, `min_sample_size?`, `top_k?` | Find slices with largest deviation from baseline |
| `anistroph_sample_rows` | `dataset_id`, `n?`, `filters?`, `columns?`, `sort_by?`, `descending?` | Return up to N raw rows, optionally filtered |
| `anistroph_list_models` | *(none)* | List all trained models |
| `anistroph_get_model_metrics` | `model_id` | Get evaluation metrics for a model |
| `anistroph_get_model_inputs` | `model_id` | Get prediction input schema (required columns, types, transforms, mode) |
| `anistroph_predict` | `model_id`, `entity_id?`, `timestamp?`, `records?` | Make a prediction (entity lookup or records) |
| `anistroph_explain_prediction` | `model_id`, `entity_id?`, `timestamp?`, `records?`, `top_k?` | Explain a prediction with top SHAP contributors |
| `anistroph_evaluate_model` | `model_id`, `sample_size?`, `filters?` | Evaluate against held-out eval partition |
| `anistroph_find_evaluation_slices` | `model_id`, `metric?`, `min_sample_size?`, `top_k?` | Find populations where model error deviates most |

### Python service methods

| Method | Description |
|--------|-------------|
| `get_services().list_datasets()` | List all registered datasets |
| `get_services().register_dataset_from_config(config_path, source_path, parquet_path?)` | Register a dataset from YAML + source data |
| `get_services().profile(dataset_id)` | Profile a dataset |
| `get_services().train(dataset_id, target_name, model_type?, model_id?)` | Train a new model (auto-selects model type if omitted) |
| `get_services().list_models()` | List all trained models |
| `get_services().get_model(model_id)` | Get a model's metadata |
| `get_services().get_model_metrics(model_id)` | Get model evaluation metrics |
| `get_services().get_model_inputs(model_id)` | Get prediction input schema |
| `get_services().delete_model(model_id)` | Delete a model and its artifacts |
| `get_services().predict(model_id, entity_id?, timestamp?, records?)` | Make a prediction |
| `get_services().explain(model_id, entity_id?, timestamp?, records?, top_k?)` | Explain a prediction |
| `get_services().slice(dataset_id, dimensions, metric, aggregation?, filters?, limit?)` | Slice data by dimensions |
| `get_services().compare(dataset_id, dimension, metric, aggregation?, filters?)` | Compare a metric across dimension values |
| `get_services().find_interesting_slices(dataset_id, metric, ...)` | Find slices with largest deviation from baseline |
| `get_services().sample_rows(dataset_id, n?, filters?, columns?, sort_by?, descending?)` | Return up to N raw rows |
| `get_services().evaluate_model(model_id, sample_size?, filters?)` | Evaluate against held-out eval partition |
| `get_services().find_evaluation_slices(model_id, metric?, ...)` | Find populations where model error deviates most |

### What's NOT exposed via MCP

| Capability | Available via | NOT available via |
|------------|--------------|-------------------|
| Train a model | REST, Python, CLI, Web UI | MCP (by design) |
| Register a dataset | REST, Python, Web UI | MCP (by design) |
| Delete a dataset/model | REST, Python | MCP |
| Arbitrary Python execution | *(nowhere)* | MCP (only the 13 defined tools) |

---

## Web UI

Anistroph also includes a static Web UI for direct exploration and cross-interface validation.

With the server running, open:

```text
http://localhost:9500
```

The Web UI uses the same underlying Anistroph services as the REST and MCP interfaces, so predictions and analyses can be reproduced across interfaces using the same model inputs.
