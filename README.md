# Anistroph

> **Explore through Claude:**
> The primary interface for exploring datasets, models, predictions, and explanations is **Claude Desktop** (or the **Claude CLI**) over MCP. Ask questions in natural language — list datasets, find interesting slices, predict outcomes, and inspect SHAP explanations — without leaving the conversation. See **[Claude Desktop (MCP stdio)](#claude-desktop-mcp-stdio)** for setup instructions. A FastAPI **Web UI** is also included for interactive browser-based exploration; see **[Web UI](#web-ui)**.

**Anistroph: extensible, domain-agnostic predictive analytics platform.**

**Anistroph** provides a common predictive and analytical architecture across isolated reference datasets. Dataset-specific ingestion and feature preparation are supported where required, while training, evaluation, model persistence, inference, explainability, multidimensional analysis, and MCP access share one common framework.

> **Anistroph** is a **reference framework** for testing the combination of prediction, model explainability, and multidimensional analysis across different datasets.

Under the hood, the platform is powered by **Polars + DuckDB + Parquet** for fast columnar data processing, **scikit-learn + XGBoost** for classification and regression modeling, and **SHAP TreeExplainer** for per-prediction explainability. This gives predictive systems precise model-driven explanations and multidimensional discovery without duplicating the application for each new domain.

## Anistroph at a Glance

```text
                                      ANISTROPH

        ADMIN / MODEL LIFECYCLE                         RUNTIME

┌──────────┐   ┌───────────┐   ┌───────────┐   ┌──────────────────────────┐   ┌──────────┐   ┌─────────────────┐
│   DATA   │ → │  PREPARE  │ → │   TRAIN   │ → │ PREDICT • EXPLAIN •     │ → │   MCP    │ → │ CLAUDE / AGENTS │
│          │   │           │   │           │   │ EXPLORE                  │   │          │   │                 │
│ CSV      │   │ Python    │   │ XGBoost   │   │ Inference                │   │ MCP SDK  │   │ Natural-language│
│ Parquet  │   │ Polars    │   │ sklearn   │   │ SHAP TreeExplainer      │   │ stdio    │   │ interaction     │
│          │   │ Parquet   │   │ Metrics   │   │ DuckDB / Polars         │   │          │   │                 │
└──────────┘   └───────────┘   └───────────┘   └──────────────────────────┘   └──────────┘   └─────────────────┘
                                      │
                                      ▼
                              Persisted Model
                         + features + metrics
```

### Pipeline

**1. Data** — Add an isolated dataset for a prediction or analysis use case. Input can be CSV or Parquet.

**2. Prepare** — Clean and transform data, define features and the prediction target, and store prepared data in Parquet.

**3. Train** — Train and evaluate the model as an admin operation. Persist the model, feature metadata, preprocessing information, and metrics.

**4. Predict • Explain • Explore** — Run inference, explain individual predictions with SHAP, and explore multidimensional patterns in the underlying data.

**5. MCP** — Expose runtime capabilities as tools for dataset discovery, prediction, explanation, slicing, and analysis.

**6. Claude / Agents** — Discover and invoke Anistroph capabilities through MCP using natural-language interaction.

> Datasets and models remain isolated by use case, while Anistroph reuses common training, inference, explainability, analysis, and MCP services where the underlying operation is the same.

## Features

**Datasets and isolation**
- **Multi-dataset support** — Register and operate on multiple isolated datasets through configuration, not code changes. Each dataset has its own data directory, feature preparation, and model artifacts.
- **Dataset isolation** — Reference datasets and their model artifacts remain fully isolated. Predictive maintenance data never touches semiconductor data. Adding a new domain does not require modifying existing datasets.
- **Configuration-driven registration** — Datasets are declared through YAML configs (`dataset.yaml`) specifying columns, types, roles, entity/time keys, feature transforms, split strategy, and target semantics.
- **CSV and Parquet ingestion** — Ingest from CSV or Parquet sources. All data is persisted as Parquet for efficient columnar access.
- **Dataset partitioning at registration** — Every registered dataset is partitioned into separate `train.parquet`, `evaluation.parquet`, and (optionally) `validate.parquet` files. Training loads only `train.parquet`; the held-out evaluation file is never used during model fitting. Split percentages default to `TRAIN_DATASET_PCT=0.80` / `EVAL_DATASET_PCT=0.20` / `VALIDATE_DATASET_PCT=0.0` from `.env` and can be overridden per-dataset via the YAML `split:` section. Temporal datasets sort chronologically (oldest → train, newest → evaluation); non-temporal datasets shuffle with a fixed seed.

**Feature engineering**
- **Leakage-safe transforms** — Rolling windows, slopes, deltas, and temporal features only use observations up to and including the current time. No future data leaks into training or inference.
- **Categorical encoding** — One-hot encoding with learned categories persisted in `FeatureMetadata` so inference applies the identical encoding as training. Output columns follow the `{source}__{category}` naming convention (e.g. `etch_tool__ETCH_02`).
- **SHAP explanation normalization** — When one-hot encoding expands a source feature into multiple model features, `anistroph_explain_prediction` aggregates SHAP contributions back to the original source feature. The MCP response returns `{feature: "etch_tool", value: "ETCH_02", impact: +0.0024}` rather than separate `etch_tool__ETCH_01 = 0`, `etch_tool__ETCH_02 = 1` entries. Raw per-category SHAP values are retained in a `detail` field for debugging.
- **Stable feature identities** — Feature names are human-readable and preserved end-to-end. SHAP explanations map directly back to meaningful source conditions, not opaque engineered indices.
- **Dataset-specific preparation** — Each dataset may have its own feature preparation logic while sharing the same downstream training, inference, and analysis services.

**Modeling**
- **Classification** — Logistic Regression and XGBoost classifiers for binary and future-event targets. ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrix evaluation.
- **Regression** — XGBoost Regressor and Linear Regression (Ridge) for continuous targets. MAE, MSE, RMSE, R², MAPE (mean absolute % error), max error, median absolute error, 95th percentile absolute error, and baseline comparison.
- **Chronological splitting** — Time-ordered train/validation/test splits prevent temporal leakage. Registration partitions into train/eval files (80/20 by default); training further splits train into train/validation for early stopping and threshold tuning.
- **Held-out evaluation** — Post-training, run inference on the persisted `evaluation.parquet` and compare predictions against known actuals. Supports slice-level evaluation via filters (e.g. metrics for a single city, lot, or zip code) — returns both overall and filtered metrics for comparison. Available via REST (`POST /evaluations/{model_id}`), MCP (`anistroph_evaluate_model`), and the Web UI Evaluation tab.
- **Error slice discovery** — While `find_interesting_slices` finds populations where the *target* deviates, `find_evaluation_slices` finds populations where the *prediction error* deviates — identifying segments where the model performs better or worse than average. Searches 1/2/3-dimensional combinations of categorical columns (e.g. product + tool + chamber) and ranks by deviation from the overall error baseline with minimum sample-size thresholds. Available via REST (`POST /evaluations/{model_id}/slices`), MCP (`anistroph_find_evaluation_slices`), and the Web UI Evaluation tab.
- **Model persistence and reload** — Trained models, preprocessing metadata, feature identities, feature order, and evaluation metrics are persisted together as artifacts. Runtime inference loads the persisted model and never retrains.
- **Easy model additions** — New model types live in separate adapter modules under `backend/models/`. Register them in `MODEL_FACTORIES` and the shared training, inference, and explanation paths pick them up automatically.

**Inference**
- **Dual prediction modes** — Predict by entity lookup (send `entity_id` + optional `timestamp`; Anistroph loads the row from the dataset) or by records (send raw source feature values as JSON for a new or hypothetical row). The caller never constructs engineered features — no one-hot vectors, no rolling aggregates. Anistroph applies the same FeatureEngine and persisted metadata as training. Available via REST, MCP, and the Web UI.
- **Model input schema discovery** — `anistroph_get_model_inputs` returns the required source columns, their types, transforms, and the supported prediction mode for any trained model. Use it before predicting to discover what inputs a model expects. Available via REST (`GET /models/{model_id}/inputs`), MCP, and the Web UI ("Load Input Schema" button on the Prediction tab).
- **Staged prediction** — Multiple configs can share the same source data and target but define progressively larger feature sets, representing process checkpoints (e.g. before etch, after etch, after deposition, before test). Each stage has its own dataset ID, model, and persisted feature metadata.

**Explainability**
- **SHAP TreeExplainer** — Per-prediction explainability for all XGBoost models using TreeSHAP. Returns signed contributions: positive = increases prediction, negative = decreases prediction.
- **Top positive and negative contributors** — Explanations return separate `top_positive` and `top_negative` lists, each sorted by contribution magnitude.
- **Feature identity preservation** — Feature engineering, preprocessing, model persistence, and inference preserve stable feature names and order so SHAP values map back to meaningful manufacturing or operational conditions.
- **Fallback for non-tree models** — Models without SHAP support fall back to importance-weighted contributions. The `explanation_method` field indicates which method was used.

**Multidimensional analysis**
- **Manual slicing** — Slice data by 1, 2, or 3 dimensions with aggregations (mean, sum, min, max, count, std, median).
- **Interesting slice discovery** — `find_interesting_slices` automatically searches 1, 2, and 3-dimensional combinations of categorical columns, ranks by deviation from the overall baseline, and requires a minimum sample size (default 100).
- **Deterministic** — All analysis is deterministic Python. No AI or LLM generates analytical conclusions.
- **Dataset-agnostic** — The same analysis framework operates on any registered dataset, not just semiconductor or maintenance data.

**MCP runtime access**
- **MCP stdio server** — Anistroph exposes runtime analysis and inference through MCP stdio for use by clients such as Claude Desktop.
- **13 MCP tools** — Dataset discovery, dataset profiling, slicing, comparison, interesting-slice discovery, raw row sampling, model discovery, model metrics, model input schema, prediction, SHAP-based prediction explanation, held-out evaluation, and error slice discovery.
- **No training via MCP** — Model training is an administrative operation. MCP is for runtime analysis and inference only.

| Tool | What it does |
|------|-------------|
| `anistroph_list_datasets` | List all registered datasets (IDs, names, row counts, time ranges) |
| `anistroph_profile_dataset` | Profile a dataset — column types, distributions, missing values, time range |
| `anistroph_slice_data` | Slice by 1–3 dimensions with an aggregation (mean, sum, min, max, count, std, median) |
| `anistroph_compare_data` | Compare a metric across values of a single dimension |
| `anistroph_find_interesting_slices` | Auto-discover slices with the largest deviation from the overall baseline |
| `anistroph_sample_rows` | Return up to N raw rows, optionally filtered by column values (e.g. a specific `wafer_id`) |
| `anistroph_list_models` | List all trained models (IDs, types, datasets, timestamps) |
| `anistroph_get_model_metrics` | Get evaluation metrics for a trained model |
| `anistroph_get_model_inputs` | Get the prediction input schema for a model — required columns, types, transforms, and prediction mode (entity lookup vs records) |
| `anistroph_predict` | Make a prediction by entity_id + timestamp (existing row) or records (raw feature values for a new row) |
| `anistroph_explain_prediction` | Explain a prediction with top positive and negative SHAP contributors (grouped by source feature, not raw one-hot) |
| `anistroph_evaluate_model` | Evaluate a trained model against the held-out evaluation partition (aggregate metrics + prediction-vs-actual sample) |
| `anistroph_find_evaluation_slices` | Find populations where model prediction error deviates most from the overall average (multidimensional search across categorical columns) |

See [MCP](#mcp) for server setup and [README_SETUP_USAGE.md](README_SETUP_USAGE.md#example-mcp-prompts) for example prompts.

**Architecture**
- **One shared inference path** — All models share one inference service selected by persisted model metadata. Adding a new model type does not create a new inference path.
- **REST API** — Full programmatic access to dataset management, analysis, training, prediction, and explanation endpoints.
- **Web UI** — Lightweight dashboard for exploring datasets, training models, making predictions (entity lookup or records-based), inspecting SHAP explanations, and evaluating models against held-out data.
- **Docker support** — Containerized deployment with bind-mounted code and data for development.

**Current Implementation And Extension Path**

| Layer | What Anistroph has today | Where it can extend |
| --- | --- | --- |
| Datasets | Predictive maintenance (classification) and semiconductor yield (regression) reference datasets | Additional domains through configuration without code changes |
| Models | XGBoost (classifier + regressor), Logistic Regression, Linear Regression | New model adapters in separate folders, registered in `MODEL_FACTORIES` |
| Explainability | SHAP TreeExplainer for XGBoost, importance-weighted fallback for others | SHAP KernelExplainer for non-tree models, deeper visualization |
| Analysis | Manual slicing, comparison, and automated interesting-slice discovery across 1-3 dimensions | More aggregation types, statistical significance testing |
| Access | REST API, Web UI, MCP stdio (11 tools) | Additional MCP tools, GraphQL, batch inference |

The shipped MCP server exposes runtime analysis and inference only. Model training, dataset registration, and administrative operations belong in the REST API, admin CLI, or Web UI.

For a concise map of the architecture and how datasets remain isolated while sharing common services, see [Anistroph at a Glance](#anistroph-at-a-glance) and [Architecture](#architecture).

### Reference Datasets

Anistroph v0.1 ships with three reference datasets that validate the architecture across different prediction problems and multiple targets per dataset.

**Tool Predictive Maintenance (classification + regression)**

```text
Tool
  ↓
Sensor measurements
  ↓
Operating conditions
  ↓
Equipment history
  ↓
Predictive model
  ↓
Failure / maintenance risk / remaining life
```

Equipment and sensor data used to model equipment behavior. 50 machines, 60 days, 5-minute observations. Three targets:

| Target | Type | What it predicts |
|--------|------|-----------------|
| `failure_within_horizon` | classification (future_event) | Will the tool fail within 24h? |
| `remaining_useful_life_hours` | regression | Hours until next failure |
| `maintenance_required` | classification | Does the tool need maintenance now? |

**Semiconductor Wafer Yield (regression)**

```text
Product
   ↓
Process Route
   ↓
Etch Tool → Chamber → Recipe → Process Conditions
   ↓
Deposition Tool → Chamber → Recipe → Process Conditions
   ↓
Lithography Tool → Recipe → Process Conditions
   ↓
Wafer Test
   ↓
Yield / CD / Film Thickness
```

Synthetic wafer manufacturing data representing process history across tools, chambers, recipes, and operating conditions. ~50,000 wafer rows. Three regression targets:

| Target | Type | What it predicts |
|--------|------|-----------------|
| `wafer_yield` | regression | Overall wafer yield (0.0–1.0) |
| `critical_dimension_nm` | regression | Measured CD after lithography/etch (~38 nm) |
| `film_thickness_nm` | regression | Measured deposited film thickness (~510 nm) |

**Bay Area Home Prices (regression)**

```text
Location (city / zip)
   ↓
Square footage
   ↓
Bedrooms / bathrooms / lot size
   ↓
Year built / garage
   ↓
Predictive model
   ↓
Sale price
```

Synthetic Bay Area listing data across San Jose, Saratoga, and Los Gatos. ~40,000 listings. Price driven primarily by square footage and location.

| Target | Type | What it predicts |
|--------|------|-----------------|
| `price` | regression | Home sale price (USD) |

### Multi-Target Architecture

Each dataset can define multiple targets. Anistroph creates a separate dataset
config per target (e.g. `semiconductor_yield`, `semiconductor_cd`,
`semiconductor_film_thickness`), all pointing to the same source Parquet file
but with different `target:` sections. This means:

- Each target gets its own train/eval/validate partitions
- Each target trains an independent model
- Each target is evaluated independently with task-appropriate metrics
- The same feature columns serve all targets within a dataset
- Adding a new target = adding a new YAML config, no code changes

## AI Integration — Claude and ChatGPT

Anistroph exposes runtime analysis and inference through two AI integration
protocols. Neither exposes model training.

| | Claude (MCP) | ChatGPT (GPT Actions) |
|---|---|---|
| **Protocol** | Model Context Protocol (stdio) | OpenAPI / REST |
| **Transport** | Local subprocess (stdio) | Public HTTPS via ngrok tunnel |
| **Scope** | 11 runtime tools | 14 runtime REST endpoints |
| **Training exposed?** | No | No |
| **Setup** | Claude Desktop config JSON | Custom GPT → Actions → Import URL |

### Claude Desktop (MCP stdio)

No public URL needed — MCP runs as a local subprocess.

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

Then ask Claude: *"Predict wafer yield for WAFER_015000 and explain what pushed it up or down."*

### ChatGPT (GPT Actions via ngrok)

ChatGPT runs in the cloud and needs a public URL. Use ngrok to temporarily
tunnel your local server.

```bash
# One-time setup
brew install ngrok
ngrok config add-authtoken <your-token>

# Start server + tunnel
make start-gpt

# Output shows the public URL + OpenAPI spec URL
# Paste the OpenAPI URL into ChatGPT: GPTs → Create → Configure → Actions → Import from URL

# Stop when done (closes the public tunnel)
make stop-gpt
```

The filtered OpenAPI spec at `/openapi-gpt.json` excludes training and
dataset registration — only runtime prediction, explanation, and analysis
are exposed. The ngrok URL is temporary and only works while `make start-gpt`
is running.

See **[README_SETUP_USAGE.md](README_SETUP_USAGE.md)** for detailed setup
instructions for both protocols.

## Why this exists

Predictive analytics problems across different domains share a common lifecycle: data ingestion, feature engineering, target construction, model training, evaluation, persistence, inference, and explanation. Most implementations either build a domain-specific application that cannot extend to new problems, or build a generic platform that is too abstract to be useful.

Anistroph takes a middle path: **datasets remain isolated and may require domain-specific preparation, while prediction, analysis, explainability, and runtime access are provided through a common framework.**

```text
Dataset-specific preparation
        ↓
Shared services
  Analysis
  Training
  Evaluation
  Persistence
  Inference
  Explainability
        ↓
REST / MCP / UI
```

This allows an investigation to move naturally from:

> What outcome does the model predict?

to:

> What drove the model's prediction?

and then:

> Do similar populations in the underlying data show the same behavior?

Model explanation (SHAP) and observed-data analysis (slicing) provide different perspectives on the same problem. Neither should be interpreted as proof of causality.

## What you get

- **Domain-agnostic predictive analytics framework** for structured datasets
- **Three reference datasets** (predictive maintenance, semiconductor yield, home prices) with **seven targets** across classification and regression, proving the architecture works across domains and task types
- **Configuration-driven dataset registration** through YAML specs
- **Leakage-safe feature engineering** with rolling windows, slopes, categorical encoding
- **Classification and regression models** (XGBoost, Logistic Regression, Linear Regression)
- **SHAP TreeExplainer** per-prediction explainability with top positive and negative contributors
- **Multidimensional analysis** with manual slicing and automated interesting-slice discovery
- **Model persistence and reload** with stable feature identities and preprocessing metadata
- **REST API** for programmatic access to all capabilities
- **MCP stdio server** with 10 tools for Claude Desktop and other MCP clients
- **Web UI** for interactive exploration
- **Admin training CLI** for model lifecycle management
- **Docker support** with bind-mounted code and data
- **110 passing tests** covering data generation, features, training, inference, explanation, analysis, MCP, and end-to-end workflows

## Usage Patterns

### Configuration-Driven Dataset Registration

Define datasets in YAML files for isolated, version-controlled configurations. Ideal for:
- Pre-defined dataset schemas and feature transforms
- Reference implementations that can be reproduced
- Startup-time loading of known datasets

```yaml
# datasets/semiconductor_yield/dataset.yaml
dataset:
  dataset_id: semiconductor_yield
  name: Semiconductor Wafer Yield
  entity_key: wafer_id
  time_key: timestamp
  columns:
    wafer_id: {type: categorical, role: identifier}
    etch_tool: {type: categorical, role: feature}
    etch_temperature_std: {type: numeric, role: feature}
    wafer_yield: {type: numeric, role: target}
  split:
    strategy: chronological
    train: 0.70
    validation: 0.15
    test: 0.15

features:
  etch_tool:
    column: etch_tool
    transforms: [categorical]
  etch_temperature_std:
    column: etch_temperature_std
    transforms: [current]

target:
  name: wafer_yield
  type: regression
  source_column: wafer_yield
```

### Admin Model Training

Train models through the admin CLI or REST API. Training is never exposed through MCP.

```bash
python scripts/train_model.py --dataset semiconductor_yield \
  --model-type xgboost_regressor --model-id wafer-yield-xgboost
```

### Runtime Prediction and Explanation

Predict and explain through REST, MCP, or Python. Inference never retrains.

```python
from backend.services import get_services
svc = get_services()

# Predict
pred = svc.predict("wafer-yield-xgboost", entity_id="WAFER_015000")
print(f"Predicted: {pred['predicted_yield']:.4f}, Actual: {pred['actual_yield']:.4f}")

# Explain with SHAP
expl = svc.explain("wafer-yield-xgboost", entity_id="WAFER_015000", top_k=10)
print(f"Method: {expl['explanation_method']}")
for c in expl['top_positive']:
    print(f"  +{c['feature']}: {c['impact']:+.4f}")
for c in expl['top_negative']:
    print(f"  -{c['feature']}: {c['impact']:+.4f}")
```

### Multidimensional Analysis

Discover unusual populations in the underlying data without relying on the model.

```python
slices = svc.find_interesting_slices("semiconductor_yield", "wafer_yield", top_k=10)
for s in slices:
    print(f"  {s['values']} rows={s['row_count']} yield={s['metric_value']:.4f} diff={s['difference']:.4f}")
```

### MCP Integration with Claude

Expose runtime analysis and inference to Claude Desktop through MCP stdio.

```json
{
  "mcpServers": {
    "anistroph": {
      "command": "python",
      "args": ["-m", "backend.integrations.mcp.server"],
      "cwd": "/path/to/Anistroph"
    }
  }
}
```

Then ask Claude:
> "List all Anistroph models"
> "Predict wafer yield for WAFER_015000 using model wafer-yield-xgboost"
> "Explain that prediction — what are the top drivers?"
> "Find the worst yield combinations in the semiconductor dataset"

## Install

```bash
cd anistroph

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

> **macOS note:** XGBoost requires `libomp`. Install it with
> `brew install libomp` if you see an XGBoost library loading error.

## Quick start

### Option A: Use directly in Python

*For Python applications and scripts*

```bash
python -m pip install -e .
```

```python
from backend.services import get_services

svc = get_services()

# Register a dataset
svc.register_dataset_from_config(
    "datasets/predictive_maintenance/dataset.yaml",
    "data/synthetic/predictive_maintenance.csv",
)

# Train a model
result = svc.train(
    dataset_id="predictive_maintenance",
    target_name="failure_within_horizon",
    model_type="xgboost",
)

# Predict
pred = svc.predict(
    model_id=result["model_id"],
    entity_id="TOOL_000",
    timestamp="2026-07-15T12:00:00",
)
print(f"Probability: {pred['probability']:.4f}, Prediction: {pred['prediction']}")

# Explain with SHAP
expl = svc.explain(
    model_id=result["model_id"],
    entity_id="TOOL_000",
    timestamp="2026-07-15T12:00:00",
    top_k=10,
)
print(f"Method: {expl['explanation_method']}")
```

### Option B: Use as a REST service

*For shared services and web UI*

```bash
uvicorn backend.main:app --reload --port 9500
```

Predict:

```bash
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{"model_id": "wafer-yield-xgboost", "entity_id": "WAFER_015000"}'
```

Explain:

```bash
curl -X POST http://localhost:9500/explain \
  -H "Content-Type: application/json" \
  -d '{"model_id": "wafer-yield-xgboost", "entity_id": "WAFER_015000", "top_k": 10}'
```

### Option C: Use with Claude Desktop via MCP

*For agentic analysis and inference*

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "anistroph": {
      "command": "python",
      "args": ["-m", "backend.integrations.mcp.server"],
      "cwd": "/path/to/Anistroph"
    }
  }
}
```

Restart Claude Desktop and ask:
> "List all Anistroph datasets"
> "Find interesting slices in the semiconductor yield dataset"
> "Predict wafer yield for WAFER_015000"

### Option D: Docker

*For containerized deployment*

```bash
make start
```

Web UI at `http://localhost:9500`, Swagger at `http://localhost:9500/docs`.

## Architecture

```text
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

All interfaces (REST, MCP, UI) invoke the same core Python services — no separate analytical or model logic lives inside any transport. MCP exposes no arbitrary Python execution; it is runtime analysis and inference only.

### Dataset Abstraction

Dataset-specific concepts never enter the generic ML pipeline. Instead:

```text
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

- **DatasetSpec** (`backend/datasets/spec.py`) — describes columns, types, roles, entity/time keys.
- **FeatureSpec** (`backend/features/spec.py`) — declares transforms per column (current, mean, std, slope, rolling windows, categorical encoding).
- **TargetSpec** (`backend/targets/spec.py`) — declares the prediction task type and target column. Supported types: `regression` (numerical outcome), `classification` (binary class/probability), `binary` (alias), `future_event` (classification with time horizon). The task type determines the default model and evaluation metrics. Extensible for future types (forecasting, anomaly detection).

Each reference dataset is a **configuration**, not an architectural dependency.

### Dataset Isolation

```text
data/
├── predictive_maintenance/
│   └── data.parquet
│
└── semiconductor_yield/
    └── data.parquet

artifacts/models/
├── predictive-maintenance-xgboost/
└── wafer-yield-xgboost/
```

Each dataset may have its own ingestion and feature-preparation logic while using common downstream services. This provides the foundation for adding additional analytical domains without creating separate applications.

## Prediction + Discovery

Prediction and multidimensional analysis are separate but complementary workflows.

```text
                 Data
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
   Predictive             Anistroph
     Model            dimensional analysis
       │                     │
       ▼                     ▼
   Prediction          unusual population
       │                     │
       ▼                     ▼
      SHAP             observed behavior
       │
       ▼
Why did the MODEL      Where in the DATA
produce this           is the behavior
prediction?            concentrated?
```

This allows an investigation to move naturally from:

> What outcome does the model predict?

to:

> What drove the model's prediction?

and then:

> Do similar populations in the underlying data show the same behavior?

Model explanation and observed-data analysis provide different perspectives on the same problem. Neither should be interpreted as proof of causality.

## Reference Datasets

### Tool Predictive Maintenance

The reference dataset (`datasets/predictive_maintenance/dataset.yaml`) defines:

- 50 machines, 60 days, 5-minute observations (~864K rows)
- Sensors: temperature, vibration, pressure, current, voltage, rpm, flow_rate
- Maintenance age, operating hours
- Failure event with failure mode (NONE/THERMAL/PRESSURE/VIBRATION/POWER)
- Three targets:

| Target | Type | Dataset config |
|--------|------|---------------|
| `failure_within_horizon` | classification | `datasets/predictive_maintenance/` |
| `remaining_useful_life_hours` | regression | `datasets/predictive_maintenance_rul/` |
| `maintenance_required` | classification | `datasets/predictive_maintenance_maint/` |

- Learnable deterioration patterns (vibration drift, temperature drift, pressure instability, maintenance age → increased failure probability)

**Trained models:**

| Model | Target | ROC-AUC | F1 |
|-------|--------|---------|-----|
| `predictive-maintenance-xgboost` | failure_within_horizon | 0.85 | 0.61 |
| `maintenance-required-xgboost` | maintenance_required | 1.00 | 0.94 |
| `rul-xgboost` | remaining_useful_life_hours | MAE=27.9h | — |

### Semiconductor Wafer Yield

The reference dataset (`datasets/semiconductor_yield/dataset.yaml`) defines:

- ~50,000 wafer rows (one row = one completed wafer)
- Categorical context: product_id, fab_id, process_route, etch/deposition tools, chambers, recipes
- Numeric process measurements: etch/deposition temperature, pressure, gas flow, RF power, process time
- Lithography: exposure dose, focus offset
- Maintenance age per tool
- Three regression targets:

| Target | Type | Dataset config |
|--------|------|---------------|
| `wafer_yield` | regression | `datasets/semiconductor_yield/` |
| `critical_dimension_nm` | regression | `datasets/semiconductor_cd/` |
| `film_thickness_nm` | regression | `datasets/semiconductor_film_thickness/` |

- Hidden interactions: ETCH_02 + CH_B, temperature variability, maintenance age, product/recipe combinations

**Trained models:**

| Model | Target | R² | MAE |
|-------|--------|-----|-----|
| `wafer-yield-xgboost` | wafer_yield | 0.81 | 0.0065 |
| `critical-dimension-xgboost` | critical_dimension_nm | 0.89 | 0.24 nm |
| `film-thickness-xgboost` | film_thickness_nm | 0.98 | 1.63 nm |

### Bay Area Home Prices

The reference dataset (`datasets/home_prices/dataset.yaml`) defines:

- ~40,000 home listings across San Jose, Saratoga, and Los Gatos
- Square footage, bedrooms, bathrooms, lot size, year built, garage
- City / zip code as dominant price driver
- Regression target: `price` (USD)

**Trained model: `home-prices-xgboost`** — R²=0.97, MAE=$94K, MAPE=3.23%

## Synthetic Data Generation

### Predictive Maintenance

```bash
python scripts/generate_sensor_data.py
```

Generates `data/synthetic/predictive_maintenance.csv` and `data/raw/predictive_maintenance.parquet`.

Options:
```bash
python scripts/generate_sensor_data.py --machines 50 --days 60 --interval 5 --seed 42
```

### Semiconductor Yield

```bash
python scripts/generate_semiconductor_yield_data.py --wafers 50000
```

Generates `data/semiconductor_yield/data.parquet`.

Options:
```bash
python scripts/generate_semiconductor_yield_data.py --wafers 50000 --seed 42
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
curl -X POST http://localhost:9500/datasets \
  -H "Content-Type: application/json" \
  -d '{"config_path": "datasets/predictive_maintenance/dataset.yaml", "source_path": "data/synthetic/predictive_maintenance.csv"}'
```

## Training

Training is an admin operation. Use the CLI or REST API.

The model type is **auto-selected from the dataset's task type** when omitted:

| Task type (YAML `target.type`) | Default model | Evaluation metrics |
|-------------------------------|---------------|-------------------|
| `regression` | `xgboost_regressor` | MAE, MSE, RMSE, R², MAPE, max error |
| `classification` | `xgboost` | ROC-AUC, PR-AUC, precision, recall, F1 |
| `binary` (alias) | `xgboost` | same as classification |
| `future_event` (alias) | `xgboost` | same as classification |

```bash
# model_type is optional — auto-selected from task type
python scripts/train_model.py --dataset semiconductor_yield --model-id wafer-yield-xgboost
```

Available model types (can be specified explicitly to override): `xgboost`, `logistic_regression`, `xgboost_regressor`, `linear_regression`.

```python
# Auto-select from task type (recommended)
result = svc.train(
    dataset_id="predictive_maintenance",
    target_name="failure_within_horizon",
)
# result["model_type"] == "xgboost" (classification dataset)

# Or specify explicitly
result = svc.train(
    dataset_id="semiconductor_yield",
    target_name="wafer_yield",
    model_type="linear_regression",
)
```

## Evaluation

Evaluation runs a trained model against the **held-out evaluation partition** (`evaluation.parquet`) — the 15% slice reserved at dataset registration time that is never seen during training. This gives an unbiased estimate of model performance on unseen data.

**Key points:**
- Training uses only `train.parquet`; evaluation uses only `evaluation.parquet` — no overlap.
- Evaluation is a runtime operation: it loads the persisted model, runs inference (`predict`, not `fit`), and computes metrics from scratch each time. No retraining occurs.
- Results are not cached — each call recomputes. Safe to re-run; produces the same numbers as long as the model artifact and eval partition are unchanged.

**Classification metrics:** ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix. Decision thresholds optimized for F1 on the validation set.

**Regression metrics:** MAE, MSE, RMSE, R², MAPE (mean absolute % error), max error, median absolute error, 95th percentile absolute error, mean prediction error, and baseline comparison against a constant mean predictor.

**Slice-level evaluation:** Pass `filters` to evaluate on a subset of the eval partition (e.g. `{"city": "Saratoga"}` or `{"lot_id": ["LOT_001"]}`). The response includes both overall metrics and `filtered_metrics` for the matching rows, so you can compare performance across segments — e.g. MAPE for San Jose vs Saratoga, or error for ETCH_02 vs ETCH_01 wafers.

**Error slice discovery:** `find_evaluation_slices` searches multidimensional combinations of categorical columns to find populations where the *prediction error* deviates most from the overall average — identifying segments where the model performs better or worse than expected. This complements `find_interesting_slices` (which finds where the *target* deviates) by answering "where does the *model* struggle?"

Available via REST API, Python, MCP, and the Web UI Evaluation tab.

```python
# Evaluate against the held-out eval partition
result = svc.evaluate_model("wafer-yield-xgboost", sample_size=50)
print(result["metrics"]["r2"])  # 0.82
print(result["metrics"]["mape"])  # 3.25 (% error)

# Slice-level evaluation (filtered to a single city)
result = svc.evaluate_model("home-prices-xgboost", filters={"city": "Saratoga"})
print(result["metrics"]["mape"])        # overall MAPE
print(result["filtered_metrics"]["mape"])  # Saratoga-only MAPE
print(result["filtered_row_count"])     # n rows matching

# Find populations where model error is worst
slices = svc.find_evaluation_slices("home-prices-xgboost", top_k=10)
for s in slices:
    vals = " + ".join(f"{k}={v}" for k, v in s["values"].items())
    print(f"  {vals}: n={s['row_count']}, MAE={s['metric_value']:.0f}, diff={s['difference']:+.0f}")
# Example output:
#   zip_code=95071: n=177, MAE=126022, diff=+31110  (Saratoga 95071 — worst)
#   zip_code=95122: n=180, MAE=65337, diff=-29575   (San Jose 95122 — best)

# Training-time metrics (stored in model registry, no recompute)
metrics = svc.get_model_metrics("wafer-yield-xgboost")
```

## Inference

The caller does not construct features. For temporal datasets, provide entity_id and timestamp. For non-temporal datasets, provide entity_id only.

```python
# Temporal (predictive maintenance)
pred = svc.predict(
    model_id="predictive-maintenance-xgboost",
    entity_id="TOOL_000",
    timestamp="2026-07-15T12:00:00",
)

# Non-temporal (semiconductor yield)
pred = svc.predict(
    model_id="wafer-yield-xgboost",
    entity_id="WAFER_015000",
)
```

Anistroph retrieves the relevant rows, builds features using the same Feature Engine + persisted FeatureMetadata, and predicts. Inference never retrains.

## Explainability

Per-prediction explainability uses **SHAP TreeExplainer (TreeSHAP)** for XGBoost models. Returns signed contributions: positive = increases prediction, negative = decreases prediction.

In simple terms, it answers:

> **"The model predicted 88.3% yield. Which inputs pushed the prediction up or down, and by how much?"**

### Example: Semiconductor wafer explanation

**Wafer WAFER_016756** (processed on ETCH_02 + CH_B, high temperature variability):

```text
The model predicted 88.3% yield.
Which inputs pushed the prediction up or down, and by how much?

Inputs that pushed the prediction DOWN:
  etch_tool = ETCH_02                  -2.4 pp   (this wafer used ETCH_02)
  etch_temperature_std = 2.50          -2.0 pp   (high temperature variability)
  etch_chamber = CH_B                  -1.3 pp   (this wafer used chamber CH_B)
  maintenance_age_etch = 446.7 hours   -0.6 pp   (old maintenance on the tool)

Inputs that pushed the prediction UP:
  product_id = PROD_B                  +0.03 pp  (this product slightly helps)
  deposition_pressure_mean = 5.02      +0.03 pp  (stable deposition pressure)
  exposure_dose = 25.0                 +0.01 pp  (nominal exposure dose)
                                     ─────────
Predicted yield                        88.3%
```

The explanation correctly identifies ETCH_02, high temperature variability, and CH_B as the dominant factors pushing yield down — matching the injected hidden relationships in the synthetic data.

### Example: Predictive maintenance explanation

**TOOL_010 at 2026-06-28T12:00** (predicted failure probability 98.6%):

```text
The model predicted 98.6% failure probability.
Which inputs pushed the prediction up or down, and by how much?

Inputs that pushed the prediction UP:
  maintenance_age_hours = 112.3        +3.23 pp  (high maintenance age)
  operating_hours = 693.7              +1.20 pp  (high operating hours)
  temperature_mean_6h = 84.6           +0.54 pp  (elevated temperature)
  pressure_mean_1h = 90.1              +0.39 pp  (pressure drift)

Inputs that pushed the prediction DOWN:
  vibration_max_6h = 3.0               -0.27 pp  (low vibration slightly reduces risk)
```

### Usage

```python
expl = svc.explain(
    model_id="wafer-yield-xgboost",
    entity_id="WAFER_015000",
    top_k=10,
)
print(f"Method: {expl['explanation_method']}")
print(f"Predicted yield: {expl['predicted_yield']:.4f}")
print("\nTop positive (increase yield):")
for c in expl['top_positive']:
    print(f"  {c['feature']}: {c['impact']:+.4f}")
print("\nTop negative (decrease yield):")
for c in expl['top_negative']:
    print(f"  {c['feature']}: {c['impact']:+.4f}")
```

### Response format

```json
{
  "model_id": "wafer-yield-xgboost",
  "entity_id": "WAFER_016756",
  "predicted_yield": 0.8827,
  "target_name": "wafer_yield",
  "explanation_method": "shap_tree_explainer",
  "top_positive": [
    {"feature": "product_id__PROD_B", "impact": 0.0003, "value": 1.0},
    {"feature": "deposition_pressure_mean_current", "impact": 0.0003, "value": 5.02}
  ],
  "top_negative": [
    {"feature": "etch_tool__ETCH_02", "impact": -0.0236, "value": 1.0},
    {"feature": "etch_temperature_std_current", "impact": -0.0195, "value": 2.50},
    {"feature": "etch_chamber__CH_B", "impact": -0.0130, "value": 1.0}
  ],
  "top_drivers": [
    {"feature": "etch_tool__ETCH_02", "impact": -0.0236, "value": 1.0},
    {"feature": "etch_temperature_std_current", "impact": -0.0195, "value": 2.50}
  ]
}
```

- `top_positive`: features that pushed the prediction up (sorted by impact descending)
- `top_negative`: features that pushed the prediction down (sorted by impact ascending)
- `top_drivers`: combined list sorted by absolute impact (backward compatible)
- `explanation_method`: `shap_tree_explainer` for XGBoost, `importance_weighted` for others

### Feature identity preservation

Feature engineering, preprocessing, model persistence, and inference preserve stable, human-readable feature identities and feature order:

1. **FeatureSpec** defines source columns and transforms
2. **FeatureEngine** produces named output columns using the `{source}__{category}` convention for one-hot and `{source}_current` for passthrough
3. **FeatureMetadata** persists the exact feature names and order
4. **Model artifact** stores `feature_metadata.json` alongside `model.joblib`
5. **Inference** reloads feature metadata and selects features in the same order
6. **SHAP** produces values in the same order as the feature matrix columns
7. **Explanation normalization** groups one-hot SHAP values back to the original source feature — sums contributions across all categories, identifies the active category (value=1), and returns `{feature, value, impact}` in human-readable form

This ensures explanations always map back to meaningful manufacturing or operational conditions (e.g. `etch_tool = ETCH_02` = "the wafer was processed on etch tool ETCH_02") rather than opaque engineered indices.

**SHAP explanation normalization in detail:** When a categorical source column like `etch_tool` is one-hot encoded into `etch_tool__ETCH_01`, `etch_tool__ETCH_02`, `etch_tool__ETCH_03`, SHAP returns a separate impact for each. The explanation layer groups these by splitting on the `__` separator, sums the impacts, and reports the active category:

```
One-hot SHAP values:          Grouped explanation:
  etch_tool__ETCH_01 = -0.001    →  feature: "etch_tool"
  etch_tool__ETCH_02 = +0.002    →  value:   "ETCH_02"     (the active category)
  etch_tool__ETCH_03 = -0.0005   →  impact:  +0.0005       (sum of all three)
```

Raw per-category SHAP values are retained in a `detail` field for debugging but are not in the default MCP response. This makes agentic explanations clean — Claude sees `etch_tool = ETCH_02 contributed +0.0005` rather than trying to interpret three separate one-hot entries.

SHAP explains **why the model produced a particular prediction**. It does not establish that a feature physically caused the observed outcome.

For non-XGBoost models, explanations fall back to importance-weighted contributions. The `explanation_method` field indicates which method was used.

## Multidimensional Analysis

Anistroph separately analyzes patterns in the underlying dataset without relying on the predictive model.

```python
# Manual slicing
slices = svc.slice(
    "semiconductor_yield",
    dimensions=["etch_tool", "etch_chamber"],
    metric="wafer_yield",
)

# Automated interesting-slice discovery
interesting = svc.find_interesting_slices(
    "semiconductor_yield",
    "wafer_yield",
    top_k=20,
)
```

The analysis framework supports:

- Single-dimension slicing
- Two-dimensional slicing
- Three-dimensional slicing
- Baseline comparisons
- Minimum population thresholds (default 100 rows)
- Ranked discovery of unusual populations

The same analytical framework operates across registered datasets rather than containing domain-specific analysis logic.

## REST API

```bash
uvicorn backend.main:app --reload --port 9500
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
| POST | `/predictions/explain` | Explain prediction (SHAP) |

Swagger docs at `http://localhost:9500/docs`, ReDoc at `http://localhost:9500/redoc`.

## MCP

Anistroph exposes deterministic capabilities through MCP stdio. The 10 tools are listed in the [Features section](#mcp-runtime-access) above.

Run the MCP server:
```bash
python -m backend.integrations.mcp.server
```

MCP, REST, and the Web UI all invoke the same core services — see [Architecture](#architecture). Model training is not exposed through MCP.

## Web UI

The Web UI provides an interactive dashboard for:

- Exploring registered datasets and their profiles
- Training models with configurable parameters
- Making predictions and inspecting results
- Viewing SHAP explanations with top positive and negative contributors
- Slicing data and discovering interesting populations

Access at `http://localhost:9500` when the server is running.

## Adding a New Dataset

1. Create `datasets/<your_dataset>/dataset.yaml` with DatasetSpec, FeatureSpec, and TargetSpec. Set `target.type` to `regression` or `classification` — this determines the default model and evaluation metrics.
2. Place your data in `data/<your_dataset>/data.parquet` (or ingest from CSV).
3. Register via `svc.register_dataset_from_config(...)` or REST.
4. Train via `scripts/train_model.py` or REST. The model type is auto-selected from `target.type` if not specified.
5. Predict, explain, and analyze through the same shared services.

No core pipeline code changes required. The generic ML pipeline reads the specs and never hard-codes domain-specific logic.

## Adding a New Model

1. Create `backend/models/<your_model>.py` with a class implementing the `Predictor` contract.
2. Set `model_type` and `task_type` ("classification" or "regression").
3. Register it in `MODEL_FACTORIES` in `backend/ml/training.py`.
4. Add it to `_load_predictor` in `backend/ml/inference.py`.

The shared training, inference, and explanation paths pick it up automatically. If the model supports SHAP (e.g. tree-based), implement `explain_instance` for per-prediction explanations.

## Tests

```bash
pytest
```

The suite includes 127 tests covering:
- **Unit tests:** DatasetSpec parsing, validation, ingestion, profiling, feature transforms (with leakage assertions), target construction (entity isolation, horizon boundaries), ML training/evaluation/persistence/reload, prediction, feature parity, SHAP explainability, interesting-slice discovery.
- **Integration tests:** REST API (all endpoints), MCP tools (discovery, schemas, all tool calls, invalid inputs).
- **End-to-end acceptance test:** generate → register → ingest → profile → features → target → split → train → evaluate → persist → reload → predict → explain → REST → MCP (verifying same results).

## Technology Stack

- **Python** — core language
- **FastAPI + Uvicorn** — REST API and Web UI
- **Polars + DuckDB** — columnar data processing
- **Parquet** — data persistence
- **scikit-learn** — Logistic Regression, Linear Regression, evaluation metrics
- **XGBoost** — gradient-boosted tree models (classifier + regressor)
- **SHAP** — TreeExplainer for per-prediction explainability
- **joblib** — model artifact persistence
- **MCP SDK** — stdio server for Claude integration
- **pytest** — testing

No additional infrastructure (databases, message queues, vector stores) is required.

## Documentation

- **[README_SETUP_USAGE.md](README_SETUP_USAGE.md)** — Detailed setup, usage, model training, prediction, MCP tools, and reference model documentation.
- **[README_TEST.md](README_TEST.md)** — Testing instructions and MCP testing guide.
- **[RELEASE_NOTES.md](RELEASE_NOTES.md)** — Release notes for v0.1.
- **[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)** — Technical architecture overview.

## License

MIT
