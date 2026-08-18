# Anistroph

[![GitHub
Release](https://img.shields.io/github/v/release/vrraj/anistroph?label=release&color=orange&logo=github)](https://github.com/vrraj/anistroph/releases)
[![License:
MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/vrraj/anistroph/blob/main/LICENSE)
[![Python
3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-147%20passing-brightgreen)](https://vrraj.github.io/anistroph/setup-usage#testing)
[![MCP
Tools](https://img.shields.io/badge/MCP-13%20tools-purple)](https://github.com/vrraj/anistroph#mcp-and-agent-access)
[![GitHub
Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://vrraj.github.io/anistroph/)

**Anistroph** (derived from *anisotropy*, reflecting how insights can
shift with the direction of analysis) is a multi-domain predictive
analytics architecture for structured data.

Different datasets retain their own **schemas, features, targets,
preprocessing, and models** while using shared services for **training,
prediction, explainability, evaluation, and multidimensional analysis**.
New datasets and prediction problems can be added without rebuilding the
shared runtime.

**Claude and AI agents** can discover datasets and models, inspect
required inputs, run predictions, explain results, evaluate model
performance, and analyze data through **MCP (stdio and Streamable
HTTP)**. The same runtime is also accessible through **REST/OpenAPI and
the Anistroph Web UI**.

![Anistroph predictive analytics pipeline](https://raw.githubusercontent.com/vrraj/anistroph/main/docs/images/anistroph-pipeline.png)

### AI Agent Analysis & Validation

> Claude and AI agents can orchestrate Anistroph's prediction and
> analytical capabilities through MCP. Predictions, explanations,
> evaluations, and analyses are executed by Anistroph's shared services
> and can be independently reproduced through the Web UI or REST API
> when validation is required.

## What the Architecture Supports

-   **Multi-domain datasets** --- datasets from different domains use
    the same training, prediction, explanation, evaluation, and
    analytical services while retaining dataset-specific schemas,
    features, targets, preprocessing, and models.
-   **Declarative dataset configuration** --- YAML defines the dataset
    schema, model inputs, transforms, target semantics, and split
    strategy.
-   **Temporal and non-temporal prediction** --- temporal models can
    reconstruct rolling features from entity history at prediction time;
    non-temporal models support direct entity or records-based
    inference.
-   **Multiple targets** --- one source dataset can support independent
    models for different outcomes.
-   **Multiple model types** --- regression and classification are
    implemented today through model adapters.
-   **Process-stage prediction** --- separate models can predict the
    same outcome at different workflow stages using only information
    available at that point.
-   **Explainability** --- XGBoost predictions use TreeSHAP, with
    transformed categorical features normalized back to human-readable
    source features.
-   **Multidimensional analysis** --- discover populations where
    observed outcomes differ materially across 1-, 2-, and 3-dimensional
    combinations.
-   **Multidimensional model evaluation** --- identify populations where
    model error is materially better or worse than the overall baseline.
-   **MCP and agent access** --- Claude and other MCP-compatible agents
    can discover models, inspect input contracts, predict, explain,
    evaluate, and analyze.
-   **Cross-interface validation** --- agent-generated predictions,
    explanations, evaluations, and analyses can be reproduced against
    the same persisted model and shared runtime through REST or the Web
    UI.

> Anistroph is a reference architecture. The included synthetic datasets
> and models demonstrate how the components fit together rather than
> claiming validation across every potential domain.

------------------------------------------------------------------------

## Reference Datasets

Anistroph includes synthetic reference datasets across multiple domains to exercise different architectural capabilities. The primary reference implementations cover **semiconductor manufacturing, predictive maintenance, and semiconductor materials procurement**, with a lightweight real-estate dataset providing an additional cross-domain regression test.

| Reference domain | What it exercises |
|---|---|
| **Semiconductor Manufacturing** | Multiple regression targets, process-stage prediction, explainability, multidimensional analysis and evaluation |
| **Predictive Maintenance** | Temporal sensor data, classification + regression, rolling features, equipment-health prediction |
| **Semiconductor Materials Procurement** | Temporal prediction, rolling demand features, 4-week demand forecasting, shortage-risk classification |
| Real estate | Lightweight non-manufacturing regression example for cross-domain validation |

The same source data can support multiple target configurations. For
example, the semiconductor manufacturing source supports separate yield,
critical-dimension, and film-thickness models, each with its own
feature/target configuration, partitions, metrics, and persisted model
artifacts.

### Trained Reference Models

Held-out evaluation metrics for the shipped reference models:

| Domain | Target | Task | Held-out metric |
|--------|--------|------|-----------------|
| Semiconductor Yield | `wafer_yield` | Regression | R² = 0.81 |
| Semiconductor CD | `critical_dimension_nm` | Regression | R² = 0.89 |
| Semiconductor Film Thickness | `film_thickness_nm` | Regression | R² = 0.98 |
| Predictive Maintenance — Failure | `failure_within_horizon` | Classification | ROC-AUC = 0.85, F1 = 0.61 |
| Predictive Maintenance — Maintenance | `maintenance_required` | Classification | ROC-AUC = 1.00, F1 = 0.94 |
| Predictive Maintenance — RUL | `remaining_useful_life_hours` | Regression | MAE = 27.9h |
| Home Prices | `price` | Regression | R² = 0.97 |
| Procurement — Demand | `material_demand_next_4w` | Regression | R² = 0.96, MAE = 11.1 |
| Procurement — Shortage Risk | `shortage_risk_next_4w` | Classification | ROC-AUC = 0.99, F1 = 0.90 |

Models are trained on the train partition and evaluated on the held-out
evaluation partition (most recent 20% for temporal datasets, random 20%
for non-temporal). The two never overlap.

For dataset-specific prompts and worked examples, see the [Setup & Usage
Guide](docs/setup-usage.md).

------------------------------------------------------------------------

## Install / Setup

The quickest path is to install Anistroph, generate/register the
reference datasets, and connect Claude Desktop through MCP stdio.

### 1. Clone

``` bash
git clone https://github.com/vrraj/anistroph
cd anistroph
```

### 2. One-shot setup

``` bash
make install
```

`make install`:

-   checks for `libomp` on macOS, required by XGBoost;
-   creates `.venv` and installs Anistroph in editable mode;
-   creates `.env` from `.env.example`;
-   generates and registers the thirteen reference dataset
    configurations;
-   prints a ready-to-paste Claude Desktop MCP configuration with
    absolute paths.

### 3. Connect Claude Desktop

Paste the generated configuration into:

``` text
~/Library/Application Support/Claude/claude_desktop_config.json
```

It will look like:

``` json
{
  "mcpServers": {
    "anistroph": {
      "command": "/absolute/path/to/anistroph/.venv/bin/python",
      "args": ["-m", "backend.integrations.mcp.server"],
      "cwd": "/absolute/path/to/anistroph"
    }
  }
}
```

Restart Claude Desktop and try:

> "List all Anistroph datasets and models."
>
> "Predict wafer yield for WAFER_015000 and explain what pushed it up or
> down."
>
> "Where does the wafer-yield model perform worst across product, tool,
> and chamber combinations?"

MCP stdio does **not** require the FastAPI server to be running.

### 4. Optional: start the service

Start the service when you want REST/OpenAPI, MCP Streamable HTTP, or
the Web UI:

``` bash
make start-native    # local .venv
# or
make start           # Docker Compose
```

Access points:

-   **MCP stdio:** Claude Desktop / Claude CLI --- no server required
-   **MCP Streamable HTTP:** `http://localhost:9500/mcp`
-   **REST / OpenAPI:** `http://localhost:9500/docs`
-   **Web UI:** `http://localhost:9500`

For the full installation, MCP setup, examples, and troubleshooting
guide, see [Setup & Usage](docs/setup-usage.md).

------------------------------------------------------------------------

## Core Architecture Concepts

### Dataset and Model Isolation

Each dataset declares its own schema and predictive contract through
configuration rather than embedding domain concepts into the shared
runtime.

``` text
             Dataset
                │
                ▼
          DatasetSpec
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
 FeatureSpec TargetSpec ModelSpec
       │        │        │
       └────────┼────────┘
                ▼
         Common Pipeline
```

-   **DatasetSpec** --- columns, types, roles, entity keys, and optional
    time keys.
-   **FeatureSpec** --- model inputs and transforms, including
    categorical encoding and temporal rolling features.
-   **TargetSpec** --- target semantics such as regression,
    classification, or future event.
-   **Model artifacts** --- trained model, feature metadata,
    preprocessing contract, feature order, and evaluation metrics.

Dataset-specific preparation can be added where a domain requires it
without placing domain logic inside shared training, inference,
analysis, or transport services.

### Shared Predictive Runtime

Once trained, models participate in the same runtime regardless of
domain:

-   **Predict** --- run inference for an entity or source-feature
    record.
-   **Explain** --- identify the source features that contributed most
    to an individual prediction.
-   **Evaluate** --- measure persisted-model performance against
    held-out data, overall or for selected populations.
-   **Analyze** --- slice and compare observed data and discover unusual
    multidimensional populations.

This separation is what allows semiconductor, equipment-health,
procurement, home-price, and future datasets to use the same service
contracts.

------------------------------------------------------------------------

## Modeling and Evaluation

### Models Currently Implemented

  -----------------------------------------------------------------------
  Task                    Models                  Typical output
  ----------------------- ----------------------- -----------------------
  Classification          XGBoost Classifier,     probability / class
                          Logistic Regression     

  Regression              XGBoost Regressor,      continuous value
                          Linear Regression       
                          (Ridge)                 
  -----------------------------------------------------------------------

`TargetSpec` declares the prediction task. When a model type is not
explicitly supplied, the task determines the default model family and
evaluation metrics.

### Held-Out Evaluation

Datasets are partitioned at registration into training and evaluation
partitions. Temporal datasets split chronologically; non-temporal
datasets shuffle with a fixed seed.

Training uses only `train.parquet`. Evaluation runs the persisted model
against `evaluation.parquet` without fitting or modifying the model.

**Regression:** MAE, MSE, RMSE, R², MAPE, max error, median absolute
error, p95 absolute error, mean prediction error, and baseline
comparison.

**Classification:** ROC-AUC, PR-AUC, precision, recall, F1, and
confusion matrix.

### Multidimensional Model Evaluation

Aggregate metrics can hide populations where a model performs materially
differently. Anistroph can evaluate model error across 1-, 2-, and
3-dimensional categorical combinations.

``` text
Overall Evaluation
       │
       ├── Aggregate metrics
       │
       └── Multidimensional Evaluation
                    │
          ┌─────────┼──────────────┐
          ▼         ▼              ▼
       Product   Product × Tool   Product × Tool × Chamber
```

This separates two questions:

-   **Multidimensional analysis:** where is an observed outcome
    unusually high or low?
-   **Multidimensional model evaluation:** where does the model perform
    unusually well or poorly?

------------------------------------------------------------------------

## Temporal Prediction and Rolling Forecasts

Temporal models can use history-dependent features without retraining
for every new prediction period.

For example, a procurement model may predict **material demand over the
next four weeks** using current inventory, supplier metrics, production
plans, and recent consumption trends.

``` text
Recent Entity History
        ↓
Current Rolling Features
        ↓
Persisted Model
        ↓
Future Prediction
```

Anistroph separates:

-   **Training** --- learns relationships between features and outcomes.
-   **Temporal feature calculation** --- reconstructs current
    rolling/history-based inputs from entity history.
-   **Forecast horizon** --- defines the future period represented by
    the target.

For a rolling four-week target:

``` text
As of Week 10 → predict Weeks 11–14
As of Week 11 → predict Weeks 12–15
As of Week 12 → predict Weeks 13–16
```

The `as_of` point is the boundary of known information.
History-dependent transforms use only observations available through
that point, preventing future leakage.

The required inference history is derived from the model's configured
feature windows. New observations can therefore produce new predictions
with updated feature values while the persisted model remains unchanged.

### Temporal vs Non-Temporal Datasets

Anistroph distinguishes two dataset types based on whether a `time_key` is declared in the YAML:

| | Temporal dataset | Non-temporal dataset |
|---|---|---|
| **YAML** | `time_key: <column>` | `time_key` omitted or null |
| **Examples** | Procurement (weekly), Predictive Maintenance (5-min sensor) | Semiconductor Yield (per wafer), Home Prices (per listing) |
| **Splitting** | Chronological (oldest → train, newest → eval) | Random with fixed seed |
| **Rolling transforms** | Available (`mean`, `std`, `slope`, `delta` over time windows) | Not applicable |
| **Prediction** | Entity lookup loads history up to `as_of` date to build rolling features | Entity lookup fetches single row; records mode also available |

A dataset can have a timestamp column for chronological splitting without being "temporal" in the forecasting sense. The key distinction is whether the model's feature transforms include rolling windows — if they do, the runtime requires `entity_id + timestamp` for prediction so it can load historical observations. The `anistroph_get_model_inputs` tool exposes this via `requires_timestamp` and `prediction_mode`, so agents know which inputs to request.

For the full treatment of `as_of`, rolling feature reconstruction,
inference history, forecast horizons, and retraining, see [Setup & Usage
→ Temporal Prediction](docs/setup-usage.md#temporal-prediction).

------------------------------------------------------------------------

## Explainability

Anistroph exposes a common explanation interface while allowing the
explanation method to depend on the model family.

### XGBoost / TreeSHAP

XGBoost classification and regression models use **SHAP TreeExplainer
(TreeSHAP)**. Contributions are signed, showing which inputs pushed an
individual prediction higher or lower.

### Source-Feature Normalization

Preprocessing can expand one source field into several model features:

``` text
product_id = PROD_B

        ↓ one-hot encoding

product_id__PROD_A = 0
product_id__PROD_B = 1
product_id__PROD_C = 0

        ↓ SHAP + normalization

feature = product_id
value   = PROD_B
impact  = summed contribution
```

`anistroph_explain_prediction` groups one-hot SHAP values back to the
original source feature and returns the original input value. Raw
per-category contributions remain available in a `detail` field for
debugging.

Models without TreeSHAP support currently use an importance-weighted
fallback and identify the method through `explanation_method`.

An explanation describes **why the model produced a prediction**; it is
not treated as proof that a feature physically caused the observed
outcome.

------------------------------------------------------------------------

## Multidimensional Discovery

Prediction and explanation are complemented by deterministic analysis of
the underlying dataset.

Anistroph can slice data by one, two, or three dimensions and aggregate
metrics using mean, sum, min, max, count, standard deviation, or median.
`find_interesting_slices` searches categorical combinations and ranks
populations by deviation from the overall baseline while enforcing a
minimum sample size.

``` text
                 DATA
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
 Predictive Model       Dimensional Analysis
       │                     │
       ▼                     ▼
 Prediction            Unusual Population
       │                     │
       ▼                     ▼
    SHAP              Observed Behavior
       │                     │
       ▼                     ▼
Why did the model?     Where in the data?
```

The analytical engine is dataset-agnostic: dimensions and metrics come
from the selected dataset rather than being hard-coded for a specific
domain.

------------------------------------------------------------------------

## Multiple Targets

A single source dataset can support multiple predictive outcomes through
separate dataset configurations.

``` text
SEMICONDUCTOR MANUFACTURING
├── wafer_yield                    regression
├── critical_dimension_nm          regression
└── film_thickness_nm              regression

PREDICTIVE MAINTENANCE
├── failure_within_horizon         classification
├── remaining_useful_life_hours    regression
└── maintenance_required           classification

SEMICONDUCTOR PROCUREMENT
├── material_demand_next_4w        regression
└── shortage_risk_next_4w          classification
```

Each target receives its own feature/target contract, task type,
partitions, metrics, and persisted model artifacts while still using the
common Anistroph services.

------------------------------------------------------------------------

## Process-Stage Prediction

The semiconductor reference implementation includes four stage
configurations (`semiconductor_yield_stage_a` through `_d`) that predict
the same target using progressively larger feature sets.

``` text
Before Etch ─→ ETCH ─→ After Etch ─→ DEPOSITION ─→ LITHO ─→ Final Test
     │                       │                                  │
     ▼                       ▼                                  ▼
  Model A                 Model B                            Model D
     │                       │                                  │
planned features       + actual etch                   + complete process
                       measurements                        history
     │                       │                                  │
     ▼                       ▼                                  ▼
Yield prediction       Yield prediction                  Yield prediction
```

Each model is trained only on features available at its prediction
point. This allows predictions to become progressively better informed
as more process information becomes available without leaking
future-process data into earlier-stage models.

**Current limitation:** each stage configuration copies the source
Parquet during registration. A future shared-Parquet mode could allow
multiple configurations to reference the same underlying data.

------------------------------------------------------------------------

## MCP and Agent Access

Anistroph exposes **13 domain-agnostic MCP tools** over both **stdio**
and **Streamable HTTP**. The tools operate against registered dataset
and model metadata rather than being duplicated for each domain.

  -------------------------------------------------------------------------
  Tool                                  Description
  ------------------------------------- -----------------------------------
  `anistroph_list_datasets`             Discover registered datasets and
                                        basic metadata

  `anistroph_profile_dataset`           Inspect schema, distributions,
                                        missing values, and time range

  `anistroph_slice_data`                Slice data by 1--3 dimensions and
                                        aggregate a metric

  `anistroph_compare_data`              Compare a metric across dimension
                                        values

  `anistroph_find_interesting_slices`   Find multidimensional populations
                                        with large baseline deviations

  `anistroph_sample_rows`               Retrieve filtered/sample source
                                        rows

  `anistroph_list_models`               Discover trained models and
                                        task/model metadata

  `anistroph_get_model_metrics`         Retrieve persisted model metrics

  `anistroph_get_model_inputs`          Discover required source inputs,
                                        transforms, and prediction mode

  `anistroph_predict`                   Run inference using persisted
                                        preprocessing and model artifacts

  `anistroph_explain_prediction`        Explain an individual prediction

  `anistroph_evaluate_model`            Evaluate a persisted model against
                                        held-out data

  `anistroph_find_evaluation_slices`    Find populations where model error
                                        differs from the overall baseline
  -------------------------------------------------------------------------

Model interaction is self-describing: an agent can discover a model,
inspect its required source-level inputs, and then invoke prediction or
explanation without constructing engineered model features itself.

Training, dataset registration, and deletion are intentionally not
exposed through MCP. They remain administrative operations through
Python, REST, CLI, or the Web UI.

### Agent-to-UI Validation

Because MCP, REST, and the Web UI call the same `AnistrophServices`
layer, an agent-generated test case can be reproduced independently.

``` text
Claude / AI Agent
       ↓
Discover Model + Input Schema
       ↓
Generate / Select Source-Feature Record
       ↓
Predict • Explain • Evaluate • Analyze
       ↓
Retain Model + Inputs + Operation
       ↓
Re-run through REST or Web UI
       ↓
Validate Result
```

Records-based inference accepts ordinary source-feature values.
Anistroph applies the same persisted feature metadata and preprocessing
used during training, making the source-feature JSON a portable test
case across interfaces.

------------------------------------------------------------------------

## System Architecture & Technology Stack

Anistroph keeps user-facing interfaces thin and routes them through the
same core services.

  -----------------------------------------------------------------------
  Layer                   Technology              Role
  ----------------------- ----------------------- -----------------------
  Language                **Python**              Core services, data
                                                  preparation, ML
                                                  orchestration

  API / Service           **FastAPI + Uvicorn**   REST API and Web UI
                                                  service layer

  Data processing         **Polars + DuckDB**     Columnar
                                                  transformations,
                                                  querying, analytical
                                                  slicing

  Data persistence        **Parquet**             Dataset storage

  Dataset configuration   **YAML**                Declarative schemas,
                                                  features, targets, and
                                                  split strategy

  ML                      **scikit-learn**        Logistic Regression,
                                                  Linear/Ridge
                                                  Regression, metrics

  Gradient boosting       **XGBoost**             Classification and
                                                  regression

  Explainability          **SHAP TreeExplainer +  TreeSHAP for XGBoost;
                          common explanation      model-specific
                          interface**             explanation behind a
                                                  shared contract

  Model artifacts         **joblib**              Model persistence and
                                                  reload

  Agent integration       **MCP SDK**             Runtime tools over
                                                  stdio and Streamable
                                                  HTTP

  Testing                 **pytest**              Unit, integration, MCP,
                                                  and end-to-end tests

  Containerization        **Docker**              Containerized runtime
                                                  option
  -----------------------------------------------------------------------

No database, message queue, or vector store is required by the current
reference implementation.

------------------------------------------------------------------------

## Synthetic Reference Data

Reproducible synthetic-data generators under `scripts/` exercise the
architecture without requiring proprietary datasets.

-   **Predictive Maintenance** --- 50 machines × 60 days of 5-minute
    sensor observations, with failure, remaining-useful-life, and
    maintenance targets.
-   **Semiconductor Manufacturing** --- 50,000 wafer rows spanning
    product, fab, process route, tools, chambers, recipes, process
    conditions, maintenance state, and three regression targets.
-   **Bay Area Home Prices** --- 40,000 synthetic listings across San
    Jose, Los Gatos, and Saratoga.
-   **Semiconductor Materials Procurement** --- roughly 100,000 weekly
    `week × fab × material` rows across 8 fabs, 100 materials, 15
    suppliers, and 160 weeks, supporting demand and shortage-risk
    targets.

The generators use a fixed seed by default and intentionally include
noise and interactions so the datasets are learnable but imperfect. They
are reference/demo data, not benchmark datasets or evidence of
real-world model performance.

------------------------------------------------------------------------

## Adding a Dataset

A new dataset does not require changes to the shared prediction,
explanation, evaluation, or analysis services.

1.  Create `datasets/<dataset>/dataset.yaml` defining schema, features,
    target, and split strategy.
2.  Provide the source data as CSV or Parquet.
3.  Add dataset-specific preparation only where the domain requires it.
4.  Register the dataset.
5.  Train and evaluate a model.
6.  Use the same Python, REST, MCP, and Web UI runtime services.

Minimal structure:

``` yaml
dataset:
  dataset_id: <id>
  name: <name>
  entity_key: <entity column>
  time_key: <optional timestamp column>
  columns:
    <column>: {type: numeric|categorical|boolean|timestamp, role: identifier|feature|target|event|metadata}

features:
  <feature>:
    column: <source column>
    transforms: [current|categorical]

target:
  name: <target name>
  type: regression|classification|binary|future_event
  source_column: <label column>

split:
  strategy: chronological|random
  train: 0.80
  eval: 0.20
```

Registration validates the source against the configuration, persists
the dataset, profiles it, creates train/evaluation partitions, and
records dataset metadata. Feature transforms and target construction
occur during training.

For the complete YAML schema, transform reference, temporal features,
registration workflow, and worked examples, see [Setup & Usage →
Configure Your Own
Dataset](docs/setup-usage.md#configure-your-own-dataset).

------------------------------------------------------------------------

## Adding a Model

Model families participate in the common runtime through adapters:

1.  add a model adapter under `backend/models/`;
2.  declare its `model_type` and supported task;
3.  register it with `MODEL_FACTORIES`;
4.  add load behavior to the inference layer;
5.  provide model-specific explanation behavior where supported.

The model then participates in the common training, persistence,
inference, evaluation, and runtime architecture.

Potential extension paths include multiclass classification, specialized
forecasting, anomaly detection, additional explainers, model
versioning/promotion, and monitoring/drift detection.

------------------------------------------------------------------------

## Training and Runtime Examples

### Train

``` bash
python scripts/train_model.py --dataset semiconductor_yield \
  --model-type xgboost_regressor --model-id my-wafer-yield-model
```

### Predict

``` python
from backend.services import get_services

svc = get_services()

pred = svc.predict(
    model_id="my-wafer-yield-model",
    entity_id="WAFER_015000",
)
```

### Explain

``` python
expl = svc.explain(
    model_id="my-wafer-yield-model",
    entity_id="WAFER_015000",
    top_k=10,
)
```

### Discover multidimensional patterns

``` python
interesting = svc.find_interesting_slices(
    "semiconductor_yield",
    "wafer_yield",
    top_k=20,
)
```

------------------------------------------------------------------------

## Interfaces

-   **MCP stdio** --- local Claude Desktop/CLI and other MCP-compatible
    clients; no FastAPI server required.
-   **MCP Streamable HTTP** --- `/mcp` for remote MCP clients and tool
    routers.
-   **REST / OpenAPI** --- dataset management, training, model
    discovery, prediction, explanation, evaluation, and analysis.
-   **ChatGPT / GPT Actions** --- `/openapi-gpt.json` exposes
    runtime-only endpoints; `make start-gpt` starts the service with an
    ngrok tunnel for cloud access.
-   **Web UI** --- direct dataset exploration, model training,
    prediction, explanation, slicing, evaluation, and cross-interface
    validation.

See [Setup & Usage](docs/setup-usage.md) for interface configuration and
example workflows.

------------------------------------------------------------------------

## Tests

``` bash
pytest
```

The current suite contains **147 tests** spanning dataset
specifications, ingestion, feature transforms and leakage checks, target
construction, model training/evaluation/persistence/reload, inference,
feature parity, SHAP explainability, multidimensional discovery, REST,
MCP, and end-to-end workflows.

------------------------------------------------------------------------

## Documentation

-   **[Setup & Usage](docs/setup-usage.md)** --- installation,
    Claude/MCP usage, dataset configuration, temporal prediction,
    operations, examples, and API reference
-   **[Technical Architecture](docs/technical-architecture.md)** ---
    deeper implementation and architecture details
-   **[GitHub Pages](docs/index.md)** --- project documentation landing
    page
-   **[Release Notes](RELEASE_NOTES.md)** --- release history

## License

MIT
