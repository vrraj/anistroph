# Anistroph

**A domain-agnostic reference architecture for predictive analytics, explainability, and multidimensional discovery across structured datasets.**

Anistroph provides a common predictive lifecycle while allowing each dataset to retain its own schema, features, targets, preprocessing, and models.

```text
                  DATASETS (Multi-domain)
                            │
             Data Preparation / Feature Engineering
                            │
                     Train / Eval Split
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
        MODEL TRAINING                 EVALUATION
             │                             │
       train.parquet                 evaluation.parquet
             │                             │
             ▼                             ▼
        Train Model              Evaluate Persisted Model
             │                             │
             ▼                             ▼
       Persist Model                Model Metrics
                                   + Multidimensional
                                      Evaluation
             │                             │
             └──────────────┬──────────────┘
                            ▼
                     SHARED RUNTIME
                  Predict • Explain • Analyze
                            │
                  MCP • REST/OpenAPI • UI
                            │
              Claude • MCP Agents • Applications
```

### What the architecture supports

- **Multi-domain datasets** — add datasets from different domains without changing the shared prediction, explanation, evaluation, or multidimensional slicing services. Dataset-specific schemas, features, targets, and preprocessing remain isolated behind common runtime contracts.
- **Multiple targets** — one source dataset can train independent models for different outcomes.
- **Multiple model types** — regression and classification today, with additional task/model types extensible through adapters.
- **Process-stage prediction** — models can predict at different points in a workflow using only features available at that stage.
- **Multidimensional evaluation** — evaluate a model overall, then automatically compare its error across meaningful 1-, 2-, and 3-dimensional populations such as `Product`, `Product × Tool`, and `Product × Tool × Chamber` to identify where model performance is materially better or worse.
- **Claude & agentic access** — Claude Desktop, Claude Code, and other MCP-compatible agents can discover models, inspect required inputs, generate test records, predict, explain, evaluate, and analyze.
- **Cross-interface validation** — Claude/agent-generated inference for a model — predictions, explanations, and summaries — can be cross-validated in the Anistroph Web UI using the same source-feature JSON and persisted model/runtime.


Anistroph ships with a few **synthetic reference datasets** (semiconductor manufacturing, predictive maintenance, Bay Area home prices) that exercise the architecture across regression, classification, temporal, and multidimensional patterns. You can **add your own dataset** by authoring a `dataset.yaml` and registering it — see [Adding a Dataset](#adding-a-dataset) for the process summary and [README_SETUP_USAGE.md](README_SETUP_USAGE.md) for the full YAML reference and worked examples.

> Anistroph is a reference architecture. The included datasets and models demonstrate how the components fit together rather than claiming validation across every potential domain.

---

## At a Glance

Anistroph separates **dataset-specific modeling** from a **shared predictive runtime**. Each dataset can define its own features, targets, preprocessing, and model artifacts while using common services for inference, explainability, evaluation, multidimensional analysis, and runtime access.

```text
Semiconductor ──→ Yield / CD / Film Models ──────┐
Maintenance ────→ Failure / RUL Models ──────────┤
Home Prices ────→ Price Model ───────────────────┼─→ Shared Runtime
Future Domains ─→ Domain-Specific Models ────────┘
                                                   │
                                      Predict • Explain • Evaluate
                                                   │
                                          MCP • REST • UI
```

- **Dataset-specific:** schema, features, targets, preprocessing, feature metadata, and model artifacts.
- **Shared:** training/evaluation services, persistence, inference, explainability, multidimensional analysis, and interfaces.
- **Extensible:** additional targets, model families, process-stage predictions, domains, and agent integrations.

At runtime, Anistroph supports complementary workflows:

- **Predict** — what outcome does the selected model predict for this record or entity?
- **Explain** — which source features contributed most to that individual prediction?
- **Evaluate** — how well does the persisted model perform on unseen held-out data, both overall and across multidimensional populations?
- **Discover** — where are unusually high/low outcomes or important combinations concentrated in the underlying data?
- **Generate & test with Claude / agents** — discover a model and its required input features, generate a valid synthetic test record, then run prediction and explanation through the shared runtime.
- **Cross-validate** — reproduce the same agent-generated record in the Anistroph Web UI and compare the prediction, explanation, and resulting summary.

Explanation, evaluation, and observed-data analysis provide different perspectives and are not treated as proof of causality.

---


## Install / Setup

Install process:

1. **Clone and install** — venv + `pip install -e .`
2. **Generate and register datasets** — `make setup`
3. **Connect Claude Desktop** — add MCP server config (no server required)
4. **(Optional) Start the server** — `make start-native` or `make start`


**1. Clone and install:**

```bash
git clone https://github.com/vrraj/anistroph
cd anistroph

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **macOS note:** XGBoost requires `libomp`. Install it with
> `brew install libomp` if you see an XGBoost library loading error.

**2. Generate and register reference datasets:**

```bash
make setup
```

This runs the three synthetic data generators (predictive maintenance, semiconductor yield, Bay Area home prices) and registers all eleven dataset configs (multi-target and staged-prediction variants share source parquet files). It is **idempotent** — re-running skips datasets that are already generated and registered. See `scripts/setup_datasets.py` for flags (`--skip-gen`, `--force`).

**3. Connect Claude Desktop via MCP (stdio — no server required):**

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

Restart Claude Desktop and ask:

> "List all Anistroph datasets"
> "Find interesting slices in the semiconductor yield dataset"
> "Predict wafer yield for WAFER_015000 and explain what pushed it up or down"

> For more sample prompts check [README_SETUP_USAGE.md → Example MCP prompts](README_SETUP_USAGE.md#example-mcp-prompts).

**4. (Optional) Start the server for Web UI, REST, or MCP HTTP:**

> Start the server only if you want the browser-based Web UI, programmatic REST/OpenAPI access, or MCP over Streamable HTTP for remote clients.

```bash
make start-native    # local .venv, no Docker required
# or
make start           # Docker Compose
```

Primary access points:

- **MCP (stdio):** Claude Desktop / Claude CLI — no server required (step 3 above)
- **MCP (Streamable HTTP):** http://localhost:9500/mcp — requires server (step 4)
- **Web UI:** http://localhost:9500 — requires server (step 4)
- **REST / OpenAPI:** http://localhost:9500/docs — requires server (step 4)


---

## Dataset and Model Isolation

Each use case is represented by an isolated dataset configuration and its associated model artifacts.

```text
data/
├── predictive_maintenance/
│   └── data.parquet
└── semiconductor_yield/
    └── data.parquet

artifacts/models/
├── anistroph-sentinel-v1/
└── wafer-yield-xgb-v001/
```

A dataset describes its own schema and semantics through `DatasetSpec`, `FeatureSpec`, and `TargetSpec`:

```text
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

- **DatasetSpec** — columns, types, roles, entity keys, and time keys.
- **FeatureSpec** — feature transforms and categorical encoding.
- **TargetSpec** — target semantics such as regression, binary classification, or future event.
- **Model artifacts** — trained model, feature metadata, preprocessing information, feature order, and evaluation metrics.

Dataset-specific preparation can be added where a domain requires it without placing domain concepts inside the shared training, inference, analysis, or transport layers.

Feature transforms such as rolling windows, slopes, and deltas use only observations up to and including the current time, preventing future data from leaking into training or inference.

---

## Modeling Architecture

### Models currently implemented

| Task | Models | Typical output |
|---|---|---|
| Classification | XGBoost Classifier, Logistic Regression | probability / class |
| Regression | XGBoost Regressor, Linear Regression (Ridge) | continuous value |

### Evaluation

Every registered dataset is partitioned at registration into `train.parquet` and `evaluation.parquet` (80/20 by default; temporal datasets split chronologically, others shuffle with a fixed seed). Training uses only the train partition; the held-out evaluation partition is never used during model fitting.

Evaluation is separate from training. A persisted model is run against the held-out `evaluation.parquet` partition, predictions are compared with known outcomes, and metrics are recomputed without fitting or modifying the model.

**Regression metrics:** MAE, MSE, RMSE, R², MAPE, max error, median absolute error, 95th-percentile absolute error, mean prediction error, and baseline comparison.

**Classification metrics:** ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrix.

### Slice-Level and Multidimensional Model Evaluation

Evaluation can also be filtered to a specific population, allowing overall model performance to be compared with a segment such as a city, lot, product, tool, or chamber.

`find_evaluation_slices` extends this across 1-, 2-, and 3-dimensional categorical combinations and ranks populations by how much their prediction error deviates from the overall error baseline.

```text
Overall Evaluation
       │
       ├── Aggregate metrics
       │     MAE / RMSE / R² / MAPE
       │     ROC-AUC / PR-AUC / F1
       │
       └── Multidimensional Model Evaluation
                    │
          ┌─────────┼──────────────┐
          ▼         ▼              ▼
       Product   Product × Tool   Product × Tool × Chamber
```

This answers a different question from multidimensional data analysis:

- **Multidimensional analysis:** Where in the data is the outcome unusually high or low?
- **Multidimensional model evaluation:** Where does the model perform unusually well or poorly?

Evaluation is available through Python, REST, MCP (`anistroph_evaluate_model`, `anistroph_find_evaluation_slices`), and the Web UI.


### Task-Driven Model Selection

`TargetSpec` declares the prediction task and target semantics. The task type determines the default model family and evaluation metrics when a model type is not explicitly supplied.

| Task type | Default implementation | Evaluation |
|---|---|---|
| Regression | XGBoost Regressor | MAE, MSE, RMSE, R², MAPE, error distributions |
| Classification / binary / future event | XGBoost Classifier | ROC-AUC, PR-AUC, precision, recall, F1 |

Alternative implemented models such as Linear/Ridge Regression and Logistic Regression can be selected explicitly. Additional model families can be added through the model-adapter contract.


## Explainability

Anistroph exposes a **common explainability interface**, but the explanation technique depends on the underlying model family. Explainability is therefore treated as a model capability rather than assuming that one explainer works for every model.

### Source-Feature Explanation Normalization

Preprocessing can expand one source feature into several model features. For example, `product_id = PROD_B` may become:

```text
product_id__PROD_A = 0
product_id__PROD_B = 1
product_id__PROD_C = 0
```

SHAP operates on those transformed features, but agent-facing explanations are more useful when mapped back to the original input:

```text
One-hot SHAP values
       ↓
Group by source feature
       ↓
Sum contributions
       ↓
Attach original input value
       ↓
Human-readable explanation
```

Instead of exposing three one-hot contributions, the normalized response can return:

```json
{
  "feature": "product_id",
  "value": "PROD_B",
  "impact": 0.001001
}
```

**Explanation normalization requirement:** when preprocessing expands one source feature into multiple transformed model features, `anistroph_explain_prediction` aggregates their SHAP contributions back to the original source feature and returns the original feature name, original input value, and summed contribution. Raw per-category SHAP contributions are retained in a `detail` field for debugging but are not the default MCP response.


### Current implementation

- **XGBoost classification and regression models** use **SHAP TreeExplainer (TreeSHAP)** for per-prediction explanation.
- TreeSHAP returns signed feature contributions showing which inputs pushed a prediction higher or lower.
- **Models without TreeSHAP support** currently use an importance-weighted fallback. The response identifies the method used through `explanation_method`.


### Model-Specific Explainers

The architecture can support model-specific explainers behind the same interface. Examples include:

| Model family | Possible explanation approach |
|---|---|
| XGBoost / Random Forest / other tree models | TreeSHAP |
| Logistic / Linear models | Coefficients or feature-level contributions |
| Other black-box models | KernelSHAP, permutation-based methods, or another compatible explainer |
| Specialized forecasting / neural models | Model-specific explanation adapter |

These are architectural extension paths, not claims that every explainer above is implemented today.

Feature identities and ordering are preserved through training and persistence so explanations can be mapped back to meaningful source fields.

The same runtime and MCP tool can therefore expose prediction explanations across model families while the model adapter selects the appropriate explanation method.

Example question:

> The model predicted 88.3% wafer yield. Which inputs pushed the prediction up or down?

An explanation describes **why the model produced a prediction**. It should not be interpreted as proof that a feature physically caused the observed outcome.

---

## Multidimensional Discovery

Prediction is complemented by deterministic analysis of the underlying dataset.

Anistroph can slice data by one, two, or three dimensions and aggregate metrics using mean, sum, min, max, count, standard deviation, or median. `find_interesting_slices` searches combinations of categorical dimensions and ranks populations by deviation from the overall baseline while enforcing a minimum sample size.

```text
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

The analytical engine is dataset-agnostic: dimensions and metrics come from the selected dataset rather than being hard-coded for semiconductor, maintenance, or another domain.

---

## Agent-to-UI Validation Workflow

Because MCP, REST, and the Web UI call the same `AnistrophServices` layer, an agent-generated prediction can be reproduced directly in the UI.

```text
Claude / MCP Agent
       ↓
Discover Models
       ↓
Get Model Input Schema
(anistroph_get_model_inputs)
       ↓
Generate Source-Feature Test Record
       ↓
Predict by Record
       ↓
Explain
       ↓
Generate / retain Feature JSON
       ↓
Paste JSON into Anistroph Web UI
       ↓
Re-run Prediction + Explanation
       ↓
Validate Agent Result
```

This is enabled by **records-based inference**: the caller supplies ordinary source feature values rather than engineered model features. Anistroph applies the same persisted feature metadata and preprocessing used during training.

Prediction supports two modes: **entity lookup** (send `entity_id` + optional `timestamp`; Anistroph loads the row from the dataset) and **records** (send raw source feature values for a new or hypothetical row). In both cases the caller supplies source-level values — Anistroph applies the same persisted feature metadata and preprocessing as training.

The feature JSON therefore becomes a portable test case across interfaces. A record generated by an MCP-compatible agent can be run through MCP and then copied into the Web UI to reproduce the prediction and explanation against the same persisted model.

This provides a practical validation path for agentic workflows: **model discovery → schema discovery → synthetic test record → prediction → explanation → cross-interface reproduction**.


## MCP and Agent Access

The current MCP runtime exposes **13 tools**, including model input-schema discovery, records-based prediction, normalized explanations, held-out evaluation, and multidimensional error-slice discovery.


Anistroph currently supports two MCP transports over the same service layer: **stdio** for local MCP clients such as Claude Desktop/CLI, Cursor, and Cline, and **Streamable HTTP** at `/mcp` for remote MCP clients and tool routers. Both expose the same runtime tool contract.

Model interaction is self-describing: agents can discover models and inspect required source-level inputs before prediction. The complete cross-interface workflow is shown in [Agent-to-UI Validation Workflow](#agent-to-ui-validation-workflow).


The MCP tools are registered as **domain-agnostic capabilities**. They operate against the schema and model metadata of the selected dataset rather than being duplicated for each domain.


### Runtime MCP tools

| Tool | Description |
|---|---|
| `anistroph_list_datasets` | Discover registered datasets and basic metadata. |
| `anistroph_profile_dataset` | Inspect a dataset's schema, column types, distributions, missing values, and time range. |
| `anistroph_slice_data` | Slice a selected dataset by 1–3 dimensions and aggregate a selected metric using fields defined by that dataset. |
| `anistroph_compare_data` | Compare a metric across values of a selected dimension. |
| `anistroph_find_interesting_slices` | Search the selected dataset for multidimensional populations with large deviations from the overall baseline. |
| `anistroph_sample_rows` | Retrieve sample rows, optionally filtered by dataset fields. |
| `anistroph_list_models` | Discover trained models, their datasets, task/model types, and timestamps. |
| `anistroph_get_model_metrics` | Retrieve evaluation metrics for a persisted model. |
| `anistroph_get_model_inputs` | Return required source columns, types, transforms, and supported prediction mode for a trained model |
| `anistroph_predict` | Run inference using a registered model and its persisted feature/preprocessing metadata. |
| `anistroph_explain_prediction` | Explain an individual prediction using the explanation method supported by the model. |
| `anistroph_evaluate_model` | Evaluate a persisted model against held-out data, optionally for a filtered population. |
| `anistroph_find_evaluation_slices` | Find multidimensional populations where model error differs most from the overall baseline. |

The same tools can therefore operate on different datasets. For example, `anistroph_slice_data` can slice wafer yield by tool/chamber/recipe or a customer dataset by segment/plan/region without creating separate MCP tools.

Training and dataset administration are intentionally not exposed through MCP; they remain administrative operations through Python, REST, CLI, or the Web UI.

---


## Synthetic Test Data Generation

Anistroph includes reproducible synthetic-data generators under `scripts/` to exercise different domains, target types, feature interactions, and multidimensional discovery patterns without requiring proprietary datasets.

### Predictive Maintenance — `generate_sensor_data.py`

**Domain:** manufacturing tool-health monitoring  
**Shape:** 50 machines × 60 days × 5-minute intervals (~864K rows)

Each tool receives its own baseline operating profile and deterioration rate, with maintenance cycles resetting deterioration. Generated signals include temperature, vibration, pressure, current, voltage, RPM, flow rate, maintenance age, and operating hours, with a small amount of anomalous sensor behavior.

Targets include:

- `failure` — binary failure outcome driven by a weighted equipment-risk signal;
- `remaining_useful_life_hours` — regression target derived from time to the next failure;
- `maintenance_required` — binary maintenance-intervention target.

The generator also produces a `failure_mode` column (`NONE`, `THERMAL`, `PRESSURE`, `VIBRATION`, `POWER`) as synthetic data for the multiclass-classification extension path. It is not a currently supported target — the runtime today handles binary classification and regression only.

### Semiconductor Manufacturing — `generate_semiconductor_yield_data.py`

**Domain:** semiconductor fab process — lithography → etch → deposition  
**Shape:** 50,000 wafer rows (~2 years)

The generator produces product/fab/process-route identifiers, tool/chamber/recipe combinations, etch and deposition process parameters, lithography conditions, and maintenance state.

Three regression targets are generated:

- `wafer_yield` — baseline around 0.975 with continuous penalties and embedded interactions, including the `ETCH_02 × CH_B × high temperature variability` slice used for multidimensional-discovery demonstrations;
- `critical_dimension_nm` — approximately 38 nm, influenced by exposure dose, focus offset, etch conditions, product, and an embedded tool/recipe interaction;
- `film_thickness_nm` — approximately 510 nm, influenced by deposition time, tool/recipe, pressure, temperature, maintenance age, and an embedded deposition interaction.

### Bay Area Home Prices — `generate_home_prices_data.py`

**Domain:** real estate  
**Shape:** 40,000 synthetic listings across San Jose, Los Gatos, and Saratoga

Features include city, ZIP code, square footage, bedrooms, bathrooms, lot size, year built, and garage stalls.

The regression target `price` is generated from location-specific price-per-square-foot baselines plus property-size effects, bedroom/bathroom premiums, property age, lot size, garage capacity, and controlled noise.

### Common Generator Design

- **Reproducible** — `seed=42` by default.
- **Learnable but imperfect** — targets contain noise and interactions rather than being determined by a single feature.
- **Discovery-oriented** — embedded interactions create known patterns that can be used to exercise multidimensional analysis and evaluation.
- **Multi-target** — semiconductor and predictive-maintenance source Parquet files support multiple target configurations.
- **Output** — generated Parquet files are written under `data/<dataset>/data.parquet` and remain gitignored.

These generators create **reference/demo data**, not benchmark datasets or evidence of real-world model performance.

---

## Multiple Targets per Dataset

A single source dataset can support multiple prediction targets. Anistroph represents each target through its own dataset configuration — for example `semiconductor_yield`, `semiconductor_cd`, and `semiconductor_film_thickness` can point to the same source Parquet data while defining different `target:` sections. Each target receives independent partitions, training, evaluation, model artifacts, and task-appropriate metrics.

```text
SEMICONDUCTOR MANUFACTURING
│
├── wafer_yield                   regression
├── critical_dimension_nm         regression
└── film_thickness_nm             regression

TOOL / PREDICTIVE MAINTENANCE
│
├── failure                       classification
├── remaining_useful_life_hours   regression
└── maintenance_required          classification
```


Each target retains its own task type, metrics, feature/preprocessing metadata, partitions, and persisted model artifact. Multiclass classification remains an architectural extension rather than part of the current predictive-maintenance target set.

---

## Reference Implementations

### Semiconductor Manufacturing — Multiple Regression Targets

The semiconductor reference dataset represents wafer-level manufacturing history across products, process routes, tools, chambers, recipes, operating conditions, maintenance state, lithography conditions, and measured manufacturing outcomes.

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
Wafer Yield
```

The included synthetic dataset contains 50,000 wafer rows. The synthetic generator contains learnable nonlinear relationships involving tool/chamber combinations, temperature variability, maintenance age, and product/recipe combinations. A trained XGBoost regressor on the yield target typically achieves R² around 0.8, demonstrating that the reference pipeline can learn the relationships embedded in the generated dataset. Train your own model with `make setup` followed by `python scripts/train_model.py --dataset semiconductor_yield --model-type xgboost_regressor`.

### Predictive Maintenance — Classification + Regression

The predictive-maintenance reference dataset exercises a different problem structure: temporal equipment/sensor data supporting multiple equipment-health targets.

```text
Tool
  ↓
Sensor Measurements
  ↓
Operating Conditions
  ↓
Equipment History
  ↓
Predictive Model
  ↓
Failure / Maintenance Risk
```

It contains 50 machines and 60 days of 5-minute observations. The currently documented reference model uses a 24-hour future-failure target; the additional targets broaden the dataset across binary classification and regression. Train your own model with `python scripts/train_model.py --dataset predictive_maintenance --model-type xgboost`.

---


### Bay Area Home Prices — Regression

A third reference domain uses synthetic Bay Area housing data across San Jose, Saratoga, and Los Gatos. The model predicts home sale price from structured property and location features, demonstrating that the same Anistroph runtime can operate outside manufacturing.

| Target | Task | Represents |
|---|---|---|
| `price` | Regression | Home sale price (USD) |

## System Architecture & Technology Stack

Anistroph keeps user-facing interfaces thin and routes them through the same core services. Dataset-specific schemas and models remain isolated while prediction, explanation, evaluation, multidimensional analysis, persistence, and interface access share common contracts.

| Layer | Technology | Role |
|---|---|---|
| Language | **Python** | Core services, data preparation, ML orchestration |
| API / Service | **FastAPI + Uvicorn** | REST API and Web UI service layer |
| Data processing | **Polars + DuckDB** | Columnar transformations, querying, analytical slicing |
| Data persistence | **Parquet** | Efficient dataset storage |
| Dataset configuration | **YAML** | Declarative dataset schemas, features, targets, and split strategy |
| ML | **scikit-learn** | Logistic Regression, Linear/Ridge Regression, metrics |
| Gradient boosting | **XGBoost** | Classification and regression models |
| Explainability | **Common explanation interface + SHAP TreeExplainer** | TreeSHAP for XGBoost today; model-specific explainers can be added behind the same runtime contract |
| Model artifacts | **joblib** | Model persistence and reload |
| Agent integration | **MCP SDK** | Domain-agnostic runtime tools over stdio and Streamable HTTP |
| Testing | **pytest** | Unit, integration, MCP, and end-to-end tests |
| Containerization | **Docker** | Containerized development/runtime option |

No database, message queue, or vector store is required by the current reference implementation.

---

## Extending Anistroph

The architecture can extend along two complementary dimensions:

| Task type | Example | Model family |
|---|---|---|
| **Regression** | Predict wafer yield | XGBoost Regressor, Linear/Ridge |
| **Classification** | Predict whether a tool will fail | XGBoost Classifier, Logistic Regression |
| **Multiclass Classification** *(extension)* | Predict the most likely failure mode | Multiclass-capable classifier |
| **Forecasting** *(extension)* | Forecast demand for the next 12 weeks | Time-series / forecasting models |
| **Anomaly Detection** *(extension)* | Detect unusual sensor behavior | Anomaly-detection models |


### Example Metrics for Additional Task Types

Evaluation metrics remain task-specific rather than domain-specific:

| Example problem | Example metrics | What they indicate |
|---|---|---|
| Demand forecasting | MAE, RMSE, WAPE, Bias / Mean Error | Forecast error, large misses, relative error, and systematic over/under-prediction |
| Customer churn / propensity | ROC-AUC, PR-AUC, Precision, Recall, F1 | Ranking and classification quality, especially for imbalanced outcomes |
| Real-estate valuation | MAE, RMSE, R² | Prediction error and variance explained |

These are extension examples, not results from current Anistroph reference implementations.

### Multiclass Classification

A natural extension of the current binary classification path is **multiclass classification**, where a model returns a probability for each possible class. For example, the `failure_mode` column already generated by the predictive-maintenance data generator could classify the likely type of tool failure (THERMAL, PRESSURE, VIBRATION, POWER). This would extend the task/model abstraction from binary outcomes to multiple possible outcomes while retaining the same lifecycle.

### Process-Stage Prediction

Process-stage prediction is implemented today — the semiconductor reference dataset includes four stage configs (`semiconductor_yield_stage_a` through `_d`) that predict the same target (wafer yield) using progressively larger feature sets representing different points in the manufacturing flow.

```text
                         PROCESS FLOW

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

Each model has a different **feature availability boundary**:

| Prediction point | Features available |
|---|---|
| **Pre-Etch** | Product, route, recipe, setpoints, and assigned equipment if known |
| **Post-Etch** | Pre-Etch information + actual tool/chamber, temperature, pressure, RF power, and process time |
| **Pre-Final-Test** | Prior process history + deposition, lithography, and available metrology measurements |

Predictions become progressively better informed as a wafer moves through the process, with each stage trained only on features available at that point — no future-process leakage.

**Known limitation — parquet duplication:** each stage config currently copies the source parquet at registration. A future "shared parquet" mode would let multiple configs reference the same data without duplication.

Other extensions: model-specific explainability adapters, model versioning and promotion, monitoring/drift detection.

---

## Training and Runtime Examples

### Train a model

```bash
python scripts/train_model.py --dataset semiconductor_yield \
  --model-type xgboost_regressor --model-id my-wafer-yield-model
```

### Predict

```python
from backend.services import get_services

svc = get_services()

pred = svc.predict(
    model_id="my-wafer-yield-model",
    entity_id="WAFER_015000",
)
```

### Explain

```python
expl = svc.explain(
    model_id="my-wafer-yield-model",
    entity_id="WAFER_015000",
    top_k=10,
)
```

### Discover multidimensional patterns

```python
interesting = svc.find_interesting_slices(
    "semiconductor_yield",
    "wafer_yield",
    top_k=20,
)
```

---

## Interfaces

### REST

Start the service:

```bash
uvicorn backend.main:app --reload --port 9500
```

The REST API covers dataset management, profiling, analysis, training, model discovery, metrics, prediction, batch prediction, and explanation. Swagger is available at `http://localhost:9500/docs` while the service is running.

### Claude Desktop / Claude Code

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

Example prompts:

> "List the available Anistroph datasets and models."
>
> "Find interesting yield combinations in the semiconductor dataset."
>
> "Predict wafer yield for WAFER_015000 and explain the strongest drivers."

### Web UI

The Web UI supports dataset exploration, model training, prediction, SHAP explanation, slicing, and interesting-population discovery.

### ChatGPT / GPT Actions

The filtered OpenAPI spec at `/openapi-gpt.json` exposes runtime-only endpoints (prediction, explanation, evaluation, analysis) to ChatGPT and other OpenAPI-consuming agents — training and dataset administration are excluded. Use `make start-gpt` to start the server with an ngrok tunnel for cloud-based access. See [README_SETUP_USAGE.md](README_SETUP_USAGE.md) for detailed setup.

### Docker

```bash
make start
```

---

## Adding a Dataset

1. Create `datasets/<dataset>/dataset.yaml` — schema, features, target, split strategy.
2. Add source data as CSV or Parquet.
3. Add dataset-specific preparation where required.
4. **Register the dataset** — `svc.register_dataset_from_config(config_path, source_path)`. Registration runs a fixed pipeline:
   1. **Load + parse the YAML** → schema, feature specs, target spec.
   2. **Ingest the source** — read CSV/Parquet, coerce column types per the YAML, validate against the spec (fails on missing columns or type mismatches), persist the full dataset as a single Parquet.
   3. **Profile** — compute per-column stats (distributions, null counts, cardinality, time range, entity count) used by the UI Data tab and `anistroph_profile_dataset`.
   4. **Partition** — split into `train.parquet` / `evaluation.parquet` / `validate.parquet` (80/20/0 by default). Temporal datasets sort chronologically (oldest → train, newest → eval); non-temporal shuffle with a fixed seed. The two partitions never overlap.
   5. **Register metadata** — write a `DatasetMeta` record to `artifacts/dataset_registry.json` with paths to the parquets, feature/target specs, and a pointer back to the YAML.

   Registration does **not** train a model, apply feature transforms, or construct the target — those happen at training time. After register, the dataset is ready for `train()`.

   See **[README_SETUP_USAGE.md](README_SETUP_USAGE.md#register-a-dataset)** for the Python call example, partition file table, and the full registration reference.
5. Train a model — model type auto-selected from target type if omitted.
6. Evaluate on the held-out partition.
7. Use the common inference, explanation, analysis, REST, MCP, and UI services.

The degree of dataset-specific work depends on the problem. The architecture isolates that work rather than assuming every domain has identical feature engineering.

### Dataset Configuration Skeleton

A `dataset.yaml` has three required blocks — `dataset:` (schema + identifiers + split), `features:` (model inputs and their transforms), `target:` (what to predict) — plus an optional `split:`:

```yaml
# datasets/<your_dataset>/dataset.yaml
dataset:
  dataset_id: <id>
  name: <human-readable name>
  entity_key: <unique-entity column>           # required
  time_key: <timestamp column>                 # optional — enables rolling windows + chronological splits
  columns:
    <col>: {type: numeric|categorical|boolean|timestamp, role: identifier|feature|target|event|metadata}

features:                                       # the inference contract — every entry becomes a model input
  <feature>:
    column: <source column>
    transforms: [current|categorical]          # non-temporal: current + categorical only
    # transforms: [current, mean: {windows: [1h, 6h]}, std: {windows: [1h, 6h]}]   # temporal: rolling ops available

target:
  name: <target name>
  type: regression|classification|binary|future_event
  source_column: <label column>
  # horizon: 24h, positive_class: 1            # future_event only

split:
  strategy: chronological|random
  train: 0.80
  eval: 0.20
```

For the full YAML reference — column types/roles, the complete transform table, target semantics, split configuration, and worked examples (non-temporal regression, temporal classification with rolling windows, multi-target pattern) — see **[README_SETUP_USAGE.md](README_SETUP_USAGE.md)**.

## Adding a Model

1. Add a model adapter under `backend/models/` implementing the predictor contract.
2. Declare its `model_type` and task type.
3. Register it with `MODEL_FACTORIES`.
4. Add the corresponding load behavior to the inference layer.
5. Implement model-specific explanation behavior when supported.

The model then participates in the common training, persistence, inference, and runtime architecture.

---

## Tests

```bash
pytest
```

The current suite contains **147 tests** spanning dataset specifications, ingestion, feature transforms and leakage checks, target construction, model training/evaluation/persistence/reload, inference, feature parity, SHAP explainability, multidimensional discovery, REST, MCP, and end-to-end workflows.

---

## Documentation

- **[README_SETUP_USAGE.md](README_SETUP_USAGE.md)** — setup, usage, training, prediction, and MCP examples
- **[README_TEST.md](README_TEST.md)** — testing and MCP testing guide
- **[RELEASE_NOTES.md](RELEASE_NOTES.md)** — release notes
- **[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)** — deeper architecture details

## License

MIT
