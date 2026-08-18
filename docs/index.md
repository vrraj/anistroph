---
layout: default
title: "Anistroph: Predictive Analytics, Explainability, and Multidimensional Discovery"
description: "A domain-agnostic reference architecture for predictive analytics, model explainability, and multidimensional discovery across structured datasets."
---

# Anistroph

Anistroph is a **multi-domain predictive analytics architecture** in which datasets from different domains share common prediction, explanation, evaluation, and multidimensional analysis services while keeping their own schemas, features, targets, preprocessing, and models.

> **Source + releases:** [GitHub repo](https://github.com/vrraj/anistroph) · [Setup & Usage Guide](setup-usage) · [Technical Architecture](technical-architecture)

Anistroph provides a common predictive lifecycle:

Dataset Feature Specification  ⮕ train  ⮕ predict  ⮕ explain  ⮕ evaluate  ⮕ discover patterns , while allowing each dataset to retain its own schema, features, targets, preprocessing, and models. 
> The same runtime serves Claude Desktop, MCP-compatible agents, REST clients, and a Web UI through one **shared service layer**.

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
       Persist Model              Held-out Metrics
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

## What Anistroph does

- **Predict** — what outcome does a model forecast for a given record or entity?
- **Explain** — which source features drove an individual prediction (SHAP, normalized back to original inputs)?
- **Evaluate** — how well does a model perform on held-out data, overall and across multidimensional populations?
- **Discover** — where in the data are outcomes unusually high or low, and where does the model perform unusually well or poorly?
- **Temporal prediction & rolling forecasts** — for temporal datasets, Anistroph dynamically reconstructs rolling features from entity history at prediction time. The required history window is derived from the model's feature configuration (not specified by the caller), and only the bounded entity history is scanned — not the full dataset. The model stays fixed between retraining cycles; only the feature values change as new observations arrive. A procurement demand model, for example, can produce a new 4-week forecast each week without retraining.
- **Agent access** — Claude Desktop, Claude CLI, and other MCP-compatible agents can discover models, inspect required inputs, generate test records, predict, explain, and evaluate — all through 13 domain-agnostic MCP tools.

## Why it exists

Most predictive analytics projects rebuild the same plumbing for each new dataset: ingestion, feature engineering, train/test splitting, model persistence, inference, explanation, evaluation, and agent access. Anistroph isolates that plumbing behind common contracts so each dataset only defines what is unique to it — its schema, features, and target.

The result is a reference architecture where adding a new domain (semiconductor, maintenance, real estate, or something else) means authoring one YAML file and calling register — not rewriting the pipeline.

## Tech stack

| Layer | Technology | Role |
|---|---|---|
| Language | **Python** | Core services, data preparation, ML orchestration |
| API / Service | **FastAPI + Uvicorn** | REST API and Web UI service layer |
| Data processing | **Polars + DuckDB** | Columnar transformations, analytical slicing |
| Data persistence | **Parquet** | Efficient dataset storage |
| Dataset configuration | **YAML** | Declarative schemas, features, targets, split strategy |
| ML | **scikit-learn** | Logistic Regression, Linear/Ridge Regression, metrics |
| Gradient boosting | **XGBoost** | Classification and regression models |
| Explainability | **SHAP TreeExplainer** | Per-prediction feature contributions, normalized to source features |
| Model artifacts | **joblib** | Model persistence and reload |
| Agent integration | **MCP SDK** | Domain-agnostic runtime tools over stdio and Streamable HTTP |
| Testing | **pytest** | Unit, integration, MCP, and end-to-end tests (147 tests) |

No database, message queue, or vector store is required.

## Quick start

```bash
git clone https://github.com/vrraj/anistroph
cd anistroph
make install
```

`make install` creates a virtualenv, installs the package, generates and registers all reference datasets, and prints a ready-to-paste Claude Desktop MCP config with absolute paths filled in.

Then paste the printed config into Claude Desktop and ask:

> "List all Anistroph datasets"
> "Predict wafer yield for WAFER_015000 and explain what pushed it up or down"

## Code examples

### Train a model

```bash
python scripts/train_model.py --dataset semiconductor_yield \
  --model-type xgboost_regressor --model-id my-wafer-yield-model
```

### Predict (entity lookup)

```python
from backend.services import get_services

svc = get_services()
pred = svc.predict(
    model_id="my-wafer-yield-model",
    entity_id="WAFER_015000",
)
print(pred)
```

### Predict (records — new or hypothetical rows)

```python
pred = svc.predict(
    model_id="my-wafer-yield-model",
    records=[{
        "product_id": "PROD_B",
        "etch_tool": "ETCH_02",
        "etch_chamber": "CH_B",
        "etch_temperature_std": 1.8,
        # ... all source columns listed in the model's features block
    }],
)
```

### Predict (temporal — rolling forecast)

```python
# For temporal models, timestamp is the "as of" date — the last known point
# in history. Anistroph loads entity history through that point, computes
# rolling features, and applies the fixed trained model.
pred = svc.predict(
    model_id="semiconductor_procurement_demand-xgboost_regressor-...",
    entity_id="FAB_A__MAT_0001",
    timestamp="2025-06-09",   # as_of: predict using history through this week
)
# Returns predicted material demand for the next 4 weeks (June 16 → July 7)
# No retraining needed — the model is static, only the feature values change
```

### Explain a prediction

```python
expl = svc.explain(
    model_id="my-wafer-yield-model",
    entity_id="WAFER_015000",
    top_k=10,
)
# Returns SHAP contributions normalized back to source features
# (one-hot groups are summed back to the original categorical column)
```

### Evaluate on held-out data

```python
result = svc.evaluate_model(
    model_id="my-wafer-yield-model",
    sample_size=1000,
)
# Returns MAE, RMSE, R², MSE, MAPE, max_error
```

### Discover multidimensional patterns

```python
interesting = svc.find_interesting_slices(
    "semiconductor_yield",
    "wafer_yield",
    top_k=20,
)
# Finds tool/chamber/recipe combinations where yield deviates most from baseline
```

### MCP via Claude Desktop

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

13 MCP tools are available — `anistroph_list_datasets`, `anistroph_predict`, `anistroph_explain_prediction`, `anistroph_evaluate_model`, `anistroph_find_interesting_slices`, `anistroph_get_model_inputs`, and more.

### REST API

```bash
# Start the server
make start-native

# Predict via REST
curl -X POST http://localhost:9500/predictions \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my-wafer-yield-model", "entity_id": "WAFER_015000"}'

# OpenAPI docs
open http://localhost:9500/docs
```

## Reference datasets

Anistroph ships with synthetic reference datasets across four domains exercising regression, classification, temporal forecasting, and multidimensional patterns:

| Dataset | Domain | Rows | Targets |
|---|---|---|---|
| **Semiconductor Manufacturing** | Wafer fab process | 50,000 | Wafer yield, critical dimension, film thickness (regression) |
| **Predictive Maintenance** | Tool health monitoring | ~864,000 | Failure within 24h (classification), remaining useful life (regression), maintenance required (classification) |
| **Bay Area Home Prices** | Real estate | 40,000 | Sale price (regression) |
| **Semiconductor Materials Procurement** | Supply chain & materials planning | ~100,000 | Material demand next 4 weeks (regression), shortage risk next 4 weeks (classification) |

Each dataset is declared through a `dataset.yaml` and registered with one call. You can add your own dataset by authoring a YAML and registering it — see [Adding a Dataset](https://github.com/vrraj/anistroph#adding-a-dataset) in the README, and the [Dataset Configuration reference](setup-usage#dataset-configuration) in the Setup & Usage Guide for the full YAML schema, transform table, and worked examples.

## Architecture in brief

Anistroph separates **dataset-specific modeling** from a **shared predictive runtime**:

```text
Semiconductor ──→ Yield / CD / Film Models ──────┐
Maintenance ────→ Failure / RUL Models ──────────┤
Home Prices ────→ Price Model ───────────────────┼─→ Shared Runtime
Procurement ────→ Demand / Shortage Models ──────┤
Future Domains ─→ Domain-Specific Models ────────┘
                                                   │
                                      Predict • Explain • Evaluate
                                                   │
                                          MCP • REST • UI
```

- **Dataset-specific:** schema, features, targets, preprocessing, model artifacts — declared in YAML, isolated per dataset.
- **Shared:** training/evaluation, persistence, inference, explainability, multidimensional analysis, and all interfaces.
- **Extensible:** additional targets, model families, process-stage predictions, domains, and agent integrations.

Feature transforms (rolling windows, slopes, deltas) are leakage-safe — they use only observations up to and including the current time. Datasets are partitioned at registration (80/20 train/eval); temporal datasets split chronologically.

## Documentation

- **[Setup & Usage Guide](setup-usage)** — dataset configuration, temporal prediction & retraining, operations, MCP setup, API reference
- **[Technical Architecture](technical-architecture)** — deeper architecture details
- **[Full README](https://github.com/vrraj/anistroph#readme)** — install, features, temporal prediction, extending, tests
- **[Release Notes](https://github.com/vrraj/anistroph/blob/main/RELEASE_NOTES.md)** — version history

## Links

- [GitHub Repository](https://github.com/vrraj/anistroph)
- [Full README](https://github.com/vrraj/anistroph#readme)

## License

MIT
