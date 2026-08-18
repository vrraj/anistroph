---
description: A multi-domain predictive analytics architecture for
  prediction, explainability, evaluation, multidimensional analysis,
  temporal forecasting, and AI agent access.
layout: default
title: "Anistroph: Multi-Domain Predictive Analytics and AI Agent
  Access"
---

# Anistroph

**Anistroph is a multi-domain predictive analytics architecture for
structured data --- connecting dataset-specific models to shared
prediction, explainability, evaluation, multidimensional analysis, and
AI-agent services.**

Different domains can have different schemas, features, targets,
preprocessing, and models without rebuilding the surrounding analytics
stack.

A semiconductor manufacturing team can predict wafer yield at different
process stages. A maintenance team can predict equipment failure and
remaining useful life. A procurement team can forecast material demand
and shortage risk. Each problem is different, but all use the same
Anistroph runtime.

> **Explore:** [GitHub](https://github.com/vrraj/anistroph) · [Setup &
> Usage](setup-usage) · [Technical Architecture](technical-architecture)

![Anistroph predictive analytics
pipeline](https://raw.githubusercontent.com/vrraj/anistroph/main/docs/images/anistroph-pipeline.png)

## Predictive Use Cases

Anistroph includes synthetic reference implementations designed to
exercise different parts of the architecture.

| Reference domain | Predictive use cases | What it demonstrates |
|---|---|---|
| **Semiconductor Manufacturing** | Wafer yield, critical dimension, film thickness | Multiple targets, process-stage prediction, SHAP explainability, multidimensional evaluation |
| **Predictive Maintenance** | Equipment failure, remaining useful life, maintenance required | Temporal sensor data, classification + regression, history-based features |
| **Semiconductor Materials Procurement** | 4-week material demand, 4-week shortage risk | Rolling forecasts, temporal prediction, inventory/supplier signals, multidimensional analysis |
| Real estate | Home price prediction | Lightweight cross-domain regression validation |

The reference datasets are not intended as production benchmarks. They
provide concrete, reproducible problems for demonstrating how the same
architecture behaves across different predictive domains.

## What Anistroph Does

### Predict

Run persisted models against existing entities or new source-feature
records. The runtime applies the same feature metadata and preprocessing
used during training.

### Explain

Explain individual predictions using model-specific explainability.
XGBoost models use TreeSHAP, with transformed categorical features
normalized back to understandable source fields and values.

### Evaluate

Evaluate persisted models against held-out data using task-appropriate
metrics. Evaluation can also be scoped to specific populations rather
than relying only on aggregate model performance.

### Discover

Analyze the underlying data across one, two, or three dimensions to find
populations where outcomes differ materially from the overall baseline.

### Forecast Over Time

For temporal problems, Anistroph reconstructs rolling and
history-dependent model inputs as of a prediction point. New
observations can produce updated forecasts without requiring the model
to be retrained for every forecast period.

### Work Through AI Agents

Claude and other MCP-compatible agents can discover datasets and models,
inspect required inputs, generate or select source-feature records,
predict, explain, evaluate, and analyze through domain-agnostic MCP
tools.

## AI Agents as an Interface to Predictive Analytics

Anistroph exposes predictive and analytical capabilities to Claude and
other AI agents through **MCP stdio and Streamable HTTP**.

``` text
User
 │
 ▼
Claude / AI Agent
 │
 ├── Discover datasets and models
 ├── Inspect model input requirements
 ├── Predict and explain
 ├── Evaluate model performance
 └── Analyze multidimensional populations
 │
 ▼
Anistroph Shared Runtime
 │
 ├── Registered Datasets
 ├── Feature / Preprocessing Contracts
 └── Persisted Models
```

Agents operate against the same runtime used by REST/OpenAPI and the Web
UI rather than implementing their own prediction or analytical logic.

### AI Agent Analysis & Validation

> Claude and AI agents can orchestrate Anistroph's prediction and
> analytical capabilities through MCP. Predictions, explanations,
> evaluations, and analyses are executed by Anistroph's shared services
> and can be independently reproduced through the Web UI or REST API
> when validation is required.

This makes the source-feature inputs, selected model, and analytical
operation portable across interfaces. An agent-driven workflow can
therefore be inspected or reproduced independently without changing the
underlying model execution path.

## One Architecture, Different Prediction Problems

### Multi-Domain Datasets

Each dataset retains its own schema, features, targets, preprocessing,
and model artifacts. Shared services remain independent of
semiconductor, maintenance, procurement, real estate, or future domain
concepts.

``` text
Semiconductor ──→ Yield / CD / Film Models ──────┐
Maintenance ────→ Failure / RUL Models ──────────┤
Procurement ────→ Demand / Shortage Models ──────┼─→ Shared Runtime
Real Estate ────→ Price Model ───────────────────┤
Future Domains ─→ Domain-Specific Models ────────┘
                                                   │
                                  Predict • Explain • Evaluate • Analyze
                                                   │
                                      MCP • REST/OpenAPI • Web UI
```

### Declarative Dataset and Model Contracts

Dataset YAML defines what is unique to a prediction problem:

-   source schema and identifiers;
-   model input features and transforms;
-   prediction target and task type;
-   temporal semantics where applicable;
-   train/evaluation split strategy.

The shared services consume these contracts rather than embedding
domain-specific rules into the runtime. To add your own dataset, author
a `dataset.yaml` and register it — see [Adding a
Dataset](https://github.com/vrraj/anistroph#adding-a-dataset) in the
README and the [Dataset Configuration
reference](setup-usage#configure-your-own-dataset) in the Setup & Usage
Guide for the full YAML schema, transform table, and worked examples.

### Multiple Targets

One source dataset can support independent predictive outcomes.
Semiconductor manufacturing, for example, uses separate models for wafer
yield, critical dimension, and film thickness while sharing the same
underlying source data.

### Process-Stage Prediction

Anistroph can train separate models for different points in a process
using only the information available at each stage.

``` text
Before Etch ──→ After Etch ──→ Deposition / Lithography ──→ Final Test
     │               │                         │
     ▼               ▼                         ▼
 Early Model     Mid-Process Model        Later-Stage Model
```

This supports progressively better-informed predictions without
introducing future-process information into earlier-stage models.

### Temporal Prediction

Temporal prediction separates the trained model from the changing
history used to construct its current inputs.

``` text
Entity History through as_of
            │
            ▼
   Rolling / Current Features
            │
            ▼
      Persisted Model
            │
            ▼
       Future Outcome
```

The longest configured rolling window determines how much entity history
is needed. The forecast target determines what future period is being
predicted.

### Source-Level Explainability

Model preprocessing can expand one business/source feature into several
engineered model features. Anistroph normalizes explanation output back
to the original source feature so users and agents do not need to reason
about one-hot encoded columns.

### Multidimensional Analysis and Evaluation

Anistroph distinguishes between two related questions:

-   **Where is the outcome unusual?** --- multidimensional analysis of
    observed data.
-   **Where is the model unusually good or bad?** --- multidimensional
    evaluation of held-out prediction error.

For example, a wafer-yield model can be evaluated overall and then
across `Product`, `Product × Tool`, and `Product × Tool × Chamber`
populations.

## How the System Fits Together

Each dataset follows a common predictive lifecycle:

**Dataset → define prediction targets → select and transform source
features → prepare training data → train models → evaluate persisted
models**

Once trained, models enter the shared runtime:

``` text
Dataset-Specific Layer
Schema • Features • Target • Preprocessing • Model
                         │
                         ▼
                  SHARED RUNTIME
          Predict • Explain • Evaluate • Analyze
                         │
          ┌──────────────┼───────────────┐
          ▼              ▼               ▼
         MCP        REST / OpenAPI      Web UI
          │
   Claude / AI Agents
```

The separation allows new datasets and models to be introduced without
rebuilding the runtime services or agent interfaces.

## For Developers

Anistroph is implemented as a modular Python architecture with thin
interfaces over a shared service layer.

| Layer | Technology | Role |
|---|---|---|
| Language | **Python** | Core services, data preparation, ML orchestration |
| API / Service | **FastAPI + Uvicorn** | REST/OpenAPI and Web UI service layer |
| Data processing | **Polars + DuckDB** | Columnar transformations, querying, analytical slicing |
| Persistence | **Parquet** | Dataset and partition storage |
| Configuration | **YAML** | Dataset schemas, features, targets, and split strategy |
| ML | **XGBoost + scikit-learn** | Regression and classification models |
| Explainability | **SHAP TreeExplainer** | Per-prediction XGBoost explanations |
| Model artifacts | **joblib** | Model persistence and reload |
| Agent access | **MCP SDK** | Domain-agnostic tools over stdio and Streamable HTTP |
| Testing | **pytest** | Unit, integration, MCP, and end-to-end coverage |

No database, message queue, or vector store is required by the current
reference implementation.

### Runtime Interfaces

-   **MCP** --- 13 domain-agnostic tools for dataset/model discovery,
    prediction, explanation, evaluation, and analysis.
-   **REST / OpenAPI** --- programmatic access to runtime and
    administrative capabilities.
-   **Web UI** --- exploration, model interaction, and cross-interface
    validation.
-   **Python services** --- direct access to the shared application
    layer.

Training and dataset administration are intentionally excluded from the
MCP agent tool surface.

## Extending Anistroph

The architecture is designed to extend along several dimensions:

-   add structured datasets from new domains;
-   define additional prediction targets against an existing source
    dataset;
-   add model adapters and task types;
-   introduce new model-specific explainability methods;
-   add additional process-stage models;
-   extend agent and application interfaces while retaining the shared
    runtime.

The current implementation supports regression and binary
classification. Multiclass classification, specialized forecasting
models, anomaly detection, model versioning/promotion, and
monitoring/drift detection are natural extension paths.

## Explore the Project

-   **[GitHub Repository](https://github.com/vrraj/anistroph)** ---
    source, releases, tests, and full README
-   **[Guided Walkthrough](walkthrough)** --- step-by-step inference
    lifecycle across three datasets (regression + classification,
    temporal + non-temporal)
-   **[Setup & Usage Guide](setup-usage)** --- Claude/MCP usage, dataset
    configuration, temporal prediction, operations, examples, and API
    reference
-   **[Technical Architecture](technical-architecture)** --- deeper
    implementation and architecture details
-   **[Full README](https://github.com/vrraj/anistroph#readme)** ---
    install, features, temporal prediction, extending, tests
-   **[Release
    Notes](https://github.com/vrraj/anistroph/blob/main/RELEASE_NOTES.md)**
    --- version history

## License

MIT
