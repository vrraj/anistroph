---
description: Technical architecture of Anistroph --- prediction
  contracts, feature and target processing, training, evaluation,
  runtime services, AI agent access, and extension points.
layout: default
title: Technical Architecture \| Anistroph
---

# Anistroph --- Technical Architecture

This document describes how Anistroph separates dataset-specific
prediction problems from shared services for **training, prediction,
explainability, evaluation, and multidimensional analysis**.

> **New to Anistroph?** See the [Anistroph project overview](https://vrraj.github.io/anistroph/) for the architecture, capabilities, and design goals.
>
> For installation, Claude/MCP setup, dataset configuration, and worked examples, see [Setup & Usage](setup-usage).

## Contents

-   [Architecture Overview](#1-architecture-overview)
-   [Architectural Principles](#2-architectural-principles)
-   [Dataset & Prediction Contracts](#3-dataset--prediction-contracts)
-   [Feature & Target Processing](#4-feature--target-processing)
-   [Training & Evaluation](#5-training--evaluation)
-   [Runtime Architecture](#6-runtime-architecture)
-   [Services & Interfaces](#7-services--interfaces)
-   [AI Agent Access & Cross-Interface
    Validation](#8-ai-agent-access--cross-interface-validation)
-   [Persistence & Registries](#9-persistence--registries)
-   [Extensibility](#10-extensibility)
-   [Implementation Map](#11-implementation-map)

## 1. Architecture Overview

Anistroph uses dataset-specific configuration to define a prediction
problem while keeping the surrounding analytical and runtime services
generic.

Each registered dataset provides its own schema, features, target
semantics, preprocessing, and model artifacts. Those dataset-specific
contracts feed the same training, evaluation, prediction, explanation,
and analytical services.

![Anistroph technical architecture](https://raw.githubusercontent.com/vrraj/anistroph/main/docs/images/anistroph-pipeline.png)

At runtime, the architecture exposes four complementary capabilities:

-   **Predict** --- run point-in-time or future predictions from
    source-level inputs.
-   **Explain** --- explain individual predictions, including SHAP-based
    explanations for XGBoost models.
-   **Evaluate** --- measure persisted-model performance on held-out
    data and across multidimensional populations.
-   **Analyze** --- slice, aggregate, compare, and discover patterns in
    observed data.

The same capabilities are available through MCP, REST/OpenAPI, and the
Web UI.

## 2. Architectural Principles

### Dataset and model isolation

Domain-specific concepts remain in dataset and prediction configuration
rather than being embedded in shared runtime services. Semiconductor,
maintenance, procurement, real-estate, and future datasets can therefore
use the same application architecture.

### Configuration-driven prediction problems

`DatasetSpec`, `FeatureSpec`, and `TargetSpec` define the schema, model
inputs, transforms, and prediction target. The shared pipeline
interprets these contracts rather than requiring a separate training or
inference implementation for every dataset.

### Shared training and inference semantics

The same feature engine is used during training and inference. Persisted
feature metadata preserves categorical encodings and feature order so
runtime prediction applies the same preprocessing contract used to train
the model.

### Leakage-safe temporal processing

Temporal features use only observations available at the prediction
point. History scans are bounded by the longest configured rolling
window, and temporal datasets are split chronologically.

### Shared service layer

`AnistrophServices` is the common application layer used by MCP, REST,
and the Web UI. Interfaces remain thin and do not implement their own
model or analytical logic.

### Runtime and administrative boundaries

MCP exposes model discovery, prediction, explanation, evaluation, and
analysis. Dataset registration, model training, deletion, and arbitrary
Python execution remain outside the agent-facing MCP tool surface.

## 3. Dataset & Prediction Contracts

A prediction problem is defined by three related specifications:

``` text
DatasetSpec
    +
FeatureSpec
    +
TargetSpec
    ↓
Prediction Contract
```

### 3.1 DatasetSpec

`backend/datasets/spec.py`

A Pydantic model describing the source dataset, including:

-   `dataset_id`
-   `entity_key`
-   optional `time_key`
-   column types: numeric, categorical, boolean, timestamp, string
-   column roles: identifier, feature, event, target, metadata, ignore

The dataset contract defines what the source data means without
prescribing a particular model.

### 3.2 FeatureSpec

`backend/features/spec.py`

Defines the source features and transformations used to construct model
inputs.

Supported transforms include:

-   `current`
-   `mean`
-   `min`
-   `max`
-   `std`
-   `median`
-   `slope`
-   `delta`
-   `categorical`
-   `hour_of_day`
-   `day_of_week`
-   `elapsed_time`

Windowed transforms accept duration strings such as `1h`, `6h`, or
`13w`.

`FeatureSpec.max_history_window()` derives the longest configured
history requirement. Runtime inference uses this value to limit temporal
history scans to the data actually needed to reconstruct current model
inputs.

### 3.3 TargetSpec

`backend/targets/spec.py`

Defines the outcome being predicted.

Supported target types are:

-   `regression`
-   `classification`
-   `binary` --- legacy alias for classification
-   `future_event` --- forward-looking boolean outcome with a time
    horizon

`future_event` targets are constructed independently per entity so an
event on one entity cannot label another.

## 4. Feature & Target Processing

### 4.1 Feature Engine

`backend/features/engine.py`

The feature engine interprets `FeatureSpec` for both training and
inference.

For categorical features, encodings are fitted during training and
persisted in `FeatureMetadata`. Runtime inference then applies the
identical categories and transformed feature order.

For temporal features, calculations are performed per entity and only
against observations available through the prediction point.

### 4.2 Target Engine

`backend/targets/engine.py`

The target engine constructs labels according to `TargetSpec`.
Regression and classification targets use configured source columns,
while `future_event` constructs forward-looking labels within each
entity's history.

### 4.3 Leakage Prevention

Anistroph keeps feature construction and target construction temporally
separated:

-   rolling windows are trailing and per entity;
-   features never use observations after the prediction point;
-   future-event targets may look forward, while features do not;
-   categorical categories are learned from training data;
-   temporal datasets use chronological train/evaluation splits.

This keeps the training representation aligned with what would actually
be available at runtime.

## 5. Training & Evaluation

Training is orchestrated by `backend/ml/training.py`.

``` text
Registered Dataset
        │
        ▼
Build Features + Target
        │
        ▼
Train / Evaluation Split
        │
        ▼
Fit Model
        │
        ▼
Held-Out Evaluation
        │
        ▼
Persist Model + Contracts + Metrics
```

### 5.1 Model Contract

`backend/ml/base.py` defines the `Predictor` abstraction:

-   `fit`
-   `predict`
-   `predict_proba`
-   `save`
-   `load`
-   `feature_importance`

Implemented model adapters include:

-   `backend/models/logistic.py` --- Logistic Regression with
    `StandardScaler`
-   `backend/models/xgboost.py` --- XGBoost classifier
-   `backend/models/xgboost_regressor.py` --- XGBoost regressor
-   `backend/models/linear_regression.py` --- Linear/Ridge regression
    baseline

### 5.2 Held-Out Evaluation

`backend/ml/evaluation.py`

Classification metrics include:

-   ROC-AUC
-   PR-AUC
-   precision
-   recall
-   F1
-   confusion matrix

Regression metrics include:

-   MAE
-   MSE
-   RMSE
-   R²
-   MAPE
-   max error

Classification thresholds can be optimized for F1 on validation data.

### 5.3 Multidimensional Evaluation

Aggregate model metrics can hide populations where performance differs
materially.

Anistroph can evaluate persisted-model error across one-, two-, and
three-dimensional categorical combinations, for example:

``` text
Overall Model
     │
     ├── Product
     ├── Product × Tool
     └── Product × Tool × Chamber
```

This answers a different question from multidimensional data analysis:

-   **Analysis:** where is the observed outcome unusual?
-   **Evaluation:** where is model performance unusually good or poor?

## 6. Runtime Architecture

Runtime inference is implemented in `backend/ml/inference.py`.

Prediction accepts a `model_id` and either an existing entity or
source-level records. Anistroph loads the persisted model contract,
reconstructs the required features, applies the stored preprocessing,
and invokes the model.

### 6.1 Temporal Inference

For temporal models, inference reconstructs features from entity history
using the same feature engine used during training.

Parquet is scanned lazily with predicate pushdown. The lower history
boundary is derived from the model's longest configured feature window,
avoiding a full-dataset load when only recent history is required.

### 6.2 Explainability

`backend/ml/explain.py`

XGBoost models use SHAP TreeExplainer. One-hot SHAP contributions are
grouped back to the original source feature so explanations remain
understandable at the dataset level.

Models without TreeSHAP support use an importance-weighted fallback.

### 6.3 Multidimensional Analysis

`backend/analysis/slice.py`

The analytical engine provides deterministic operations such as:

-   slice
-   aggregate
-   compare
-   multidimensional population discovery

These operations use Polars and remain independent of model training.

## 7. Services & Interfaces

### 7.1 AnistrophServices

`backend/services.py`

`AnistrophServices` is the **unified service layer powering all datasets
and interfaces**.

It coordinates dataset, model, prediction, explanation, evaluation, and
analytical operations behind a common application boundary.

``` text
Claude / AI Agents       Applications          Web UI
        │                     │                  │
       MCP                 REST API              │
        │                     │                  │
        └──────────────┬──────┴──────────────────┘
                       ▼
                AnistrophServices
                       │
        Dataset • Model • Prediction
       Explanation • Evaluation • Analysis
```

### 7.2 MCP

`backend/integrations/mcp/`

Two MCP transports call the same service layer:

-   **stdio** --- local subprocess transport used by Claude Desktop and
    other local MCP clients.
-   **Streamable HTTP** --- `/mcp` endpoint on the FastAPI server for
    remote MCP clients.

The MCP layer does not expose arbitrary Python execution or model
training.

### 7.3 REST / OpenAPI

`backend/main.py`, `backend/api/`

FastAPI routers expose datasets, analysis, models, predictions,
explanations, evaluation, and administrative operations. Routes delegate
to `AnistrophServices`.

### 7.4 Web UI

`frontend/index.html`

The Web UI provides dataset, analysis, training, model, and prediction
workspaces and adapts to registered dataset specifications.

## 8. AI Agent Access & Cross-Interface Validation

Claude and other AI agents can use MCP to discover datasets and models,
inspect model input requirements, run predictions, explain results,
evaluate model performance, and perform multidimensional analysis.

The agent is an orchestration and interaction layer; model execution and
analytical operations remain inside Anistroph.

Because MCP, REST, and the Web UI invoke the same `AnistrophServices`
layer, an agent-generated operation can be reproduced through another
interface using the same model and inputs.

``` text
Claude / AI Agent
       │
       ▼
Discover Model + Input Contract
       │
       ▼
Predict • Explain • Evaluate • Analyze
       │
       ▼
AnistrophServices
       │
       ▼
Persisted Model + Dataset
       │
       ├──────── MCP
       ├──────── REST / OpenAPI
       └──────── Web UI
```

This provides cross-interface validation without creating separate
execution paths for agent-driven analysis.

## 9. Persistence & Registries

### 9.1 Data Storage

Registered dataset data is persisted primarily as **Parquet**.

### 9.2 Dataset Registry

`artifacts/dataset_registry.json`

`DatasetRegistry` stores lightweight dataset metadata independently of
the Parquet files.

### 9.3 Model Artifacts

Persisted model artifacts are stored under:

``` text
artifacts/models/<model_id>/
    model.joblib
    imputer.joblib
    metadata.json
    feature_spec.json
    feature_metadata.json
    target_spec.json
    metrics.json
```

`artifacts/models/model_index.json` provides the model metadata index.

Together, the model artifact and persisted specifications preserve the
model, preprocessing contract, feature metadata, target definition, and
evaluation results required for runtime use.

## 10. Extensibility

Anistroph is designed so new prediction problems extend defined
architectural boundaries rather than requiring changes throughout the
system.

| Extension | Architecture point |
|---|---|
| New dataset or domain | `DatasetSpec` + `FeatureSpec` + `TargetSpec` |
| New target against existing data | New target/configuration |
| New target semantics | `TargetSpec` + target engine |
| New model family | `Predictor` model adapter |
| New explanation method | Explanation layer |
| New analytical operation | Analysis service |
| New REST/UI interface capability | `AnistrophServices` |
| New MCP capability | MCP tool over an existing service operation |

The current architecture supports regression and binary classification.
Additional task types and model families can be introduced behind the
same contracts.

## 11. Implementation Map

``` text
backend/
├── datasets/
│   ├── spec.py             DatasetSpec
│   └── ...                 Dataset registry and loading
├── features/
│   ├── spec.py             FeatureSpec
│   └── engine.py           Shared feature construction
├── targets/
│   ├── spec.py             TargetSpec
│   └── engine.py           Target construction
├── ml/
│   ├── base.py             Predictor contract
│   ├── training.py         Training pipeline
│   ├── evaluation.py       Model evaluation
│   ├── inference.py        Runtime prediction
│   ├── explain.py          Explanation layer
│   └── registry.py         Model artifact store
├── models/
│   ├── logistic.py
│   ├── xgboost.py
│   ├── xgboost_regressor.py
│   └── linear_regression.py
├── analysis/
│   └── slice.py            Multidimensional analysis
├── integrations/
│   └── mcp/                MCP transports and tools
├── api/                    REST routers
├── main.py                 FastAPI application
└── services.py             Shared application service layer

frontend/
└── index.html              Web UI

artifacts/
├── dataset_registry.json
└── models/
    ├── model_index.json
    └── <model_id>/         Persisted model artifacts
```

The implementation map is intentionally secondary to the architecture:
source-code modules implement the contracts and service boundaries
described above rather than defining the architecture themselves.
