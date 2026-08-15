# Anistroph Release Notes

## v0.1 — Multi-Dataset Predictive Analysis Reference Architecture

Anistroph v0.1 demonstrates a common predictive and analytical architecture across two isolated reference datasets.

The goal of this release is to validate the architecture and framework across different data and prediction problems, rather than build a domain-specific application.

## Reference Datasets

### Tool Predictive Maintenance

Equipment and sensor data used to model equipment behavior and predict maintenance or failure risk.

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
Failure / maintenance risk
```

This dataset provides a reference implementation for predictive analysis using equipment and sensor data.

### Semiconductor Wafer Yield

Synthetic wafer manufacturing data representing process history across tools, chambers, recipes, and operating conditions.

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

This dataset extends the architecture to a different prediction target and adds multidimensional yield analysis.

The semiconductor reference dataset contains approximately 30,000 synthetic wafer records.

## Common Architecture

The datasets remain isolated while using common Anistroph services.

```text
┌──────────────────────┐     ┌──────────────────────┐
│ Tool / Maintenance   │     │ Semiconductor Yield  │
│                      │     │                      │
│ Sensor Data          │     │ Wafer Process Data   │
│ Failure Target       │     │ Yield Target         │
└──────────┬───────────┘     └──────────┬───────────┘
           │                            │
           └────────────┬───────────────┘
                        ↓
                Anistroph Framework
                        │
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
      Training       Inference      Analysis
          │             │             │
          └─────────────┼─────────────┘
                        ↓
                   Explainability
                        ↓
                     MCP stdio
                        ↓
                       Claude
```

Dataset-specific ingestion and feature preparation are supported where required.

Common framework components are reused for training, evaluation, model persistence, inference, explainability, multidimensional analysis, and MCP access wherever the underlying operation is the same.

## Model Training and Evaluation

Model training is treated as an administrative/model-lifecycle operation rather than a runtime MCP capability.

For the semiconductor yield reference implementation, an XGBoost regression model estimates wafer yield from manufacturing and process features.

```text
Manufacturing conditions
          ↓
       XGBoost
          ↓
   Predicted Yield
        89.8%
```

A simple regression baseline is retained for comparison.

Model performance is evaluated using:

- MAE
- RMSE
- R²
- Comparison against the baseline model

The trained model, preprocessing metadata, feature identities, feature order, and evaluation metrics are persisted together.

Runtime inference loads the persisted model and does not retrain it.

## Prediction Explainability

Per-prediction explainability uses **SHAP TreeExplainer (TreeSHAP)** for the XGBoost model.

For an individual wafer, an explanation may look like:

```text
Baseline model prediction            96.2%

Etch chamber = CH_B                  -2.8 pts
Etch temperature variation           -1.9 pts
Etch recipe = ER_04                  -1.1 pts
Maintenance age                      -0.8 pts
Product = P3                         +0.2 pts
                                     ─────────
Predicted yield                      89.8%
```

Feature engineering, preprocessing, model persistence, and inference preserve stable, human-readable feature identities so model contributions can be mapped back to meaningful source conditions.

SHAP explains **why the model produced a particular prediction**. It does not establish that a feature physically caused the observed outcome.

## Multidimensional Analysis

Anistroph separately analyzes patterns in the underlying dataset without relying on the predictive model.

For example:

```text
Overall yield                              96.4%

ETCH_02                                    96.1%
CH_B                                       95.9%

ETCH_02 + CH_B                             92.7%

ETCH_02 + CH_B
+ high temperature variation               88.7%
```

This allows combinations of conditions to be examined even when individual dimensions appear relatively normal.

The initial analysis framework supports:

- Single-dimension slicing
- Two-dimensional slicing
- Three-dimensional slicing
- Baseline comparisons
- Minimum population thresholds
- Ranked discovery of unusual populations

The same analytical framework is intended to operate across registered datasets rather than contain semiconductor-specific analysis logic.

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

Model explanation and observed-data analysis provide different perspectives on the same problem.

Neither should be interpreted as proof of causality.

## MCP Runtime Access

Anistroph exposes runtime analysis and inference through MCP stdio for use by clients such as Claude.

Runtime capabilities include:

- Dataset discovery and summaries
- Model discovery and metrics
- Prediction
- SHAP-based prediction explanation
- Manual dimensional slicing
- Automated interesting-slice discovery

Model training remains an administrative operation.

MCP tools call the same underlying Anistroph services rather than implementing separate analytical or model logic.

## Dataset Isolation

Reference datasets and their model artifacts remain isolated.

Conceptually:

```text
data/
├── predictive_maintenance/
│   └── data.parquet
│
└── semiconductor_yield/
    └── data.parquet

artifacts/models/
├── maintenance-xgb-v001/
└── wafer-yield-xgb-v001/
```

Each dataset may have its own ingestion and feature-preparation logic while using common downstream services.

This provides the foundation for adding additional analytical domains without creating separate applications.

## Reference Implementation Scope

The datasets and models in this release are reference implementations intended to exercise and validate the Anistroph architecture.

The semiconductor dataset is synthetic. Relationships between process variables and yield are intentionally introduced so prediction, explainability, and multidimensional discovery can be tested.

Results should therefore be interpreted as demonstrations of system behavior rather than semiconductor manufacturing conclusions.

The current release exercises the framework across:

```text
Data
  ↓
Feature preparation
  ↓
Model training
  ↓
Evaluation
  ↓
Model persistence
  ↓
Inference
  ↓
Explainability
  ↓
Multidimensional discovery
  ↓
Investigation
```

Applying the architecture to production datasets would require domain-specific validation of the source data, feature engineering, statistical methodology, model performance, and resulting interpretations.

## Architectural Direction

The two reference datasets demonstrate the central Anistroph design principle:

> **Datasets remain isolated and may require domain-specific preparation, while prediction, analysis, explainability, and runtime access are provided through a common framework.**

The semiconductor yield implementation is therefore not intended to turn Anistroph into a semiconductor application. It is a second reference domain for testing whether the architecture can support different predictive and analytical problems without duplicating the application.
