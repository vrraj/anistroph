# Anistroph Release Notes

## v1.0.0 — Production-Ready Predictive Analytics Framework

Anistroph v1.0.0 is the first stable release. It takes the v0.1 reference
architecture and makes it production-ready with dual inference modes, staged
prediction, SHAP explanation normalization, model input schema discovery,
UI-based verification, expanded MCP tooling, and comprehensive documentation.

### What's new since v0.1

**Inference**

- **Dual prediction modes** — Predict by entity lookup (`entity_id` +
  optional `timestamp`; Anistroph loads the row from the dataset) or by
  records (raw source feature values as JSON for a new or hypothetical
  row). The caller never constructs engineered features — no one-hot
  vectors, no rolling aggregates. Anistroph applies the same FeatureEngine
  and persisted metadata as training. Available via REST, MCP, and Web UI.
- **Model input schema discovery** — `anistroph_get_model_inputs` returns
  the required source columns, their types, transforms, and the supported
  prediction mode (`entity_lookup` vs `entity_lookup_or_records`). Use it
  before predicting to discover what inputs a model expects. Available via
  REST (`GET /models/{model_id}/inputs`), MCP, and Web UI.
- **Records-based inference fix** — Datasets with a `time_key` for
  chronological splitting no longer fail when records-based prediction
  omits `entity_key`/`time_key` columns. The FeatureEngine adds
  placeholder columns when they're absent.

**Staged prediction**

- **4 semiconductor yield stages** — Stage A (Before Etch, 7 features),
  Stage B (After Etch, 17 features), Stage C (After Deposition, 26
  features), Stage D (Before Test, 27 features). All share the same source
  parquet and target (`wafer_yield`) with progressively larger feature
  sets. R² progression: -0.001 → 0.819 → 0.824 → 0.825. The dominant
  predictive signal is in etch actuals (Stage A → B jump).

**Explainability**

- **SHAP explanation normalization** — When one-hot encoding expands a
  source feature into multiple model features (`{source}__{category}`),
  the explanation layer aggregates SHAP contributions back to the original
  source feature. Returns `{feature, value, impact}` in human-readable
  form (e.g. `etch_tool = ETCH_02, impact = +0.0024`) rather than separate
  one-hot entries. Raw per-category SHAP values retained in a `detail`
  field for debugging.
- **Records-based explain** — Fixed the same `entity_key`/`time_key` bug
  in `explain_prediction` as in `predict`. SHAP explanation now works with
  records-based input on all dataset types.

**MCP**

- **13 MCP tools** (was 9 in v0.1) — added `anistroph_get_model_inputs`
  for input schema discovery.
- **Dual MCP transport** — stdio (for local clients like Claude Desktop,
  Cursor, Cline) and Streamable HTTP at `/mcp` (for remote clients,
  custom agents, and tool routers like Axiolex). Both expose the same 13
  tools and call the same service layer.
- **Compact `anistroph_list_models`** — returns only model_id, model_type,
  dataset_id, target_name, target_type, and created_at. Full metrics
  remain available via `anistroph_get_model_metrics`.
- **PR curve downsampling** — Classification PR curves downsampled to at
  most 200 points (was ~105K) to keep MCP responses under 1 MB.

**Web UI**

- **Records-based prediction** — Prediction tab (now "Predict & Explain")
  supports a mode dropdown: entity lookup or records. "Load Input Schema"
  button fetches required columns and pre-fills a JSON template.
- **Tooltips** — Added tooltips on prediction mode, records textarea, and
  schema loading explaining what to send and what not to send.

**Datasets**

- **11 registered datasets** (was 2 in v0.1):
  - Home prices (40K rows, regression)
  - Predictive maintenance (864K rows, classification)
  - Predictive maintenance RUL (864K rows, regression)
  - Predictive maintenance maintenance (864K rows, classification)
  - Semiconductor yield (50K rows, regression)
  - Semiconductor yield Stage A/B/C/D (50K rows each, regression)
  - Semiconductor CD (50K rows, regression)
  - Semiconductor film thickness (50K rows, regression)
- **11 trained models** across all datasets.

**Documentation**

- `README.md` — updated with Inference feature section, SHAP
  normalization details, 13 MCP tools table, staged prediction, and
  records-based prediction.
- `docs/setup-usage.md` — comprehensive usage guide with Python/REST/
  MCP/UI examples for every operation, including:
  - §2a: How to author `dataset.yaml` (columns vs features, transforms)
  - §8: Dual prediction modes with verification instructions
  - §9: SHAP explanation normalization with naming convention rules
  - §18: Staged prediction architecture
  - Model input schema discovery (4 interfaces)
  - REST API endpoint table (19 endpoints)

**Quality**

- **147 tests passing** — unit, integration, and end-to-end tests covering
  datasets, features, training, inference, explanation, MCP, REST API,
  and SHAP grouping.

### Migration from v0.1

- Version string updated from `0.1.0` to `1.0.0` in `pyproject.toml`,
  `backend/main.py`, and health check endpoint.
- No breaking changes to REST API or MCP tool signatures. Existing tools
  and endpoints work unchanged.
- `anistroph_list_models` response format changed (compact summary instead
  of full model dump). Use `anistroph_get_model_metrics` for full metrics.

---

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
