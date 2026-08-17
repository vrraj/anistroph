# Release Notes

## Version 1.0.0 — Initial Public Release

### Overview

Anistroph is a multi-domain predictive analytics architecture in which datasets from different domains share common prediction, explanation, evaluation, and multidimensional analysis services while keeping their own schemas, features, targets, preprocessing, and models.

This is the first public release. The complete API surface is documented in [docs/setup-usage.md](https://vrraj.github.io/anistroph/setup-usage.html).

---

## Core Capabilities

### Datasets and Targets
- 11 registered datasets across 3 source domains (predictive maintenance, semiconductor yield, home prices)
- Multi-target architecture — one source parquet can train independent models for different outcomes
- Process-stage prediction — models predict at different points in a workflow using only features available at that stage (Stage A → D semiconductor yield example)
- Train/eval partitioning at registration time, leakage-safe feature transforms

### Inference
- Dual prediction modes — entity lookup (`entity_id` + optional `timestamp`) or records (raw source feature values as JSON). The caller never constructs engineered features.
- Model input schema discovery — `anistroph_get_model_inputs` returns the required source columns, types, transforms, and supported prediction mode
- Available via REST, MCP, and Web UI

### Explainability
- SHAP TreeExplainer for tree-based models
- One-hot normalization — contributions from `{source}__{category}` columns are aggregated back to the original source feature and returned as `{feature, value, impact}`
- Raw per-category SHAP retained in a `detail` field for debugging

### Evaluation
- Held-out evaluation against `evaluation.parquet` (regression: MAE, MSE, RMSE, R², MAPE, max_error; classification: ROC-AUC, PR-AUC, F1, log loss)
- Slice-level evaluation — apply categorical filters to compare metrics on a subset
- Error slice discovery — automatically searches 1/2/3-dimensional combinations for populations where prediction error deviates most from baseline

### Multidimensional Analysis
- Manual slicing on 1, 2, or 3 dimensions with baseline comparisons and minimum population thresholds
- Automated interesting-slice discovery ranked by deviation from baseline

### MCP Runtime Access
- 13 MCP tools across stdio (local clients: Claude Desktop, Cursor, Cline) and Streamable HTTP at `/mcp` (remote clients, custom agents, tool routers)
- Both transports call the same service layer as REST and UI

### Web UI
- Tabs for Datasets, Data, Analysis, Train, Predict & Explain, and Evaluation
- Dataset-driven dropdowns populated from dataset profiles
- Records-based prediction with "Load Input Schema" prefill
- Hash-based routing for deep links

---

## Documentation Structure

- **[README.md](https://github.com/vrraj/anistroph#readme)** — Quick start and high-level overview
- **[docs/setup-usage.md](https://vrraj.github.io/anistroph/setup-usage.html)** — Dataset YAML reference, operations, MCP setup, API reference
- **[docs/technical-architecture.md](https://vrraj.github.io/anistroph/technical-architecture.html)** — Deeper architecture details
- **[RELEASE_NOTES.md](https://github.com/vrraj/anistroph/blob/main/RELEASE_NOTES.md)** — Version history

---

## Public API Surface

Stable MCP tools (13):
- `anistroph_list_datasets` — list registered datasets
- `anistroph_get_dataset_profile` — dataset schema, columns, target
- `anistroph_sample_rows` — raw row inspection with filters, columns, sort, limit
- `anistroph_find_interesting_slices` — ranked unusual populations in the data
- `anistroph_list_models` — compact model summary
- `anistroph_get_model_metrics` — full training and validation metrics
- `anistroph_get_model_inputs` — prediction input schema for a model
- `anistroph_predict` — entity lookup or records-based prediction
- `anistroph_explain_prediction` — SHAP explanation with one-hot normalization
- `anistroph_slice_data` — manual 1/2/3-dimensional slicing
- `anistroph_compare_slices` — baseline comparison across slices
- `anistroph_evaluate_model` — held-out evaluation with optional slice filters
- `anistroph_find_evaluation_slices` — error slice discovery

Stable REST endpoints (19): dataset discovery, profiling, row sampling, slicing and comparison, model listing, metrics, input schema, prediction, explanation, training, evaluation, and error slice discovery. Full table in `docs/setup-usage.md`.

Stable Python entry points:
- `AnistrophServices.register_dataset_from_config(config_path)` — register a dataset from YAML
- `AnistrophServices.train_model(...)` — train and persist a model
- `AnistrophServices.predict(model_id, ...)` — entity lookup or records-based prediction
- `AnistrophServices.explain_prediction(model_id, ...)` — SHAP explanation
- `AnistrophServices.evaluate_on_eval_set(model_id, ...)` — held-out evaluation
- `AnistrophServices.find_evaluation_slices(model_id, ...)` — error slice discovery
- `AnistrophServices.slice_data(...)` / `compare_slices(...)` — multidimensional analysis
- `AnistrophServices.find_interesting_slices(...)` — automated slice discovery
- `AnistrophServices.sample_rows(...)` — raw row inspection

---

## Compatibility

- Python 3.10+
- XGBoost, scikit-learn, Polars, SHAP for modeling and explanation
- FastAPI, Uvicorn for REST and MCP Streamable HTTP
- MCP SDK for stdio and Streamable HTTP transports

---

## Notes

This release establishes the stable 1.x API contract for Anistroph. The datasets and models shipped are synthetic reference implementations intended to exercise the architecture, not domain-specific conclusions.

Backward compatibility will be maintained within the 1.x series.

---

## Version 0.1 — Multi-Dataset Reference Architecture

Initial reference architecture validating that two isolated datasets (predictive maintenance, semiconductor yield) could share common training, inference, explainability, multidimensional analysis, and MCP runtime services while keeping their own schemas, features, and targets.

- 2 reference datasets, 2 trained models
- 9 MCP tools (stdio only)
- XGBoost regression for wafer yield, XGBoost classification for maintenance failure
- SHAP TreeExplainer for per-prediction explanation
- Manual and automated multidimensional slicing
- Single prediction mode (entity lookup only)

Superseded by 1.0.0.
