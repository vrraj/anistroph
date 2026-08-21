# Release Notes

## Version 1.0.0 — Initial Public Release

### Overview

Anistroph is a multi-domain predictive analytics architecture in which datasets from different domains share common prediction, explanation, evaluation, and multidimensional analysis services while keeping their own schemas, features, targets, preprocessing, and models.

This is the first public release. The complete API surface is documented in [docs/setup-usage.md](https://vrraj.github.io/anistroph/setup-usage.html).

---

## Core Capabilities

### Datasets and Targets
- 16 registered datasets across 4 source domains (predictive maintenance, semiconductor yield, home prices, semiconductor memory)
- Multi-target architecture — one source parquet can train independent models for different outcomes
- Process-stage prediction — models predict at different points in a workflow using only features available at that stage (Stage A → D semiconductor yield example)
- Train/eval partitioning at registration time, leakage-safe feature transforms

### Parametric Search
- Generic structured search over registered datasets — operators: eq, in, gte, lte, between, contains_range, semantic
- Self-describing search contract — `anistroph_get_search_contract` returns searchable fields (types, units, operators, aliases, categorical values / numeric ranges) and semantic filters
- Semantic filter expansion — natural engineering concepts map to deterministic predicates (e.g. "supports 55°C" → `min <= 55 AND max >= 55`; "industrial temperature" → `min <= -40 AND max >= 95`)
- Applied-filters audit — responses include the normalized query after semantic expansion
- `sample_rows` refactored to delegate to the search engine internally (API unchanged)
- Semiconductor Memory reference dataset — 2000-row catalog with DDR5/LPDDR5X components and modules, family-specific parameter rules, synthetic supply fields

### Predict-on-Search
- Search a catalog, then predict for each matching product using a trained supply model
- `anistroph_predict_on_search` — runs parametric search, then entity-lookup prediction for each product_id, ranks by prediction outcome
- Semiconductor Memory Supply dataset — 50,000 rows (2,000 products × 25 weeks) of synthetic weekly supply history with inventory, demand, backlog, PO, lead time, OTD, and allocation status
- Two trained models: supply_risk_next_4w (classification, ROC-AUC = 0.999) and lead_time_next_4w_days (regression, R² = 0.996)
- Available via REST, MCP, and Web UI

### Inference
- Dual prediction modes — entity lookup (`entity_id` + optional `timestamp`) or records (raw source feature values as JSON). The caller never constructs engineered features.
- Model input schema discovery — `anistroph_get_model_inputs` returns the required source columns, types, transforms, and supported prediction mode
- Available via REST, MCP, and Web UI

### Explainability
- SHAP TreeExplainer for tree-based models
- One-hot normalization — contributions from `{source}__{category}` columns are aggregated back to the original source feature and returned as `{feature, value, impact}`
- Raw per-category SHAP retained in a `detail` field for debugging

### Evaluation
- Held-out evaluation against `evaluation.parquet` (regression: MAE, MSE, RMSE, R², MAPE, max error, median absolute error, 95th-percentile absolute error, mean prediction error, and baseline comparison; classification: ROC-AUC, PR-AUC, precision, recall, F1)
- Slice-level evaluation — apply categorical filters to compare metrics on a subset
- Error slice discovery — automatically searches 1/2/3-dimensional combinations for populations where prediction error deviates most from baseline

### Multidimensional Analysis
- Manual slicing on 1, 2, or 3 dimensions with baseline comparisons and minimum population thresholds
- Automated interesting-slice discovery ranked by deviation from baseline

### MCP Runtime Access
- 16 native MCP tools + 1 external A2A tool (17 total) across stdio (local clients: Claude Desktop, Cursor, Cline) and Streamable HTTP at `/mcp` (remote clients, custom agents, tool routers)
- Both transports call the same service layer as REST and UI
- External tools loaded from `integrations/tool_registry.yaml` and dispatched through a shared A2A JSON-RPC invoker

### External Integrations (A2A)
- External tool registry (`integrations/tool_registry.yaml`) — configuration source for externally hosted capabilities such as Aina-Veris
- Shared A2A invoker (`backend/integrations/a2a.py`) — JSON-RPC 2.0 `tasks/send` client used by both MCP and REST
- REST: `GET /integrations/tools`, `POST /integrations/tools/{tool_name}/invoke`
- MCP: external tools appear in `tools/list` and are callable via `tools/call`
- Environment variable substitution (`${AINA_VERIS_BASE_URL}`) for deployment-portable URLs
- Aina-Veris semiconductor research agent registered as external A2A capability

### Web UI
- Tabs for Datasets, Data, Search, Analysis, Train, Predict & Explain, and Evaluation
- Parametric Search tab with contract display, dynamic filter form (categorical multi-selects, numeric min/max, semantic temperature input), and applied-filters audit
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
- `anistroph_profile_dataset` — dataset schema, columns, target
- `anistroph_sample_rows` — raw row inspection with filters, columns, sort, limit
- `anistroph_find_interesting_slices` — ranked unusual populations in the data
- `anistroph_list_models` — compact model summary
- `anistroph_get_model_metrics` — full training and validation metrics
- `anistroph_get_model_inputs` — prediction input schema for a model
- `anistroph_predict` — entity lookup or records-based prediction
- `anistroph_explain_prediction` — SHAP explanation with one-hot normalization
- `anistroph_slice_data` — manual 1/2/3-dimensional slicing
- `anistroph_compare_data` — baseline comparison across slices
- `anistroph_evaluate_model` — held-out evaluation with optional slice filters
- `anistroph_find_evaluation_slices` — error slice discovery

Stable REST endpoints: dataset discovery, profiling, row sampling, slicing and comparison, model listing, metrics, input schema, prediction, explanation, training, evaluation, and error slice discovery. Full table in `docs/setup-usage.md`.

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
