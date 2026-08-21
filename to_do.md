# To-Do: Semiconductor Memory Parametric Search Extension

Reference spec: `product-specifications/ANISTROPH_SEMICONDUCTOR_MEMORY_SPEC.md`
Reference data: `product-specifications/sample-data/semiconductor_memory_2000.csv` (2000 rows, 35 columns)

## Decisions locked with user

- **Data source**: Use the supplied `semiconductor_memory_2000.csv` directly as the
  reference source dataset. Do NOT regenerate, replace, or independently
  synthesize the catalog. The CSV will be committed to the repo on GitHub for
  cross-clone reproducibility.
- **Search vs sample_rows**: The new `search_dataset` engine extends
  `sample_rows`. `sample_rows` keeps its simple equality/IN API but delegates
  internally to the new filter engine. `search_dataset` is the full-featured
  version (eq/in/gte/lte/between/contains_range + semantic filters + contract).
- **Scope**: Phase 1 (dataset + deterministic search) = Commit Point 1.
  Phase 2 (prediction on search results) = Commit Point 2.
  Phase 3 (Aina-Veris RAG) is out of Anistroph scope (separate MCP server);
  datasheet PDFs live in `product-specifications/sample-data/` for Aina-Veris
  ingestion, not Anistroph.

## Architecture summary

New `backend/search/` package (generic, not memory-specific):
- `spec.py` — `SearchFieldSpec`, `SemanticFilterSpec`, `SearchConfig` (loaded
  from a new `search:` YAML section in dataset.yaml).
- `filters.py` — `FilterExpression`, `SortExpression`, operator application
  via Polars, semantic-filter expansion.
- `service.py` — `search_dataset()` and `get_search_contract()`.

`sample_rows` in `backend/services.py` is refactored to convert its simple
equality/IN dict filters into `FilterExpression` objects and delegate to the
new engine. Its public API and return shape are unchanged.

New MCP tools: `anistroph_get_search_contract`, `anistroph_search`.
New REST endpoints: `GET /datasets/{id}/search-contract`,
`POST /datasets/{id}/search`.

Dataset configs (multi-target pattern, one source CSV shared):
- `datasets/semiconductor_memory/dataset.yaml` — catalog config (all columns,
  no target, for search + analysis). Phase 1.
- `datasets/semiconductor_memory_supply_risk/dataset.yaml` — classification
  target `supply_risk_next_4w`. Phase 2.
- `datasets/semiconductor_memory_lead_time/dataset.yaml` — regression target
  `lead_time_next_4w_days`. Phase 2.

---

## Commit Point 1 — Phase 1: Dataset + Deterministic Search ✅ COMPLETE

### 1.1 Ingest the reference CSV as a dataset source
- [x] Copy `product-specifications/sample-data/semiconductor_memory_2000.csv`
      to `data/semiconductor_memory/data.csv` (canonical source location).
- [x] Verify column types against the spec §3.3 / §3.4 (note: empty strings
      for `module_density_gb` on components and `component_density_gb` on
      modules → nulls; `ecc` is empty/FALSE/TRUE → treat as categorical with
      nulls; `supply_risk_next_4w` is 0/1 int; `allocation_status` includes
      literal string `None`).
- [x] Add `data/semiconductor_memory/data.csv` to `scripts/setup_datasets.py`
      DATASETS list (config + source). Do NOT add a generator script.

### 1.2 Author the catalog dataset.yaml
- [x] Create `datasets/semiconductor_memory/dataset.yaml`:
      - `dataset_id: semiconductor_memory`, `entity_key: product_id`,
        no `time_key` (non-temporal snapshot).
      - All 35 columns declared with type + role. Catalog/search fields =
        `feature` or `metadata`; supply fields = `feature`; the two targets
        = `target` (declared so they're carried, but no `target:` section so
        no model is trained on this config).
      - `split: random 0.80/0.20` (non-temporal).
      - New `search:` section (see 1.4).
- [x] Register the dataset and confirm 2000 rows, profile, partitions.

### 1.3 Build the search config model (`backend/search/spec.py`)
- [x] `SearchFieldSpec`: `field`, `operators` (list), `unit` (optional),
      `aliases` (list, optional), `description` (optional).
- [x] `SemanticFilterSpec`: `name`, `type` (`range_contains` | `expands_to`),
      `min_field`/`max_field` (for range_contains), `expands_to` (list of
      FilterExpression dicts, for expands_to), `unit`, `description`.
- [x] `SearchConfig`: `searchable_fields` (dict), `semantic_filters` (dict).
- [x] Loader: parse the `search:` YAML section into `SearchConfig`.

### 1.4 Build the filter engine (`backend/search/filters.py`)
- [x] `Operator` enum: `eq`, `in`, `gte`, `lte`, `between`, `contains_range`,
      `semantic`.
- [x] `FilterExpression` pydantic model: `field`, `op`, `value`,
      `min_field`/`max_field` (required for `contains_range`), `low`/`high`
      (required for `between`).
- [x] `SortExpression`: `field`, `descending`.
- [x] `apply_filter(df, expr) -> df` — Polars predicate per operator.
- [x] `apply_filters(df, filters) -> df` — AND-combine all filters.
- [x] `expand_semantic(filters, search_config) -> filters` — resolve semantic
      filter names into deterministic FilterExpression(s).
- [x] Validate filter fields against the dataset columns; unknown field →
      ValueError.

### 1.5 Build the search service (`backend/search/service.py`)
- [x] `search_dataset()` — loads full parquet, expands semantic filters,
      applies filters, sort, head(limit), returns applied_filters audit.
- [x] `get_search_contract()` — merges YAML config with live profile data.

### 1.6 Refactor `sample_rows` to delegate to the search engine
- [x] Convert equality/IN dict filters to FilterExpression, delegate to
      `search_dataset`. Public API and return shape unchanged.
- [x] `sample_rows` does NOT use semantic filters or the search contract.

### 1.7 Wire the search config into DatasetConfig
- [x] `load_dataset_config` parses the `search:` section into `SearchConfig`.
- [x] `search_config: Optional[SearchConfig]` added to `DatasetConfig`.

### 1.8 Add service methods
- [x] `AnistrophServices.search(dataset_id, filters, sort, limit, columns)`
- [x] `AnistrophServices.get_search_contract(dataset_id)`

### 1.9 Add REST endpoints (`backend/api/search.py`)
- [x] `GET /datasets/{dataset_id}/search-contract`
- [x] `POST /datasets/{dataset_id}/search` with `SearchRequest` schema.
- [x] `SearchRequest`, `FilterExpressionRequest`, `SortExpressionRequest`
      added to `backend/schemas/api.py`.

### 1.10 Add MCP tools (`backend/integrations/mcp/tools.py`)
- [x] `anistroph_get_search_contract(dataset_id)`
- [x] `anistroph_search(dataset_id, filters, sort, limit, columns)`
- [x] MCP tool count updated (13 → 15) in docs.

### 1.11 Add Web UI Search tab
- [x] New "Search" tab in `frontend/index.html` (between Data and Analysis).
- [x] Dynamic filter form from search contract (categorical multi-selects,
      numeric min fields, semantic temperature input).
- [x] Search/Reset buttons.
- [x] Results grid with applied-filters audit collapsible.
- [x] Search contract display (collapsible details).

### 1.12 Tests
- [x] `tests/unit/test_search.py` (28 tests): all operators, semantic
      expansion, unknown field/semantic errors, AND-combination, limit cap,
      sort, columns subset, applied_filters audit, contract enrichment.
- [x] `tests/integration/test_api.py` (10 new tests): search-contract GET,
      search POST (3 acceptance queries, sort, columns, errors).
- [x] `tests/integration/test_mcp.py` (5 new tests): tool discovery,
      get_search_contract, search (acceptance + semantic), error handling.
- [x] All 3 acceptance queries pass via REST and MCP.
- [x] All 188 tests pass (was 147; +41 new search tests).

### 1.13 Documentation
- [x] `README.md`: semiconductor_memory in reference datasets; parametric
      search in Core Features; MCP tools 13→15; tests 147→188; tool table
      updated with 2 new tools.
- [x] `docs/setup-usage.md`: semiconductor_memory example queries; test
      count updated.
- [x] `docs/index.md`: semiconductor_memory in reference table; parametric
      search in Core Features; MCP tools 13→15.
- [x] `docs/technical-architecture.md`: test count 147→188; test layout
      table updated with test_search.py and updated counts.
- [x] `RELEASE_NOTES.md`: parametric search section; dataset count 11→14;
      MCP tools 13→15; Web UI Search tab.

### 1.14 Commit Point 1 verification
- [x] `pytest` — all 188 tests pass.
- [x] Register `semiconductor_memory`, confirm 2000 rows.
- [x] Run the 3 acceptance queries via REST and MCP — all pass.
- [x] `sample_rows` regression — all 147 original tests still pass.
- [ ] `git commit` — "feat: semiconductor_memory dataset + generic
      parametric search (Phase 1)".

---

## Commit Point 2 — Phase 2: Prediction on Search Results

### 2.1 Author the two prediction dataset configs
- [ ] `datasets/semiconductor_memory_supply_risk/dataset.yaml`:
      - Same source CSV, `entity_key: product_id`, no `time_key`.
      - `target: supply_risk_next_4w` (classification, positive_class 1).
      - Features = the synthetic supply fields (inventory_units,
        weekly_demand_units, inventory_coverage_weeks, backlog_units,
        backlog_ratio, open_po_units, open_po_coverage,
        supplier_lead_time_days, supplier_otd_pct, demand_trend_4w_pct,
        allocation_status) + lifecycle (part_status) as categorical.
      - Catalog fields are NOT features (they don't drive supply risk per the
        spec — supply risk is synthetic from operational context).
      - `split: random 0.80/0.20`.
- [ ] `datasets/semiconductor_memory_lead_time/dataset.yaml`:
      - Same features, `target: lead_time_next_4w_days` (regression).
- [ ] Add both to `scripts/setup_datasets.py` DATASETS list (same source CSV).

### 2.2 Train + evaluate the two models
- [ ] Train `supply_risk_next_4w` (xgboost classification). Record metrics.
- [ ] Train `lead_time_next_4w_days` (xgboost_regressor). Record metrics.
- [ ] Evaluate both on the held-out 20% eval partition.
- [ ] Confirm `get_model_inputs` reports `entity_lookup_or_records` (no
      rolling-window transforms → records-based prediction works, which is
      what predict-on-search needs).

### 2.3 Add predict-on-search service
- [ ] `AnistrophServices.predict_on_search(dataset_id, model_id, filters,
      sort_by_prediction, limit, explain)`:
      - Run `search_dataset` to get matching product_ids + rows.
      - Build records from the matching rows' feature columns.
      - Call `predict` (records mode) for the batch.
      - Optionally call `explain` for each.
      - Rank by prediction (supply_risk descending = highest risk first;
        lead_time descending = longest lead time first, or configurable).
      - Return `{dataset_id, model_id, matched, returned, ranked: [{product_id,
        prediction, ...row, explanation?}], applied_filters}`.
- [ ] Reuse the existing `predict`/`explain` services — no new inference path.

### 2.4 Add REST endpoint
- [ ] `POST /datasets/{dataset_id}/predict-on-search` with
      `PredictOnSearchRequest` (`model_id`, `filters`, `sort_by_prediction`,
      `limit`, `explain`).
- [ ] Add schema to `backend/schemas/api.py`.

### 2.5 Add MCP tool
- [ ] `anistroph_predict_on_search(dataset_id, model_id, filters,
      sort_by_prediction, limit, explain)` — search + predict + rank in one
      call. Returns ranked candidates with predictions (and optional SHAP
      explanations).
- [ ] Update MCP tool count (15 → 16) in docs.

### 2.6 Web UI — supply-risk scoring in Search tab
- [ ] After search results render, add a "Rank by supply risk" / "Rank by
      lead time" button that calls predict-on-search and re-renders the
      grid with a prediction column + optional explain drawer.
- [ ] Model dropdown populated from registered models for
      semiconductor_memory datasets.

### 2.7 Tests
- [ ] `tests/integration/test_api.py`: predict-on-search endpoint for both
      models; verify ranking order; verify explain output.
- [ ] `tests/integration/test_mcp.py`: `anistroph_predict_on_search` tool.
- [ ] Acceptance query (spec §8 Phase 2):
      > Find matching DDR5 parts and rank them by predicted four-week supply
      > risk.
      → search (DDR5_COMPONENT) + predict_on_search (supply_risk model,
      sort_by_prediction descending).

### 2.8 Documentation
- [ ] `README.md`: add the two trained models to the reference models table;
      mention predict-on-search.
- [ ] `docs/setup-usage.md`: add Phase 2 example queries (search + predict +
      rank; search + explain).
- [ ] `docs/index.md`: update reference models table.
- [ ] `RELEASE_NOTES.md`: prediction-on-search section.

### 2.9 Commit Point 2 verification
- [ ] `pytest` — all tests pass.
- [ ] Both models trained + evaluated with metrics recorded.
- [ ] Acceptance query works via REST and MCP.
- [ ] Web UI rank-by-risk button works.
- [ ] `git commit` — "feat: prediction on search results + supply-risk /
      lead-time models (Phase 2)".

---

## Out of scope (tracked, not actioned now)

- **Phase 3 — Aina-Veris RAG**: datasheet PDFs are in
  `product-specifications/sample-data/` for Aina-Veris ingestion. Anistroph
  only hands off `product_id` / `datasheet_id` lists. No RAG code in
  Anistroph. Document the handoff contract in `docs/setup-usage.md` after
  Phase 2 if needed.
- **Phase 4 — Hardening**: pagination, richer semantic aliases, performance
  tests, saved searches, additional memory families, temporal supply history.
- **Generator script**: per user decision, the supplied CSV is canonical; no
  fixed-seed generator will be written unless the spec is revised.

---

## Risks

1. **Generic-search over-engineering** — Building a fully generic search
   service for ALL datasets before validating with one. *Mitigation*: build
   generic operators + contract, but only configure `search:` + semantic
   filters for `semiconductor_memory`. Other datasets keep working via
   `sample_rows` (which now delegates but needs no `search:` section).

2. **`sample_rows` refactor regression** — 147 existing tests depend on
   `sample_rows`'s exact return shape and equality/IN behavior. *Mitigation*:
   keep the public signature and return dict identical; convert dict filters
   to `FilterExpression` internally; run the full suite before Commit Point 1.

3. **`contains_range` boundary semantics** — "supports 55°C" must mean
   `min <= 55 AND max >= 55` (inclusive). Off-by-one or exclusive bounds
   would silently drop valid products. *Mitigation*: inclusive on both ends;
   dedicated unit test with a known product (e.g. a -40..95 product must
   match 55, -40, 95, but not 96).

4. **Null handling in filters** — `module_density_gb` is null for components,
   `component_density_gb` null for modules, `ecc` null for some rows. A
   `gte`/`lte`/`between` on a null column must drop nulls (Polars does this by
   default for comparisons). An `eq` on null is not expressible via the
   current operator set (no `is_null` op) — acceptable for Phase 1; document
   as a Phase 4 gap.

5. **Literal string `"None"` in `allocation_status`** — The CSV has the
   string `None` (not a true null) for some rows. Must be treated as a
   categorical value, not a null. *Mitigation*: declare `allocation_status`
   as `categorical`; Polars reads it as a string; `eq "None"` works.

6. **Search contract staleness** — If the contract caches categorical values
   from a profile, re-registration could make it stale. *Mitigation*:
   compute the contract on-demand (merge YAML `search:` config + live
   profile) on every `get_search_contract` call. No caching.

7. **Search over full dataset vs partition** — Search must run over ALL
   products (`meta.parquet_path`), not the train/eval partition, or search
   results won't include every catalog product. *Mitigation*: `search_dataset`
   reads `meta.parquet_path` explicitly (same as `sample_rows`); partition
   paths are only for training/evaluation.

8. **Data duplication quirk** — 3 configs (catalog + 2 targets) share one
   2000-row CSV → 3 parquet copies after registration. Consistent with the
   existing multi-target pattern (semiconductor_yield has 4+ copies).
   *Mitigation*: document as a known quirk; 2000 rows × 3 is negligible.

9. **predict-on-search records mode** — Requires the model to support
   records-based prediction (no rolling-window transforms). The supply
   features are all `current`/`categorical` transforms, so
   `entity_lookup_or_records` mode applies. *Mitigation*: verify
   `get_model_inputs` reports records mode before building predict-on-search;
   if a future model uses rolling windows, predict-on-search falls back to
   entity-lookup per product_id.

10. **`ecc` column type** — Mixed empty/`FALSE`/`TRUE`. If declared boolean,
    empty → null, `FALSE`→0, `TRUE`→1. If declared categorical, all stay as
    strings. *Mitigation*: declare as `categorical` to preserve the literal
    values and avoid silent coercion; the search contract lists the distinct
    values.

11. **MCP tool count drift in docs** — Adding tools changes the documented
    count in multiple files. *Mitigation*: grep for the old count and update
    all occurrences as part of each commit point's doc step.
