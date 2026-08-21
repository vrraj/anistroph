# Anistroph Extension Specification: Semiconductor Memory Parametric Search

## 1. Purpose

Extend Anistroph with a new `semiconductor_memory` domain combining:

1.  **Deterministic parametric product search** over a structured
    semiconductor memory catalog.
2.  **Predictive analytics** on shortlisted products using synthetic
    operational supply features.
3.  **Cross-system technical-document RAG**, where Claude or another AI
    agent can pass shortlisted products to Aina-Veris for
    document-grounded technical analysis.

Product identities and exact catalog rows are synthetic. Engineering
parameter domains use real memory-industry values and family-specific
constraints.

The architecture keeps structured filtering, predictive execution, and
unstructured document retrieval separate.

## 2. User Query Scenarios

### 2.1 Parametric Search Only

> Find production DDR5 components with at least 24 Gb density, x8
> organization, 6400 MT/s or faster, and support for operation at 55°C.

Expected flow:

**Claude discovers dataset -\> inspects search contract -\> normalizes
requirements -\> Anistroph applies deterministic filters -\> matching
products**

The user does not need to use exact catalog vocabulary. For example,
`55°C` is interpreted as:

``` text
operating_temp_min_c <= 55
AND
operating_temp_max_c >= 55
```

Other examples:

-   `industrial temperature` -\> configured temperature-range rule.
-   `at least 6400 MT/s` -\> `data_rate_mt_s >= 6400`.
-   `x8 organization` -\> `bus_width_bits = 8`.
-   `production only` -\> `part_status = Production`.

The LLM may interpret intent, but Anistroph performs the final
deterministic matching.

### 2.2 Parametric Search + Prediction

> Find production DDR5 x8 components with at least 24 Gb density and
> 6400 MT/s. Of the matching products, which have the highest predicted
> supply risk over the next four weeks?

Flow:

``` text
Natural-language request
        |
        v
Structured parametric constraints
        |
        v
Deterministic product shortlist
        |
        v
Supply-risk / lead-time model
        |
        v
Ranked candidates + explanations
```

Initial predictive targets:

**`supply_risk_next_4w`** - Binary classification. - Indicates whether a
product enters a constrained-supply state within four weeks. - Features
include inventory coverage, backlog, lead time, supplier OTD, demand
trend, PO coverage, allocation status, and lifecycle status.

**`lead_time_next_4w_days`** - Regression. - Predicts expected
replenishment lead time over the next four weeks. - Uses the same
synthetic operational supply context.

These targets are deliberately synthetic. They are not inferred from
public catalog attributes alone.

### 2.3 Parametric Search + Prediction + RAG

> Find production DDR5 components with at least 24 Gb density, x8
> organization, 6400 MT/s or faster, and support for 55°C operation.
> Rank the matches by predicted four-week supply risk. For the three
> lowest-risk products, compare power-management behavior and
> initialization requirements from their technical documentation.

Expected orchestration:

``` text
Claude / AI Agent
      |
      +--> Anistroph parametric search
      |       -> matching product IDs
      |
      +--> Anistroph predictive model
      |       -> supply-risk scores / explanations
      |
      +--> Aina-Veris MCP
              -> technical-document retrieval
              -> grounded comparison
```

Responsibilities:

-   **Anistroph search:** structured, deterministic product matching.
-   **Anistroph prediction:** model-based outcomes such as supply risk
    and lead time.
-   **Aina-Veris:** retrieval from technical documents.
-   **Claude/agent:** orchestration, intent interpretation, and
    presentation.

### 2.4 Dataset Profiling and Multidimensional Analysis

Claude and other AI agents should also be able to summarize and analyze the structured memory catalog before or independently of product search. Aggregation is executed by Anistroph; the agent interprets and presents the returned results rather than receiving the full dataset and calculating groupings itself.

Example queries:

> Profile the semiconductor memory catalog. How is the portfolio distributed across product family, technology, density, data rate, and product status? Highlight notable patterns.

> Summarize the catalog by product family. Show product count, density range, available data rates, bus widths, and operating-temperature ranges for each family.

> For DDR5 components, group products by density and data rate and show the number of products in each combination.

This should reuse Anistroph's shared analysis / multidimensional-analysis capability rather than introduce memory-specific aggregation logic. Typical dimensions include `product_family`, `technology`, `component_density_gb`, `module_density_gb`, `data_rate_mt_s`, `bus_width_bits`, `part_status`, and `package`.

Recommended agent flow:

**Discover dataset -> profile / group -> interpret portfolio -> parametric search -> optionally predict -> optionally invoke Aina-Veris**

## 3. Dataset Design

### 3.1 Dataset ID

`semiconductor_memory`

### 3.2 Primary Key

`product_id`

Each row represents one fictional sellable memory SKU plus a current
synthetic supply snapshot.

### 3.3 Catalog / Search Fields

-   `product_id` - fictional stable product identifier
-   `product_family` - DDR5_COMPONENT, LPDDR5X_COMPONENT, DDR5_RDIMM,
    DDR5_UDIMM
-   `technology` - DDR5 or LPDDR5X
-   `part_type` - COMPONENT or MODULE
-   `component_density_gb`
-   `module_density_gb`
-   `speed_mhz`
-   `data_rate_mt_s`
-   `io_voltage_v`
-   `operating_temp_min_c`
-   `operating_temp_max_c`
-   `bus_width_bits`
-   `cas_latency`
-   `pin_count`
-   `package`
-   `component_config`
-   `ecc`
-   `part_status`
-   `package_width_mm`
-   `package_length_mm`
-   `package_height_mm`
-   `datasheet_id`

### 3.4 Synthetic Supply / Prediction Fields

-   `inventory_units`
-   `weekly_demand_units`
-   `inventory_coverage_weeks`
-   `backlog_units`
-   `backlog_ratio`
-   `open_po_units`
-   `open_po_coverage`
-   `supplier_lead_time_days`
-   `supplier_otd_pct`
-   `demand_trend_4w_pct`
-   `allocation_status`
-   `supply_risk_next_4w` - classification target
-   `lead_time_next_4w_days` - regression target

Public product catalogs do not provide the operational history required
for these prediction targets; these fields are synthetic reference data.

### 3.5 Family-Specific Parameter Rules

**DDR5 components** - Density: 16, 24, 32 Gb - Data rate: 4800, 5200,
5600, 6400, 7200 MT/s - Bus width: x4, x8, x16 - I/O voltage: 1.1 V -
Operating ranges include 0..95°C and -40..95°C - CAS latency tied to
data rate - 78-, 82-, or 102-ball package options

**LPDDR5X components** - Density: 16, 24, 32, 48, 64, 96, 128 Gb - Data
rate: 6400, 7500, 8533 MT/s - Bus width: x32 or x64 - I/O voltage: 0.5
V - Extended temperature-range options - 315-, 441-, 496-, or 561-ball
package options

**DDR5 modules** - Module density: 16, 32, 64, 128 GB - Data rates:
5600, 6400, 8000 MT/s where applicable - RDIMM bus width: x72/x80 -
UDIMM bus width: x64 - I/O voltage: 1.1 V

## 4. Example Records

### DDR5 Component

``` json
{
  "product_id": "ANM-D5C-0007",
  "product_family": "DDR5_COMPONENT",
  "technology": "DDR5",
  "part_type": "COMPONENT",
  "component_density_gb": 32,
  "data_rate_mt_s": 6400,
  "io_voltage_v": 1.1,
  "operating_temp_min_c": -40,
  "operating_temp_max_c": 95,
  "bus_width_bits": 8,
  "cas_latency": "CL52",
  "pin_count": 82,
  "package": "VFBGA",
  "part_status": "Production"
}
```

### LPDDR5X Component

``` json
{
  "product_id": "ANM-L5X-0031",
  "product_family": "LPDDR5X_COMPONENT",
  "technology": "LPDDR5X",
  "part_type": "COMPONENT",
  "component_density_gb": 64,
  "data_rate_mt_s": 8533,
  "io_voltage_v": 0.5,
  "operating_temp_min_c": -40,
  "operating_temp_max_c": 105,
  "bus_width_bits": 32,
  "cas_latency": "Programmable",
  "pin_count": 315,
  "package": "TFBGA",
  "part_status": "Production"
}
```

### DDR5 RDIMM

``` json
{
  "product_id": "ANM-RD5-0018",
  "product_family": "DDR5_RDIMM",
  "technology": "DDR5",
  "part_type": "MODULE",
  "module_density_gb": 64,
  "data_rate_mt_s": 6400,
  "io_voltage_v": 1.1,
  "operating_temp_min_c": 0,
  "operating_temp_max_c": 95,
  "bus_width_bits": 80,
  "cas_latency": "CL52",
  "package": "VFBGA",
  "part_status": "Production"
}
```

## 5. Parametric Query Contract

Add a generic structured-search service rather than memory-specific
filtering logic:

``` python
search_dataset(
    dataset_id: str,
    filters: list[FilterExpression],
    sort: list[SortExpression] | None = None,
    limit: int = 50
)
```

Minimum operators:

-   `eq`
-   `in`
-   `gte`
-   `lte`
-   `between`
-   `contains_range`

Example normalized query for `supports operation at 55°C`:

``` json
{
  "field": "operating_temperature",
  "op": "contains_range",
  "value": 55,
  "min_field": "operating_temp_min_c",
  "max_field": "operating_temp_max_c"
}
```

Dataset-level semantic rules can map natural engineering concepts to
deterministic filters:

``` yaml
semantic_filters:
  operating_temperature:
    type: range_contains
    min_field: operating_temp_min_c
    max_field: operating_temp_max_c

  industrial_temperature:
    expands_to:
      - field: operating_temp_min_c
        op: lte
        value: -40
      - field: operating_temp_max_c
        op: gte
        value: 95
```

## 6. MCP Interface

Expose the capability through MCP for Claude testing.

Recommended tools:

``` text
anistroph_list_datasets()
anistroph_get_dataset_profile(dataset_id)
anistroph_analyze(dataset_id, dimensions, metrics?, filters?)
anistroph_get_search_contract(dataset_id)
anistroph_search(dataset_id, filters, sort?, limit?)
anistroph_get_model_inputs(model_id)
anistroph_predict(...)
anistroph_explain_prediction(...)
```

`anistroph_get_search_contract` should return searchable fields, data
types, units, categorical values, range semantics, aliases, and
supported operators.

Agent lifecycle:

**Discover -\> inspect search contract -\> normalize requirements -\>
search -\> optionally predict -\> optionally invoke Aina-Veris**

## 7. Web UX

Add a simple Parametric Search view for `semiconductor_memory`:

-   Product family / technology
-   Density
-   Minimum data rate
-   Bus width
-   Required operating temperature
-   Package
-   Part status
-   Search / Reset
-   Sortable results grid

The operating-temperature input accepts a real requirement such as
`55°C`; the search applies range-containment semantics rather than
asking the user to select a stored temperature-range label.

Later phases can add supply-risk scoring, SHAP explanations, and an
Aina-Veris document action.

## 8. Iterative Implementation Plan

### Phase 1 - Dataset + Deterministic Search

1.  Add `semiconductor_memory` dataset configuration.
2.  Add fixed-seed synthetic catalog generator.
3.  Persist generated data as the normal Anistroph dataset artifact.
4.  Add generic search/filter service.
5.  Implement exact, range, and `contains_range` operators.
6.  Add self-describing search contract.
7.  Add MCP search tools.
8.  Add simple Web UI filters.
9.  Add tests for inferred-temperature and numeric-range queries.

Acceptance queries: - `DDR5 + x8 + >=6400 MT/s` - `supports 55°C` -
`Production + >=24 Gb + x8 + >=6400 MT/s`

### Phase 2 - Prediction on Search Results

1.  Register `supply_risk_next_4w` classification target.
2.  Register `lead_time_next_4w_days` regression target.
3.  Train/evaluate XGBoost models using synthetic supply fields.
4.  Allow prediction over a product or a search-result set.
5.  Add ranking by predicted outcome.
6.  Add explanation support.
7.  Expose through MCP and REST.
8.  Optionally add Web UI scores/explanations.

Acceptance query:

> Find matching DDR5 parts and rank them by predicted four-week supply
> risk.

### Phase 3 - Aina-Veris RAG

1.  Ingest the synthetic datasheets using `product_id` / `datasheet_id`.
2.  Keep RAG outside Anistroph and use the existing Aina-Veris MCP.
3.  Define handoff of shortlisted product IDs and document IDs.
4.  Test Claude with both MCP servers enabled.
5.  Demonstrate one query spanning search, prediction, and document
    retrieval.

Acceptance query:

> Find qualifying DDR5 products, rank by supply risk, then compare the
> three lowest-risk products for power-management behavior and
> initialization requirements using their datasheets.

### Phase 4 - Hardening

-   Pagination and limits
-   normalized-filter debug/audit output
-   unit normalization
-   richer semantic aliases
-   performance tests
-   saved searches
-   additional memory families
-   optional temporal product-supply history

## 9. Initial Non-Goals

-   Reproducing a vendor's actual catalog or part numbers
-   Redistributing vendor datasheets
-   Using RAG as a replacement for structured filtering
-   Using XGBoost for deterministic compatibility rules
-   Putting memory-specific search logic directly in MCP handlers

## 10. Reference Datasheets

Five synthetic technical documents accompany the reference data:

-   three single-product datasheets;
-   one DDR5 component family guide;
-   one DDR5 module family guide.

They use the same fictional product IDs as the catalog and contain
original synthetic technical prose suitable for Aina-Veris ingestion and
cross-system MCP demos.
