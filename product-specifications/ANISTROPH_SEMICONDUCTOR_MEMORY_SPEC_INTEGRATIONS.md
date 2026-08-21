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

### Reference Dataset Policy

The supplied `semiconductor_memory_2000.csv` is the reference source dataset for this implementation.

- Use the supplied file directly; no catalog generator is required for this phase.
- Product IDs and exact catalog rows are fictional and must not be represented as real vendor SKUs.
- Memory attributes use realistic industry parameter domains and family-specific values intended to exercise engineering parametric-search behavior.
- The dataset supports realistic engineering queries and filtering semantics, but exact row combinations are not claims of commercial product availability.
- Supply fields and predictive targets are synthetic and exist specifically for Anistroph prediction demonstrations.
- **Do not regenerate, replace, or independently synthesize the catalog during implementation unless this specification is explicitly revised.**

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

Supply and operational data should be generated separately from the fixed 2,000-product reference catalog. Use a simple fixed-seed generator to create time-based supply history for each `product_id` (for example, weekly observations).

The generated history can include inventory, demand, backlog, open purchase orders, supplier lead time, supplier on-time delivery, and allocation status. Use this history to derive future-looking targets such as `supply_risk_next_4w` and `lead_time_next_4w_days`.

The product catalog remains fixed and should not be regenerated.

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

## 5. Parametric Search Through Existing Dataset Filtering

Parametric search should **extend Anistroph's existing dataset row-filtering capability** rather than introduce a separate search subsystem or overlapping MCP tool.

The current `anistroph_sample_rows` MCP tool and underlying `svc.sample_rows()` service already support:

- equality filtering;
- IN-list filtering;
- column selection;
- sorting;
- row limits;
- access through MCP, REST, and the Web UI.

For the semiconductor-memory use case, extend this existing filter contract with the additional operators required for parametric search:

- `eq`
- `in`
- `gte`
- `lte`
- `between`
- `contains_range`

Example normalized filters for:

> Find production DDR5 components with at least 24 Gb density, x8 organization, 6400 MT/s or faster, and support for 55°C operation.

```json
{
  "dataset_id": "semiconductor_memory",
  "filters": {
    "part_status": {"eq": "Production"},
    "technology": {"eq": "DDR5"},
    "component_density_gb": {"gte": 24},
    "bus_width_bits": {"eq": 8},
    "data_rate_mt_s": {"gte": 6400},
    "operating_temperature": {"contains_range": 55}
  },
  "sort": [
    {"field": "component_density_gb", "direction": "desc"},
    {"field": "data_rate_mt_s", "direction": "desc"}
  ],
  "n": 100
}
```

`contains_range` is a semantic filter mapped to the dataset's minimum and maximum fields:

```text
operating_temp_min_c <= requested_temperature
AND
operating_temp_max_c >= requested_temperature
```

Dataset configuration may expose semantic filter metadata such as:

```yaml
search:
  enabled: true

  semantic_filters:
    operating_temperature:
      type: range_contains
      min_field: operating_temp_min_c
      max_field: operating_temp_max_c
      unit: C

    industrial_temperature:
      expands_to:
        - field: operating_temp_min_c
          op: lte
          value: -40
        - field: operating_temp_max_c
          op: gte
          value: 95
```

The LLM or application may interpret natural-language requirements into this filter structure, but the final product matching remains deterministic inside Anistroph.

### Implementation Constraint

**Do not add a new parametric-search MCP tool unless the existing dataset-filtering contract cannot be extended cleanly.**

The preferred implementation is:

```text
anistroph_sample_rows
        |
        v
svc.sample_rows()
        |
        v
Extended generic filter operators
        |
        v
Deterministic parametric results
```

This keeps parametric search available automatically through the same shared path already used by MCP, REST, and the Web UI.

## 6. MCP and Agent Flow

No new MCP tool is expected for Phase 1.

Reuse:

```text
anistroph_list_datasets
anistroph_profile_dataset
anistroph_sample_rows
anistroph_slice_data
anistroph_compare_data
anistroph_get_model_inputs
anistroph_predict
anistroph_explain_prediction
```

`anistroph_sample_rows` should be extended to accept the richer generic filter operators described above.

Agents should first discover/profile the dataset so they understand available fields and values, then translate the user's natural-language product requirements into deterministic filters.

Example lifecycle:

```text
Discover dataset
   ->
Profile / understand fields
   ->
Interpret natural-language requirements
   ->
Call anistroph_sample_rows with structured filters
   ->
Optionally analyze shortlist
   ->
Optionally invoke prediction
   ->
Optionally invoke Aina-Veris for document analysis
```

The same filtering semantics should be available through:

- MCP: `anistroph_sample_rows`
- REST: existing dataset rows endpoint
- Web UI: Data / Parametric Search controls

If the existing profile/discovery responses do not expose enough information for agents to infer valid operators, units, or range semantics, extend those responses with search metadata before considering a separate discovery tool.

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

1. Add `semiconductor_memory` dataset configuration.
2. Use the supplied `semiconductor_memory_2000.csv` directly as the reference source dataset.
3. Register the CSV through Anistroph's existing ingestion flow and persist the normal processed dataset artifacts.
4. Extend the existing `svc.sample_rows()` filtering capability rather than adding a separate search service.
5. Extend `anistroph_sample_rows` and the existing REST rows endpoint with generic comparison/range operators: `eq`, `in`, `gte`, `lte`, `between`, and `contains_range`.
6. Expose required units, field semantics, and range metadata through existing dataset profile/discovery responses where practical.
7. Add simple Web UI parametric filters using the same shared service path.
8. Add tests for equality, IN-list, numeric comparison, range, and inferred-temperature queries.
9. Verify equivalent filtering behavior across MCP, REST, and Web UI.

Acceptance queries:

- `DDR5 + x8 + >=6400 MT/s`
- `supports 55°C`
- `Production + >=24 Gb + x8 + >=6400 MT/s`

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

### Phase 3 - Aina-Veris A2A Integration

1. Ingest the synthetic datasheets into Aina-Veris using `product_id` / `datasheet_id`.
2. Keep RAG and document-retrieval implementation outside Anistroph.
3. Register the Aina-Veris semiconductor research agent as an external tool in Anistroph.
4. Invoke Aina-Veris through its A2A endpoint and Agent Card.
5. Expose the registered external capability through Anistroph MCP and REST so Claude or another external orchestrator only needs to connect to Anistroph.
6. Pass the technical research question and relevant shortlisted product/document context to the Aina-Veris agent.
7. Demonstrate one query spanning parametric search, prediction, and document retrieval.

Acceptance query:

> Find qualifying DDR5 products, rank by supply risk, then compare the three lowest-risk products for power-management behavior and initialization requirements using their datasheets.


### External Integration Registry

External agents and tools should be defined separately from Anistroph's native dataset/model tools.

Add an integration registry:

```text
integrations/
└── tool_registry.yaml
```

`tool_registry.yaml` is the configuration source for externally hosted capabilities such as Aina-Veris. The MCP server, REST API, and any internal orchestrator should consume the same registry rather than duplicating external-tool definitions.

Example:

```yaml
tools:
  - name: call_veris_semiconductor_research_agent
    provider: veris
    capability: semiconductor_memory_research
    visibility: always
    description: Query AINA Veris for grounded semiconductor-memory datasheet and application-note analysis.
    keywords:
      - semiconductor memory
      - DRAM
      - SRAM
      - NAND
      - datasheet
      - application note
    llm_parameters:
      type: object
      properties:
        prompt:
          type: string
          description: Technical semiconductor-memory research question.
      required:
        - prompt
      additionalProperties: false
    agent_owner: aina-veris
    protocol: A2A_JSONRPC
    base_url: https://<host-name>
    path: /agents/veris-semiconductor-research-agent/
```

`<host-name>` is supplied through deployment configuration. The registry should not hard-code a local/Docker/public host assumption.

The integration architecture should be:

```text
integrations/tool_registry.yaml
          |
          v
External Tool Registry / Loader
          |
          +--------------------+
          |                    |
          v                    v
      MCP Server            REST API
          |                    |
          +---------+----------+
                    |
                    v
          Shared External Tool Invoker
                    |
                    v
              A2A / Aina-Veris
```

#### MCP registration

`mcpserver.py` should load the external tool registry during server initialization and register tools whose `visibility` allows MCP exposure.

The MCP wrapper should remain thin:

```text
MCP tool call
   ->
registered external tool
   ->
shared external-tool invoker
   ->
A2A JSON-RPC request
```

Do not implement Aina-Veris-specific RAG logic in `mcpserver.py`.

#### REST exposure

REST should expose the same registered external capabilities through the shared external-tool invoker. The REST implementation should not maintain a second copy of the registry or a separate A2A implementation.

A generic invocation shape is preferred, for example:

```text
POST /integrations/tools/{tool_name}/invoke
```

with request parameters validated against the tool's `llm_parameters` schema.

The exact REST route may follow existing Anistroph conventions, but MCP and REST must resolve the same tool definition and invoke the same underlying external-tool service.

#### Implementation Constraint

External tool definitions belong in `integrations/tool_registry.yaml`.

- Do not hard-code external agent definitions in `mcpserver.py`.
- Do not duplicate definitions between MCP and REST.
- Keep transport/invocation logic shared.
- `mcpserver.py` is responsible for exposing eligible registered tools to MCP clients.
- REST is responsible for exposing the same registered capabilities to programmatic/external callers.
- Aina-Veris remains an independent A2A service.

### A2A Handoff to Aina-Veris

Anistroph registers the Aina-Veris semiconductor research agent as an external A2A capability. Claude or another external AI system continues to call Anistroph through MCP; Anistroph invokes the registered Aina-Veris agent through A2A when technical-document research is required.

```text
Claude / External AI System
          |
         MCP
          |
          v
      Anistroph
          |
          +-- Parametric Search
          +-- Predictive Analytics
          |
          +-- A2A --> Aina-Veris
                       |
                       +-- Domain-specific RAG
                       +-- Datasheets / technical documents
                       +-- Source citations
```

The Aina-Veris Agent Card is discovered at:

```text
https://<host-name>/agents/veris-semiconductor-research-agent/.well-known/agent-card.json
```

`<host-name>` is deployment configuration and must not be hard-coded. It can resolve to the appropriate local, Docker, Docker Compose, or remote host through configuration.

The Anistroph external-tool definition is stored in `integrations/tool_registry.yaml` and follows this shape:

```yaml
tools:
  - name: call_veris_semiconductor_research_agent
    provider: veris
    capability: semiconductor_memory_research
    visibility: always
    description: Query AINA Veris for grounded semiconductor-memory datasheet and application-note analysis.
    keywords:
      - semiconductor memory
      - DRAM
      - SRAM
      - NAND
      - datasheet
      - application note
    llm_parameters:
      type: object
      properties:
        prompt:
          type: string
          description: Technical semiconductor-memory research question.
      required:
        - prompt
      additionalProperties: false
    agent_owner: aina-veris
    protocol: A2A_JSONRPC
    base_url: https://<host-name>
    path: /agents/veris-semiconductor-research-agent/
```

Minimum callable signature:

```text
call_veris_semiconductor_research_agent(
    prompt: string
) -> A2A response
```

The prompt may include the technical research question and relevant shortlisted `product_id` / `datasheet_id` context from Anistroph. Aina-Veris remains responsible for domain-specific retrieval, RAG, source citations, and its supporting document tools.

No RAG implementation should be added to Anistroph.


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
