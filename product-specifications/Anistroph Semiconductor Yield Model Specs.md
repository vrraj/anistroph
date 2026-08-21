# Anistroph Semiconductor Yield Model — v0.1 Specification

## Objective

Extend the existing Anistroph architecture with a **semiconductor wafer-yield reference implementation**.

This implementation should use one primary predictive model:

> **Predict wafer-level yield from manufacturing process history and wafer-level process features.**

Do not build a die-level predictive model in v0.1.

Die-level data should still exist because it is required to calculate actual wafer yield and support wafer-map visualization and later analysis.

---

# 1. Prediction Target

The model predicts:

```text
wafer_yield
```

where:

```text
wafer_yield =
number_of_good_dies / number_of_tested_dies
```

Example:

```text
Wafer W123

Tested dies: 500
Good dies:   472

Actual yield: 94.4%
```

The ML task is:

```text
Regression
```

The output is a continuous value such as:

```text
0.944
```

---

# 2. Data Hierarchy

Preserve the manufacturing hierarchy:

```text
Lot
 └── Wafer
      ├── Process History
      └── Die Test Results
```

Use identifiers:

```text
lot_id
wafer_id
die_id
```

Die-level coordinates:

```text
die_x
die_y
```

must be retained.

---

# 3. Physical Dataset Structure

Do not force all semiconductor data into one giant flat source file.

Generate and persist at least these logical datasets.

## Lot Table

```text
lot_id
product_id
fab_id
process_node
lot_start_timestamp
```

## Wafer Table

```text
wafer_id
lot_id
wafer_number
wafer_start_timestamp
wafer_complete_timestamp
```

## Process History Table

One wafer can have many manufacturing operations.

```text
wafer_id
operation_sequence
operation_id

tool_id
chamber_id
recipe_id

process_start_timestamp
process_end_timestamp

temperature_mean
temperature_std

pressure_mean
pressure_std

gas_flow_mean
gas_flow_std

rf_power_mean
rf_power_std

process_time
```

## Die Test Table

```text
die_id
wafer_id

die_x
die_y

pass_fail
bin
test_timestamp
```

Persist analytical data as Parquet.

DuckDB/Polars should be used to query and transform it.

---

# 4. Synthetic Dataset Generator

Create:

```text
scripts/generate_semiconductor_yield_data.py
```

Generate approximately:

```text
100 lots
25 wafers per lot
~500 dies per wafer
```

Allow smaller configurable sizes for automated tests.

The generator should produce realistic but synthetic:

```text
lot data
wafer data
process history
tool assignments
chamber assignments
recipes
process measurements
die coordinates
die pass/fail results
```

---

# 5. Inject Learnable Yield Effects

Do not generate yield randomly.

Inject several relationships so the model has meaningful structure to learn.

Example:

```text
Normal process:
~97% yield
```

Introduce effects such as:

```text
ETCH_02
→ small negative effect
```

```text
CH_B
→ small negative effect
```

but:

```text
ETCH_02 + CH_B
→ larger negative interaction
```

and:

```text
ETCH_02
+ CH_B
+ high temperature variability
→ substantial yield degradation
```

Example expected pattern:

```text
Normal                                  ~97%
ETCH_02                                 ~96%
CH_B                                    ~96%
ETCH_02 + CH_B                          ~93%
ETCH_02 + CH_B + high temp variability  ~87–90%
```

Add random noise and lot/wafer variation.

No single feature should perfectly determine yield.

---

# 6. Die-Level Yield Generation

Generate die pass/fail outcomes from the process conditions.

Support optional spatial effects such as:

```text
distance_from_center
distance_from_edge
wafer quadrant
```

For example:

```text
certain degraded process conditions
+
wafer edge
→ slightly higher failure probability
```

Actual wafer yield should then be derived from the generated die results.

Do not directly assign wafer yield independently of the die outcomes.

---

# 7. Dataset Specification

Create:

```text
datasets/semiconductor_yield/
```

with approximately:

```text
dataset.yaml
features.yaml
targets.yaml
```

Example dataset semantics:

```yaml
dataset_id: semiconductor_yield
name: Semiconductor Wafer Yield

hierarchy:

  lot:
    key: lot_id

  wafer:
    key: wafer_id
    parent: lot_id

  die:
    key: die_id
    parent: wafer_id
    coordinates:
      x: die_x
      y: die_y
```

The generic Anistroph engine must understand this through configuration.

Do not hard-code semiconductor semantics into generic training code.

---

# 8. Model Grain

The training dataset must have:

```text
one row = one wafer
```

Do not train directly on die rows.

The feature pipeline must aggregate process history into a single wafer feature vector.

---

# 9. Initial Wafer Features

Include categorical manufacturing context:

```text
product_id
fab_id
```

and process-specific categorical features such as:

```text
etch_tool_id
etch_chamber_id
etch_recipe_id

deposition_tool_id
deposition_chamber_id

lithography_tool_id
```

Also include numerical process features such as:

```text
etch_temperature_mean
etch_temperature_std

etch_pressure_mean
etch_pressure_std

etch_gas_flow_mean
etch_gas_flow_std

etch_rf_power_mean
etch_rf_power_std

etch_process_time
```

Equivalent features can be generated for other configured process steps.

---

# 10. Feature Engineering

The feature engine should transform process history into wafer-level features using configuration.

Conceptually:

```text
Raw process history

wafer_id
operation
tool
sensor statistics
       ↓
Feature Engine
       ↓
One wafer feature vector
```

Example:

```text
wafer_id
product_id
fab_id

etch_tool_id
etch_chamber_id
etch_temperature_mean
etch_temperature_std

deposition_tool_id
deposition_temperature_mean

lithography_tool_id

wafer_yield
```

Training and inference must use the exact same feature-generation logic.

---

# 11. Target Specification

Create a target definition approximately:

```yaml
target:

  name: wafer_yield
  level: wafer
  type: regression

  source:
    table: die_test

  calculation:
    numerator:
      column: pass_fail
      value: true

    denominator:
      column: die_id
      aggregation: count

    aggregation: ratio
```

---

# 12. Models

Implement two regression models for comparison.

## Baseline

```text
Linear Regression or Elastic Net
```

Purpose:

```text
simple baseline
interpretability
pipeline validation
```

## Primary

```text
XGBoost Regressor
```

Do not implement broad hyperparameter search.

Use reasonable configurable defaults.

---

# 13. Data Split

Use a chronological split based on wafer completion/start time.

Example:

```text
first 70% of wafers by time → train
next 15%                   → validation
last 15%                   → test
```

Do not randomly distribute future wafers into the training set.

---

# 14. Evaluation

For regression calculate:

```text
MAE
RMSE
R²
```

Also calculate basic yield error summaries:

```text
mean prediction error
median absolute error
95th percentile absolute error
```

Store metrics with the model artifact.

---

# 15. Model Registry

Persist the fitted model:

```text
artifacts/models/
    wafer-yield-xgb-v001/
```

Include:

```text
model.json
metadata.json
feature_spec.json
target_spec.json
metrics.json
```

Metadata should include:

```text
model_id
model_type
dataset_id
dataset_version

created_at

feature list
target

train period
validation period
test period

hyperparameters
metrics
```

---

# 16. Inference API

The caller should not send the engineered feature vector.

Request:

```json
{
  "model_id": "wafer-yield-xgb-v001",
  "entity_id": "WAFER_1047"
}
```

Anistroph should:

```text
wafer_id
  ↓
retrieve wafer information
  ↓
retrieve lot context
  ↓
retrieve process history
  ↓
Feature Engine
  ↓
construct trained feature vector
  ↓
load model
  ↓
predict
```

Return:

```json
{
  "wafer_id": "WAFER_1047",
  "predicted_yield": 0.931,
  "model_id": "wafer-yield-xgb-v001"
}
```

---

# 17. Prediction Explanation

Provide model-derived feature contributions.

Use:

```text
XGBoost feature importance
```

and SHAP if practical.

Example:

```json
{
  "wafer_id": "WAFER_1047",
  "predicted_yield": 0.931,

  "top_drivers": [
    {
      "feature": "etch_chamber_id",
      "value": "CH_B",
      "impact": -0.027
    },
    {
      "feature": "etch_temperature_std",
      "value": 4.7,
      "impact": -0.018
    }
  ]
}
```

Do not use an LLM to generate feature importance.

---

# 18. Analytical Slicing

The semiconductor dataset must also work with the generic Anistroph analytical layer.

Support queries such as:

```text
yield by fab
yield by product
yield by tool
yield by chamber
yield by recipe
yield by lot

yield by tool × chamber
yield by tool × recipe
yield by chamber × temperature range
```

Example:

```python
slice_data(
    dataset_id="semiconductor_yield",
    target="wafer_yield",
    dimensions=[
        "etch_tool_id",
        "etch_chamber_id"
    ],
    aggregation="mean"
)
```

Expected result:

```text
Tool      Chamber    Wafers    Mean Yield

ETCH_01   CH_A       843       97.4%
ETCH_01   CH_B       817       97.1%
ETCH_02   CH_A       902       96.8%
ETCH_02   CH_B       791       91.9%
```

---

# 19. Actual vs Predicted Analysis

Support analysis of both:

```text
actual_yield
predicted_yield
prediction_error
```

This allows queries such as:

```text
prediction error by fab

prediction error by product

prediction error by tool

prediction error by chamber

prediction error by tool × chamber
```

This will later be useful for detecting where the model behaves poorly.

---

# 20. Wafer Map

Create a basic wafer-map visualization from:

```text
die_x
die_y
pass_fail
bin
```

The UI should allow selecting a wafer and viewing its die results spatially.

Example:

```text
          ○ ○ ○ ○
       ○ ○ ○ ○ ○ ○
     ○ ○ ○ X X ○ ○ ○
    ○ ○ ○ X X X ○ ○ ○
     ○ ○ ○ ○ X ○ ○ ○
       ○ ○ ○ ○ ○ ○
          ○ ○ ○ ○
```

This is visualization/analysis only in v0.1.

Do not train a die-level model yet.

---

# 21. REST Endpoints

Reuse the generic Anistroph APIs wherever possible.

Example:

```text
GET  /datasets/{dataset_id}/profile

POST /analysis/slice
POST /analysis/compare

POST /models/train
GET  /models/{model_id}
GET  /models/{model_id}/metrics

POST /predictions
POST /predictions/explain
```

Add a wafer-data endpoint only if useful:

```text
GET /datasets/{dataset_id}/entities/{wafer_id}
```

or another generic entity endpoint.

Avoid introducing semiconductor-specific REST architecture unless necessary.

---

# 22. MCP

The existing generic MCP tools should work with this dataset:

```text
anistroph_profile_dataset

anistroph_slice_data

anistroph_compare_data

anistroph_list_models

anistroph_get_model_metrics

anistroph_predict

anistroph_explain_prediction
```

Do not build semiconductor-specific MCP business logic.

---

# 23. Testing

## Synthetic Data Tests

Verify:

```text
lots contain wafers

wafers contain dies

process history exists for each wafer

die coordinates are valid

actual wafer yield equals aggregate die results
```

Verify that the injected hidden relationships are statistically present.

For example:

```text
ETCH_02 + CH_B
```

should have lower mean yield than the overall baseline.

---

## Feature Tests

Verify:

```text
one feature row per wafer

correct process-history aggregation

categorical fields preserved

numerical fields correctly aggregated

no target leakage
```

The model input must not include:

```text
pass_fail
bin
actual wafer_yield
future test results
```

as prediction features.

---

## Model Tests

Verify:

```text
baseline trains successfully

XGBoost trains successfully

models beat a trivial mean-yield predictor

persist/reload preserves predictions
```

Do not require unrealistically high R².

---

# 24. End-to-End Acceptance Test

Implement:

```text
generate semiconductor data
          ↓
register dataset
          ↓
persist Parquet
          ↓
profile
          ↓
calculate actual wafer yield
          ↓
generate wafer feature matrix
          ↓
chronological split
          ↓
train baseline
          ↓
train XGBoost
          ↓
evaluate
          ↓
persist model
          ↓
reload
          ↓
select wafer
          ↓
rebuild inference feature vector
          ↓
predict wafer yield
          ↓
explain prediction
          ↓
slice yield by tool/chamber
          ↓
display wafer map
```

---

# 25. Scope Boundary

For v0.1 implement **one predictive semiconductor model only**:

```text
Wafer-level yield regression
```

Do not implement:

```text
die failure prediction
defect-image models
equipment predictive maintenance inside this dataset
deep neural networks
process-control optimization
AutoML
hyperparameter search platform
```

Die-level data exists for:

```text
yield calculation
wafer visualization
future extensibility
```

---

# 26. Extensibility Requirement

The semiconductor implementation must use the same generic Anistroph:

```text
Dataset Registry
Feature Engine
Target Engine
Training Engine
Evaluation Engine
Model Registry
Inference Engine
Analysis Engine
REST API
MCP layer
```

used by the predictive-maintenance reference implementation.

Adding semiconductor yield must not result in a second independent ML architecture.

The purpose of this implementation is to prove that Anistroph can support a second, structurally different domain using the same platform abstractions.

---

# 27. Architectural Intent

This prototype should demonstrate three independent capabilities over the same semiconductor data:

```text
                    Semiconductor Data
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
          ▼                ▼                 ▼
       ANALYZE           PREDICT          VISUALIZE

   slice / compare    wafer yield       wafer maps
   interactions      XGBoost            die patterns
   dimensions        regression
```

The predictive model is only one part of Anistroph.

The longer-term product direction remains:

> **Discover how outcomes change across different dimensions and combinations of data, and use those relationships for analysis and prediction.**