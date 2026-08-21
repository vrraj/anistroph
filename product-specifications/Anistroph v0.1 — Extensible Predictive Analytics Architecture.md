# Anistroph v0.1 — Extensible Predictive Analytics Architecture

## 1. Objective

Build a Python prototype of **Anistroph**, a domain-agnostic analytical and predictive platform for structured datasets.

The first reference implementation will use **semiconductor/manufacturing equipment sensor data for predictive maintenance**, but the architecture must not contain predictive-maintenance-specific assumptions in the core pipeline.

Future datasets should include use cases such as:

- semiconductor yield analysis
- manufacturing quality
- supply-chain prediction
- CRM conversion/propensity
- equipment failure
- other tabular and time-series prediction problems

The core lifecycle is:

```text
Dataset
   ↓
Dataset Specification
   ↓
Validation / Profiling
   ↓
Feature Engineering
   ↓
Target Construction
   ↓
Training
   ↓
Evaluation
   ↓
Persisted Model
   ↓
Inference
   ↓
Analysis / Explanation
```

Anistroph is **not an AutoML platform** and does not train foundation models.

---

# 2. Architectural Principle

Dataset-specific concepts must never be hard-coded into the generic ML pipeline.

For example, the core engine must not contain assumptions such as:

```text
machine_id
temperature
vibration
failure
24-hour failure horizon
```

Instead:

```text
             Dataset
                ↓
          DatasetSpec
                ↓
       ┌────────┼────────┐
       ↓        ↓        ↓
   FeatureSpec TargetSpec ModelSpec
       │        │        │
       └────────┼────────┘
                ↓
        Generic Pipeline
```

Predictive maintenance is a **reference configuration**, not an architectural dependency.

---

# 3. Technology Stack

| Category | Technology | Role |
| --- | --- | --- |
| Runtime | Python | Application and ML implementation |
| Web framework | FastAPI | REST APIs and static UI |
| ASGI | Uvicorn | Application runtime |
| Dataframe engine | Polars | Primary transformations |
| Analytical SQL | DuckDB | Querying and slicing analytical data |
| Storage format | Parquet | Persistent analytical datasets |
| ML | scikit-learn | Preprocessing, baselines and metrics |
| Gradient boosting | XGBoost | Initial primary predictive algorithm |
| Model persistence | joblib / native XGBoost | Persist fitted models |
| Visualization | Plotly or lightweight JS | Charts and analysis |
| MCP | Python MCP SDK | External tool interface |
| Testing | pytest | Unit/integration/API tests |

Pandas may be used where required by ML libraries.

Do not introduce PostgreSQL, Spark, Kafka, MLflow, Airflow, Kubernetes, vector databases or agent frameworks for v0.1.

---

# 4. High-Level Architecture

```text
                         Web UI
                            │
                         FastAPI
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
     Dataset API        Analysis API         Model API
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ↓
                    ANISTROPH CORE
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
 Dataset Registry      Feature Engine       ML Engine
        │                   │                    │
 DatasetSpec           FeatureSpec            Models
        │                   │                    │
        └──────────────┬────┴─────────────┬─────┘
                       │                  │
                   DuckDB/Polars     Model Registry
                       │                  │
                    Parquet          Model Artifacts
                       
                            ▲
                            │
                       MCP Server
                            │
                    External AI/Agent
```

All interfaces must ultimately invoke the same core Python services.

---

# 5. Repository Structure

Use approximately:

```text
anistroph/
│
├── backend/
│   ├── main.py
│
│   ├── api/
│   │   ├── datasets.py
│   │   ├── analysis.py
│   │   ├── models.py
│   │   └── predictions.py
│
│   ├── datasets/
│   │   ├── registry.py
│   │   ├── spec.py
│   │   ├── loader.py
│   │   ├── validation.py
│   │   └── profiling.py
│
│   ├── features/
│   │   ├── engine.py
│   │   ├── spec.py
│   │   ├── numeric.py
│   │   ├── categorical.py
│   │   ├── temporal.py
│   │   └── rolling.py
│
│   ├── targets/
│   │   ├── spec.py
│   │   ├── binary.py
│   │   ├── regression.py
│   │   └── horizon.py
│
│   ├── ml/
│   │   ├── base.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── inference.py
│   │   ├── explain.py
│   │   └── registry.py
│
│   ├── models/
│   │   ├── logistic.py
│   │   └── xgboost.py
│
│   ├── analysis/
│   │   ├── slice.py
│   │   ├── aggregate.py
│   │   └── compare.py
│
│   ├── integrations/
│   │   └── mcp/
│   │       ├── server.py
│   │       └── tools.py
│
│   └── schemas/
│
├── datasets/
│   ├── predictive_maintenance/
│   │   └── dataset.yaml
│   └── examples/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── artifacts/
│   └── models/
│
├── scripts/
│   └── generate_sensor_data.py
│
├── frontend/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── docs/
│   └── architecture.md
│
├── README.md
├── TECHNICAL_ARCHITECTURE.md
├── pyproject.toml
└── docker-compose.yml
```

---

# 6. Dataset Specification

Introduce a first-class `DatasetSpec`.

The specification describes the meaning of a dataset independently of its physical data.

Example predictive-maintenance specification:

```yaml
dataset_id: predictive_maintenance
name: Predictive Maintenance Demo

entity_key: machine_id
time_key: timestamp

columns:

  machine_id:
    type: categorical
    role: identifier

  machine_type:
    type: categorical
    role: feature

  temperature:
    type: numeric
    role: feature

  vibration:
    type: numeric
    role: feature

  pressure:
    type: numeric
    role: feature

  current:
    type: numeric
    role: feature

  voltage:
    type: numeric
    role: feature

  rpm:
    type: numeric
    role: feature

  maintenance_age_hours:
    type: numeric
    role: feature

  failure:
    type: boolean
    role: event
```

The DatasetSpec should be represented internally using Pydantic models.

---

# 7. Feature Specification

Feature engineering must also be configuration-driven.

Example:

```yaml
features:

  temperature:
    transforms:
      - current
      - mean:
          windows: [1h, 6h]
      - std:
          windows: [1h, 6h]
      - slope:
          windows: [6h]

  vibration:
    transforms:
      - current
      - mean:
          windows: [1h, 6h]
      - max:
          windows: [6h]
      - std:
          windows: [1h, 6h]
      - slope:
          windows: [6h]

  pressure:
    transforms:
      - current
      - mean:
          windows: [1h]
      - std:
          windows: [6h]

  maintenance_age_hours:
    transforms:
      - current
```

The generic Feature Engine interprets these definitions.

It must not know what temperature or vibration means.

---

# 8. Supported Generic Feature Operations

Initial numeric operations:

```text
current
mean
min
max
std
median
delta
slope
```

Initial categorical handling:

```text
one-hot encoding
unknown-category handling
```

Temporal features may include:

```text
hour_of_day
day_of_week
elapsed_time
rolling windows
```

Only implement operations required by the reference use case initially, but design the transform registry to allow additional operations later.

---

# 9. Target Specification

Targets must also be configurable.

Support three conceptual target types architecturally:

```text
binary
regression
future_event
```

Only `future_event/binary` must be fully exercised by the v0.1 reference implementation.

Predictive-maintenance example:

```yaml
target:
  name: failure_within_horizon
  type: future_event
  source_column: failure
  horizon: 24h
```

The engine generates:

```text
failure_within_horizon = 1
```

when a failure occurs for the same entity inside the configured future horizon.

No future observations may enter the feature vector.

---

# 10. Reference Dataset

Generate synthetic semiconductor/manufacturing sensor data.

Minimum columns:

```text
timestamp
machine_id
machine_type

temperature
vibration
pressure
current
voltage
rpm
flow_rate

maintenance_age_hours
operating_hours

failure
failure_type
```

Suggested initial dataset:

```text
50 machines
60 days
5-minute observations
```

Inject learnable deterioration patterns.

For example:

```text
increasing vibration
+
temperature drift
+
pressure instability
+
maintenance age
        ↓
increased probability of failure
```

Include:

- normal operating noise
- differences between machines
- differences between machine types
- occasional anomalous readings
- realistic failure imbalance

Failures must not be randomly assigned independently of sensor behavior.

---

# 11. Data Ingestion

Support:

```text
CSV
Parquet
```

CSV ingestion should convert the registered analytical dataset to Parquet.

Store analytical data using Parquet and query through DuckDB/Polars.

Dataset ingestion must:

```text
read file
↓
apply DatasetSpec
↓
validate required columns
↓
validate types
↓
validate entity/time keys
↓
profile data
↓
persist Parquet
↓
register dataset
```

---

# 12. Dataset Registry

Maintain metadata independently from the physical Parquet file.

A registered dataset should expose:

```text
dataset_id
name
version
source
row_count
columns
entity_key
time_key
target definitions
feature specification
created_at
data range
physical location
```

Use lightweight local persistence for v0.1.

Do not require PostgreSQL.

---

# 13. Dataset Profiling

Generic profiling should return:

```text
row count
column count
column types
missing values
unique counts
numeric distributions
categorical distributions
time range
entity count
target/event distribution
```

Profiling must operate from DatasetSpec rather than domain-specific assumptions.

---

# 14. Training Pipeline

Implement a generic contract approximately:

```python
train_model(
    dataset_id,
    target_name,
    model_type,
    feature_spec=None,
    model_parameters=None
)
```

Pipeline:

```text
Dataset Registry
      ↓
DatasetSpec
      ↓
load Parquet
      ↓
Feature Engine
      ↓
Target Engine
      ↓
training matrix X/y
      ↓
chronological split
      ↓
preprocessing
      ↓
model.fit()
      ↓
evaluation
      ↓
persist model
      ↓
register model
```

Training is explicit.

Inference must never retrain automatically.

---

# 15. Training and Inference Feature Parity

This is a critical architectural requirement.

There must be **one Feature Engine** used by both training and inference.

Never implement:

```text
training_features.py
inference_features.py
```

with duplicated logic.

Instead:

```text
                  FeatureSpec
                       │
                 Feature Engine
                  ↙          ↘
             Training      Inference
```

The persisted model artifact must retain the feature specification used during training.

---

# 16. Train / Validation / Test

For datasets with a `time_key`, default to chronological splitting.

Example:

```text
70% training
15% validation
15% test
```

Do not randomly split time-series rows.

For non-temporal datasets, allow a conventional randomized split.

Split strategy should therefore be part of configuration:

```yaml
split:
  strategy: chronological
  train: 0.70
  validation: 0.15
  test: 0.15
```

---

# 17. Initial Model Implementations

Implement two classifiers for the reference dataset.

## Logistic Regression

Used as an interpretable baseline.

## XGBoost Classifier

Used as the primary nonlinear model.

Both must implement a common abstraction approximately:

```python
class Predictor:

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...

    def predict_proba(self, X):
        ...

    def save(self, path):
        ...

    @classmethod
    def load(cls, path):
        ...
```

Do not build extensive hyperparameter optimization in v0.1.

---

# 18. Evaluation

For binary classification:

```text
ROC-AUC
PR-AUC
precision
recall
F1
confusion matrix
```

Because equipment failures are rare, emphasize:

```text
PR-AUC
recall
precision
```

Evaluation should support configurable decision thresholds.

Evaluation logic must be model-independent.

---

# 19. Model Registry

Persist trained models as durable artifacts.

Example:

```text
artifacts/models/

predictive-maintenance-xgb-v001/
    model.json
    metadata.json
    feature_spec.json
    target_spec.json
    metrics.json
```

Metadata:

```text
model_id
model_type
dataset_id
dataset_version
created_at

target_spec
feature_spec

training_period
validation_period
test_period

hyperparameters
metrics
decision_threshold
```

The registry should be abstracted so another persistence implementation could replace filesystem storage later.

Do not add MLflow for v0.1.

---

# 20. Inference Contract

The external caller should normally **not construct engineered model features**.

For a temporal entity-based dataset, inference can accept:

```json
{
  "model_id": "predictive-maintenance-xgb-v001",
  "entity_id": "TOOL_047",
  "timestamp": "2026-08-14T12:00:00"
}
```

Anistroph performs:

```text
model_id
   ↓
retrieve ModelSpec + FeatureSpec
   ↓
identify dataset
   ↓
retrieve required historical observations
   ↓
Feature Engine
   ↓
exact model feature vector
   ↓
persisted model
   ↓
predict_proba()
```

This prevents clients from needing to understand model internals.

---

# 21. Generic Prediction Interface

The core inference service should use generic terminology:

```python
predict(
    model_id,
    entity_id=None,
    timestamp=None,
    records=None
)
```

This allows future non-temporal datasets to provide records directly.

Do not name the core function:

```python
predict_machine_failure()
```

That name may exist only as a convenience wrapper/reference endpoint if needed.

---

# 22. Explainability

Provide generic model explanation functionality.

Initially:

```text
XGBoost feature importance
SHAP if reasonably lightweight
```

Prediction explanation should return structured data:

```json
{
  "prediction": 0.82,
  "top_drivers": [
    {
      "feature": "vibration_mean_6h",
      "impact": 0.31
    },
    {
      "feature": "temperature_slope_6h",
      "impact": 0.24
    }
  ]
}
```

No LLM should generate or fabricate model drivers.

---

# 23. Analytical Engine

Begin establishing Anistroph's separate deterministic analytical layer.

Implement generic operations:

```text
slice
filter
group
aggregate
compare
```

Examples:

```python
slice_data(
    dataset_id,
    dimensions=["machine_type"],
    metric="failure",
    aggregation="mean"
)
```

or later:

```python
slice_data(
    dataset_id="wafer_yield",
    dimensions=["fab", "tool"],
    metric="yield",
    aggregation="mean"
)
```

These operations should use DuckDB/Polars.

They are independent of ML training.

---

# 24. REST API

Implement approximately:

```text
GET  /health

POST /datasets
GET  /datasets
GET  /datasets/{dataset_id}
GET  /datasets/{dataset_id}/profile

POST /analysis/slice
POST /analysis/compare

POST /models/train
GET  /models
GET  /models/{model_id}
GET  /models/{model_id}/metrics

POST /predictions
POST /predictions/batch

POST /predictions/explain
```

Domain-specific URLs should be avoided in the core API.

For example, prefer:

```text
POST /predictions
```

over:

```text
POST /machines/predict-failure
```

---

# 25. MCP

Expose deterministic Anistroph capabilities through MCP.

Initial tools:

```text
anistroph_list_datasets

anistroph_profile_dataset

anistroph_slice_data

anistroph_compare_data

anistroph_list_models

anistroph_get_model_metrics

anistroph_predict

anistroph_explain_prediction
```

MCP tools must call the same core services as REST.

Do not implement separate analytical logic inside MCP.

Do not expose arbitrary Python execution.

Model training should not initially be exposed through MCP.

---

# 26. Web Interface

Keep the UI functional and lightweight.

## Dataset Workspace

Allow:

```text
upload CSV/Parquet
register DatasetSpec
view schema
view profiling
view target
view available dimensions/features
```

## Analysis Workspace

Allow basic:

```text
filter
slice
group
aggregate
compare
```

## Training Workspace

Allow:

```text
select dataset
select target
select model

Logistic Regression
XGBoost

train
```

Show training status and resulting model ID.

## Model Workspace

Show:

```text
model metadata
features
target
training period
metrics
feature importance
```

## Prediction Workspace

For predictive maintenance:

```text
select machine
select timestamp
predict
```

Display:

```text
Failure probability: 82%
Prediction horizon: 24 hours
Risk: HIGH

Top drivers:
vibration_mean_6h
temperature_slope_6h
maintenance_age_hours
```

The UI may adapt controls based on DatasetSpec.

---

# 27. Extensibility Requirement

Adding another dataset should **not require changes to the core ML pipeline**.

For example, adding:

```text
wafer_yield
```

should primarily require:

```text
data file
+
DatasetSpec
+
FeatureSpec
+
TargetSpec
```

Example:

```yaml
dataset_id: wafer_yield

entity_key: wafer_id
time_key: timestamp

target:
  name: yield
  type: regression
  source_column: yield

features:

  chamber_temperature:
    transforms:
      - current

  chamber_pressure:
    transforms:
      - current

  deposition_time:
    transforms:
      - current

  tool_id:
    transforms:
      - categorical
```

The architecture should recognize that this is a regression target and eventually allow regression-compatible models without redesigning dataset handling.

Actual regression model implementation is optional for v0.1.

---

# 28. Future CRM Example

The architecture must also accommodate non-time-series datasets.

Example:

```yaml
dataset_id: crm_conversion

entity_key: account_id

target:
  name: converted
  type: binary
  source_column: converted

features:

  industry:
    transforms:
      - categorical

  employee_count:
    transforms:
      - current

  engagement_score:
    transforms:
      - current

  meeting_count:
    transforms:
      - current
```

This should not require a `time_key` unless the analysis requires one.

---

# 29. Testing Strategy

## Dataset Tests

Test:

```text
DatasetSpec parsing
schema validation
missing required columns
type validation
CSV ingestion
Parquet ingestion
dataset registration
profiling
```

## Feature Tests

Test every generic transform independently:

```text
current
mean
min
max
std
delta
slope
categorical encoding
```

For temporal features explicitly verify:

> A feature calculated at time T never uses observations after T.

## Target Tests

Test:

```text
binary targets
future-event target
prediction horizon boundaries
entity isolation
```

A failure on Machine B must never label Machine A.

## ML Tests

Test:

```text
training
evaluation
model persistence
model reload
prediction
training/inference feature parity
```

## API Tests

Test:

```text
dataset registration
profile
slice
training
metrics
prediction
explanation
```

## MCP Tests

Test:

```text
tool discovery
schemas
dataset profiling
slice invocation
prediction
explanation
invalid inputs
```

---

# 30. Reference End-to-End Test

Implement one complete acceptance pipeline:

```text
generate synthetic predictive-maintenance dataset
            ↓
register DatasetSpec
            ↓
ingest CSV
            ↓
convert/persist Parquet
            ↓
profile dataset
            ↓
build temporal features
            ↓
construct future-event labels
            ↓
chronological split
            ↓
train Logistic Regression
            ↓
train XGBoost
            ↓
evaluate both
            ↓
persist XGBoost model
            ↓
reload model
            ↓
select machine + timestamp
            ↓
construct inference features
            ↓
predict failure probability
            ↓
explain prediction
            ↓
return REST result
            ↓
repeat prediction through MCP
```

---

# 31. Synthetic Data Acceptance

The synthetic generator must create an intentionally learnable but imperfect relationship.

The test should verify that:

```text
models perform meaningfully above random
```

Do not require unrealistically high accuracy.

The purpose is to prove:

```text
data
→ feature engineering
→ learning
→ evaluation
→ persistence
→ inference
```

works correctly.

---

# 32. Critical Architectural Rules

1. **Predictive maintenance is a reference implementation, not the core architecture.**

2. No domain-specific sensor names in generic ML services.

3. Dataset semantics come from `DatasetSpec`.

4. Feature construction comes from `FeatureSpec`.

5. Target construction comes from `TargetSpec`.

6. Training and inference use the exact same Feature Engine.

7. Training is explicit; inference never retrains.

8. REST, MCP and UI call the same application services.

9. No LLM is required for model training or inference.

10. MCP exposes controlled capabilities, never arbitrary Python execution.

11. Analytical operations remain independent from predictive models.

12. Prevent temporal/data leakage by design and tests.

13. Prefer configuration and registries over domain-specific conditional logic.

14. Keep infrastructure lightweight for the prototype.

---

# 33. Documentation

Generate:

```text
README.md
TECHNICAL_ARCHITECTURE.md
docs/architecture.md
```

`TECHNICAL_ARCHITECTURE.md` should follow the concise reference style of AINA Veris.

README should cover:

```text
what Anistroph is
architecture
dataset abstraction
predictive-maintenance reference implementation
installation
synthetic data generation
dataset registration
training
evaluation
inference
REST
MCP
tests
```

---

# 34. v0.1 Acceptance Criteria

A fresh checkout must be able to:

1. Generate synthetic predictive-maintenance sensor data.
2. Register the dataset through a DatasetSpec.
3. Ingest CSV and persist analytical data as Parquet.
4. Profile the dataset.
5. Build features according to FeatureSpec.
6. Generate a 24-hour future-failure target.
7. Train Logistic Regression and XGBoost.
8. Evaluate both against chronologically held-out data.
9. Persist and reload a trained model.
10. Predict failure probability for a machine at a specified timestamp.
11. Explain the strongest contributing features.
12. Slice and aggregate the dataset independently of ML.
13. Perform prediction through REST.
14. Perform the same prediction through MCP.
15. Pass the complete pytest suite.

Most importantly:

> **Adding a fundamentally different structured dataset must not require rewriting Anistroph's data, feature, training, evaluation, or inference architecture.**

Predictive maintenance proves the architecture; it does not define it.