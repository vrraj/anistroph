# Anistroph — Architecture

## Overview

Anistroph is a domain-agnostic predictive analytics platform. The core ML
pipeline contains no domain-specific assumptions. Dataset semantics are
expressed through configuration (DatasetSpec, FeatureSpec, TargetSpec).

## Key Design Decisions

1. **Configuration over code.** Adding a dataset requires configuration, not
   pipeline changes.
2. **Single Feature Engine.** Training and inference share one engine. No
   duplicated feature logic.
3. **Explicit training.** Inference never retrains.
4. **Unified services.** REST, MCP, and UI call the same `AnistrophServices`.
5. **Leakage prevention by design.** Rolling windows are trailing and
   per-entity. Targets look forward; features do not. Splits are chronological.
6. **Lightweight infrastructure.** Parquet + JSON registries. No PostgreSQL,
   Spark, Kafka, MLflow, or Kubernetes for v0.1.

## Component Map

```
backend/
├── main.py                  FastAPI app
├── services.py              AnistrophServices (single entry point)
├── api/                     REST routers
│   ├── datasets.py
│   ├── analysis.py
│   ├── models.py
│   └── predictions.py
├── datasets/
│   ├── spec.py              DatasetSpec (Pydantic)
│   ├── config.py            Config bundle loader
│   ├── loader.py            CSV/Parquet ingestion → Parquet
│   ├── validation.py        Schema validation
│   ├── profiling.py         Generic profiling
│   └── registry.py          Dataset metadata registry
├── features/
│   ├── spec.py              FeatureSpec
│   ├── engine.py            FeatureEngine (train + inference)
│   ├── numeric.py           Numeric transform registry
│   ├── categorical.py       One-hot encoding
│   ├── temporal.py          Calendar/elapsed features
│   └── rolling.py           Rolling window aggregations
├── targets/
│   ├── spec.py              TargetSpec
│   ├── engine.py            TargetEngine dispatcher
│   ├── binary.py            Binary target
│   ├── regression.py        Regression target
│   └── horizon.py           Future-event horizon target
├── ml/
│   ├── base.py              Predictor abstract contract
│   ├── training.py          train_model() pipeline
│   ├── evaluation.py        Binary classification metrics
│   ├── inference.py         predict() with feature reconstruction
│   ├── explain.py           Feature importance explanation
│   └── registry.py          Model artifact registry
├── models/
│   ├── logistic.py          LogisticRegressionPredictor
│   └── xgboost.py           XGBoostPredictor
├── analysis/
│   ├── slice.py             Slice/aggregate/compare
│   ├── aggregate.py         Re-export
│   └── compare.py           Re-export
├── integrations/mcp/
│   ├── tools.py             MCP tool definitions
│   └── server.py            MCP stdio server
└── schemas/
    └── api.py               REST request/response schemas
```

## Data Formats

- **Analytical data:** Parquet
- **Dataset config:** YAML (`datasets/<id>/dataset.yaml`)
- **Registries:** JSON
- **Model artifacts:** joblib + JSON sidecar files
