---
layout: default
title: "Guided Walkthrough | Anistroph"
description: "Step-by-step inference lifecycle — discover, understand, predict, explain, evaluate, and find error slices — across three reference datasets covering regression and classification."
---

# Anistroph — Guided Walkthrough

A step-by-step walkthrough of the full inference lifecycle across three reference datasets covering **regression** and **classification**, **temporal** and **non-temporal** prediction.

Each step is a prompt you can paste into Claude Desktop (or any MCP client connected to Anistroph). Run the prompts in order within a single conversation — Claude carries context forward, so later prompts can build on earlier results.

> **Prerequisites:** Anistroph installed and datasets registered (`make install`), and the Anistroph MCP server configured in Claude Desktop. See the [Setup & Usage Guide](setup-usage.md) for setup details.

> **AI Agent Analysis & Validation**
>
> Claude and AI agents can orchestrate Anistroph's prediction and analytical capabilities through MCP. Predictions, explanations, evaluations, and analyses are executed by Anistroph's shared services and can be independently reproduced through the Web UI or REST API when validation is required.

---

## The Inference Lifecycle

Every dataset in Anistroph follows the same seven-step lifecycle, regardless of domain or task type:

| Step | Purpose | MCP Tool |
|------|---------|----------|
| **1. Discover** | See what datasets and models exist | `anistroph_list_datasets`, `anistroph_list_models` |
| **2. Understand** | Profile the dataset and slice it by dimensions | `anistroph_profile_dataset`, `anistroph_slice_data`, `anistroph_compare_data` |
| **3. Inspect the input contract** | Learn what the model needs to predict | `anistroph_get_model_inputs` |
| **4. Predict** | Run inference — entity lookup or records | `anistroph_predict` |
| **5. Explain** | SHAP feature drivers for a prediction | `anistroph_explain_prediction` |
| **6. Evaluate** | Held-out metrics on the evaluation partition | `anistroph_evaluate_model` |
| **7. Find error slices** | Where does the model break? | `anistroph_find_evaluation_slices` |

The walkthrough below runs all seven steps on three datasets that together exercise every prediction mode Anistroph supports:

| Dataset | Domain | Target | Task | Prediction Mode |
|---------|--------|--------|------|-----------------|
| **Semiconductor Procurement — Demand** | Supply chain | `material_demand_next_4w` | Regression | Temporal entity lookup (rolling features) |
| **Semiconductor Yield** | Manufacturing | `wafer_yield` | Regression | Non-temporal entity lookup + records mode |
| **Predictive Maintenance — Failure** | Equipment health | `failure_within_horizon` | Classification | Temporal entity lookup |

---

## Dataset 1 — Semiconductor Procurement: Demand Forecasting

**Why this dataset first:** It demonstrates Anistroph's temporal prediction capability — the model uses rolling-window features (4-week, 8-week, 13-week means) that are reconstructed from entity history at prediction time. The model is static; the features are not. This is the case most ML platforms cannot handle without a custom feature pipeline.

- **Dataset:** `semiconductor_procurement_demand` (~100K weekly rows, 624 fab-material series)
- **Model:** `semiconductor_procurement_demand-xgboost_regressor-20260818111627`
- **Held-out R²:** 0.96 · MAE: 11.1 · RMSE: 17.8

### Step 1 — Discover

> What datasets and models are available in Anistroph?

Claude calls `anistroph_list_datasets` and `anistroph_list_models` in parallel. You'll see 13 datasets and 13 models across four domains.

### Step 2 — Understand

> Profile the semiconductor_procurement_demand dataset. What's the row count, how many fab-material series are there, and how does 4-week material demand vary by material category?

Claude profiles the dataset (`anistroph_profile_dataset`), then slices it (`anistroph_compare_data` with dimension=`material_category`, metric=`material_demand_next_4w`). You'll see which material categories (Photoresists, Process Gases, Wet Chemicals, etc.) drive the most demand.

### Step 3 — Inspect the input contract

> What inputs does the semiconductor_procurement_demand model need to make a prediction? Does it require a timestamp, and if so, how much inference history does it need?

Claude calls `anistroph_get_model_inputs`. The response tells you:

- `prediction_mode: "entity_lookup"` — records mode is not supported because the model uses rolling-window transforms
- `requires_timestamp: true` — you must provide an `as_of` date
- `inference_history_window: "13w"` — Anistroph will load 13 weeks of entity history to build the rolling features
- `entity_key: "series_id"` — the entity is a composite of `{fab_id}__{material_id}`

This is the key concept: the caller does not need to know about the rolling features. The model declares its own input contract.

### Step 4 — Predict

> Predict the 4-week material demand for series FAB_A__MAT_0001 as of 2025-06-09. Then show me the actual demand for that period from the dataset so we can compare.

Claude calls `anistroph_predict` with `entity_id=FAB_A__MAT_0001` and `timestamp=2025-06-09`. Anistroph loads 13 weeks of history for this series up to June 9th, computes the 4w/8w/13w rolling means, applies the trained XGBoost model, and returns the forecast. Claude then calls `anistroph_sample_rows` to fetch the actual `material_demand_next_4w` value (246.6) for that week so you can compare prediction vs. actual.

### Step 5 — Explain

> Explain the prediction you just made for FAB_A__MAT_0001. Which features are driving the demand forecast up or down?

Claude calls `anistroph_explain_prediction`. The SHAP response shows the top positive and negative drivers — typically recent consumption trends (rolling means), planned wafer starts, and inventory level. Categorical features (fab, material category, supplier) appear as grouped contributions rather than separate one-hot columns.

### Step 6 — Evaluate

> Evaluate the semiconductor_procurement_demand model on its held-out evaluation set. What are the R², MAE, and RMSE?

Claude calls `anistroph_evaluate_model`. The response includes aggregate metrics (R²=0.96, MAE=11.1, RMSE=17.8) and a sample of prediction-vs-actual rows from the evaluation partition — the most recent 20% of weeks that were never used during training.

### Step 7 — Find error slices

> Find the slices where the demand forecast error is worst. Which fab, material category, or supplier combinations does the model struggle on?

Claude calls `anistroph_find_evaluation_slices`. The response ranks 1/2/3-dimensional combinations (e.g. `fab_id × material_category × supplier_id`) by how much the absolute prediction error deviates from the overall baseline. This tells you not just "the model is good overall" but "here's exactly where it's weak."

---

## Dataset 2 — Semiconductor Yield: Wafer Yield Prediction

**Why this dataset next:** It demonstrates the non-temporal case — single-row entity lookup with no rolling features. It also shows records mode (sending raw feature values instead of an entity ID) and the staged-prediction pattern (same target, progressively more features).

- **Dataset:** `semiconductor_yield` (50,000 wafers)
- **Model:** `wafer-yield-xgboost`
- **Held-out R²:** 0.81 · MAE: 0.0065

### Step 1 — Discover

> What models are available for semiconductor wafer yield prediction?

Claude calls `anistroph_list_models` and filters to models whose `target_name` is `wafer_yield`. You'll see the main model plus four staged models (A through D).

### Step 2 — Understand

> Profile the semiconductor_yield dataset. What's the row count, and how does wafer_yield vary across etch_tool and etch_chamber?

Claude profiles the dataset, then slices it by `etch_tool` × `etch_chamber` over `wafer_yield` (`anistroph_slice_data`). You'll see which tool/chamber combinations produce the lowest yields.

### Step 3 — Inspect the input contract

> What inputs does the wafer-yield-xgboost model need? Can I send raw records, or do I need to look up a wafer?

Claude calls `anistroph_get_model_inputs`. The response shows:

- `prediction_mode: "entity_lookup_or_records"` — both modes work (no rolling transforms)
- `requires_timestamp: false` — no `as_of` date needed
- `entity_key: "wafer_id"`
- The full list of required source columns (product, fab, etch settings, deposition settings, exposure dose, etc.)

Because this is a non-temporal model with only `current` and `categorical` transforms, you can either look up a wafer by ID or send a raw record with all the feature values.

### Step 4 — Predict (entity lookup)

> Show me the row for wafer_id WAFER_015000 in the semiconductor_yield dataset, then predict its wafer_yield using the wafer-yield-xgboost model.

Claude calls `anistroph_sample_rows` (filter `wafer_id=WAFER_015000`) to show the actual row, then `anistroph_predict` with `entity_id=WAFER_015000`. The response includes both the predicted and actual yield.

### Step 4b — Predict (records mode)

> Generate a plausible "worst-case" wafer record — pick etch_tool, chamber, recipe, and process settings that you'd expect to produce low yield — and run it through the wafer-yield-xgboost model. Then generate a "best-case" record and predict that too. Compare the two.

Claude uses the input schema from Step 3, composes two records using domain reasoning (e.g. ETCH_02 + CH_B + high maintenance age + low exposure dose for the worst case), and calls `anistroph_predict` with `records=[worst, best]`. This demonstrates records mode — the LLM is the test harness, fabricating realistic inputs and predicting on them without any feature pipeline code.

### Step 5 — Explain

> Explain the prediction you just made for WAFER_015000. Which features pushed the yield up, and which pushed it down?

Claude calls `anistroph_explain_prediction`. The SHAP response shows grouped one-hot drivers (e.g. `etch_tool = ETCH_02` as a single contribution) plus the continuous feature drivers (etch temperature, exposure dose, maintenance age).

### Step 6 — Evaluate

> Evaluate the wafer-yield-xgboost model on its held-out evaluation set.

Claude calls `anistroph_evaluate_model`. The response shows R²=0.81, MAE=0.0065, plus a prediction-vs-actual sample from the 10,000 held-out wafers.

### Step 7 — Find error slices

> Find the slices where the wafer yield prediction error is worst — which etch_tool, chamber, or product combinations does the model struggle on?

Claude calls `anistroph_find_evaluation_slices`. The response ranks combinations like `etch_tool × etch_chamber × product_id` by error deviation from baseline.

### Optional — Staged prediction

> I have four staged models for wafer yield — stage A (before etch), B (after etch), C (after deposition), D (before test). For wafer WAFER_015000, predict yield with each stage model and show how the prediction sharpens as more process data becomes available.

Claude calls `anistroph_predict` four times on the staged model IDs. The predictions converge toward the actual as more features are added, demonstrating the staged-prediction pattern: the same wafer is predicted at four points in the production line, each with only the features knowable at that point.

---

## Dataset 3 — Predictive Maintenance: Failure Within Horizon

**Why this dataset last:** It demonstrates the classification case — the model returns a probability, not a quantity. It's also temporal (sensor history with rolling features), so prediction requires an `as_of` timestamp. The evaluation metrics and error-slice discovery use classification-specific measures (ROC-AUC, F1, log loss) instead of regression measures (MAE, R²).

- **Dataset:** `predictive_maintenance` (864,000 sensor readings, 50 machines)
- **Model:** `predictive_maintenance-xgboost-20260817002741`
- **Held-out ROC-AUC:** 0.85 · F1: 0.61 · Precision: 0.48 · Recall: 0.81

### Step 1 — Discover

> What models are available for predictive maintenance?

Claude calls `anistroph_list_models` and filters to maintenance-related models. You'll see the failure model (classification), the RUL model (regression), and the maintenance-required model (classification).

### Step 2 — Understand

> Profile the predictive_maintenance dataset. What's the row count, how many machines are there, and what's the overall failure rate? How does failure rate vary by machine_type?

Claude profiles the dataset, then compares failure rate across `machine_type` (`anistroph_compare_data`). The failure rate is very low overall (~0.08%), so slicing by machine type helps see which types are riskier.

### Step 3 — Inspect the input contract

> What inputs does the predictive_maintenance failure model need? Does it require a timestamp?

Claude calls `anistroph_get_model_inputs`. The response shows:

- `prediction_mode: "entity_lookup"` — rolling-window transforms are used, so records mode is not supported
- `requires_timestamp: true` — you must provide an `as_of` timestamp
- `entity_key: "machine_id"`
- `inference_history_window` — the lookback required for the rolling sensor features

### Step 4 — Predict

> Predict the probability of failure within the horizon for machine TOOL_010 as of 2026-06-02T05:30:00. Then show me what actually happened to TOOL_010 in the hour after that timestamp.

Claude calls `anistroph_predict` with `entity_id=TOOL_010` and `timestamp=2026-06-02T05:30:00`. Anistroph loads the machine's sensor history up to that point, computes rolling features, and returns a failure probability. Claude then calls `anistroph_sample_rows` to show the readings after that timestamp — revealing that TOOL_010 failed at 05:35 (five minutes later) with a POWER failure mode. This is a compelling before-the-fact prediction: the model was asked to assess risk 5 minutes before the actual failure.

### Step 5 — Explain

> Explain the failure prediction you just made for TOOL_010. Which sensor readings or conditions are driving the risk score up?

Claude calls `anistroph_explain_prediction`. The SHAP response shows which sensor features (temperature, vibration, pressure, maintenance age) and their rolling trends are pushing the failure probability up or down.

### Step 6 — Evaluate

> Evaluate the predictive_maintenance failure model on its held-out evaluation set. What are the ROC-AUC, F1, precision, and recall?

Claude calls `anistroph_evaluate_model`. The response includes classification metrics: ROC-AUC=0.85, F1=0.61, precision=0.48, recall=0.81, plus the decision threshold (0.223) chosen to maximize F1. You'll also see a sample of prediction-vs-actual rows.

### Step 7 — Find error slices

> Find the slices where the failure model's prediction error is worst. Which machine_type or failure_mode combinations does the model struggle on? Use log loss as the error metric.

Claude calls `anistroph_find_evaluation_slices` with `metric=log_loss`. The response ranks combinations like `machine_type × failure_mode` by how much the per-row log loss deviates from the overall baseline. For classification, this identifies populations where the model is overconfident or underconfident — not just where it's wrong, but where its probability calibration is off.

---

## What This Walkthrough Demonstrates

| Capability | Where it appears |
|------------|-----------------|
| **Discovery** — list datasets and models via MCP | Step 1 of every dataset |
| **Multidimensional analysis** — slice and compare data by categorical dimensions | Step 2 of every dataset |
| **Self-describing input contract** — model declares what it needs, including `as_of` and history window | Step 3 of every dataset |
| **Temporal prediction** — rolling features reconstructed from entity history at prediction time | Datasets 1 and 3 |
| **Non-temporal prediction** — single-row entity lookup | Dataset 2 |
| **Records mode** — send raw feature values instead of an entity ID | Dataset 2, Step 4b |
| **Staged prediction** — same target, progressively more features | Dataset 2, optional step |
| **SHAP explainability** with one-hot feature grouping | Step 5 of every dataset |
| **Regression evaluation** — R², MAE, RMSE, MAPE | Datasets 1 and 2 |
| **Classification evaluation** — ROC-AUC, F1, precision, recall | Dataset 3 |
| **Error slice discovery** — where does the model break, by dimension combination | Step 7 of every dataset |
| **Multi-domain** — same lifecycle across supply chain, manufacturing, and equipment health | All three datasets |

The same seven MCP tools, the same seven-step lifecycle, and the same Claude conversation — across three domains, two task types, and both temporal and non-temporal prediction. No domain-specific code was written for any of these steps. The datasets declare their schemas and feature transforms in YAML; the runtime handles the rest.

---

## Next Steps

- **Add your own dataset:** See [Adding a Dataset](../README.md#adding-a-dataset) and the [Dataset Configuration reference](setup-usage.md#configure-your-own-dataset).
- **Full YAML reference and operations guide:** [Setup & Usage Guide](setup-usage.md).
- **Architecture details:** [Technical Architecture](technical-architecture.md).
- **Temporal prediction deep dive:** [Temporal Prediction, History, and Retraining](setup-usage.md#temporal-prediction).
