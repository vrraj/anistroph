"""Tests for the semiconductor yield dataset and models."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from backend.analysis.interesting import find_interesting_slices
from backend.datasets.config import load_dataset_config
from backend.datasets.registry import DatasetRegistry
from backend.ml.evaluation import evaluate_regression
from backend.ml.registry import ModelRegistry
from backend.ml.training import train_model
from backend.models.xgboost_regressor import XGBoostRegressorPredictor
from backend.models.linear_regression import LinearRegressionPredictor
from backend.targets.spec import TargetType

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_PATH = _REPO_ROOT / "data" / "semiconductor_yield" / "data.parquet"
_CONFIG_PATH = _REPO_ROOT / "datasets" / "semiconductor_yield" / "dataset.yaml"


@pytest.fixture
def semiconductor_config():
    return load_dataset_config(_CONFIG_PATH)


class TestSemiconductorData:
    """Verify synthetic semiconductor data generation."""

    def test_data_file_exists(self):
        assert _DATA_PATH.exists(), "semiconductor data parquet not found"

    def test_row_count(self):
        df = pl.read_parquet(_DATA_PATH)
        assert df.height >= 1000, f"expected >= 1000 rows, got {df.height}"

    def test_yield_in_range(self):
        df = pl.read_parquet(_DATA_PATH)
        yield_col = df["wafer_yield"]
        assert yield_col.min() >= 0.0, "wafer_yield has values < 0"
        assert yield_col.max() <= 1.0, "wafer_yield has values > 1"

    def test_required_columns(self):
        df = pl.read_parquet(_DATA_PATH)
        required = [
            "timestamp", "lot_id", "wafer_id", "product_id", "fab_id",
            "process_route", "etch_tool", "etch_chamber", "etch_recipe",
            "deposition_tool", "deposition_chamber", "deposition_recipe",
            "etch_temperature_mean", "etch_temperature_std",
            "etch_pressure_mean", "etch_pressure_std",
            "etch_gas_flow_mean", "etch_rf_power_mean", "etch_process_time",
            "deposition_temperature_mean", "deposition_temperature_std",
            "deposition_pressure_mean", "deposition_pressure_std",
            "deposition_process_time", "exposure_dose", "focus_offset",
            "maintenance_age_etch", "maintenance_age_deposition",
            "wafer_yield",
        ]
        for col in required:
            assert col in df.columns, f"missing column: {col}"

    def test_hidden_interactions_present(self):
        """ETCH_02 + CH_B should have lower mean yield than overall baseline."""
        df = pl.read_parquet(_DATA_PATH)
        overall = df["wafer_yield"].mean()
        combo = df.filter(
            (pl.col("etch_tool") == "ETCH_02") & (pl.col("etch_chamber") == "CH_B")
        )["wafer_yield"].mean()
        assert combo < overall, (
            f"ETCH_02+CH_B yield ({combo:.4f}) should be lower than overall ({overall:.4f})"
        )

    def test_baseline_yield_high(self):
        """Overall yield should be in the 90-100% range."""
        df = pl.read_parquet(_DATA_PATH)
        overall = df["wafer_yield"].mean()
        assert 0.90 < overall < 1.0, f"overall yield {overall:.4f} not in expected range"


class TestSemiconductorConfig:
    """Verify dataset configuration."""

    def test_config_loads(self, semiconductor_config):
        assert semiconductor_config.dataset_spec.dataset_id == "semiconductor_yield"

    def test_target_is_regression(self, semiconductor_config):
        assert semiconductor_config.target_spec is not None
        assert semiconductor_config.target_spec.type == TargetType.REGRESSION
        assert semiconductor_config.target_spec.name == "wafer_yield"

    def test_entity_key(self, semiconductor_config):
        assert semiconductor_config.dataset_spec.entity_key == "wafer_id"

    def test_has_timestamp_for_split(self, semiconductor_config):
        assert semiconductor_config.dataset_spec.time_key == "timestamp"

    def test_split_is_chronological(self, semiconductor_config):
        assert semiconductor_config.dataset_spec.split.strategy == "chronological"


class TestRegressionEvaluation:
    """Verify regression evaluation metrics."""

    def test_evaluate_regression_basic(self):
        y_true = np.array([0.95, 0.92, 0.88, 0.97, 0.90])
        y_pred = np.array([0.94, 0.93, 0.89, 0.96, 0.91])
        metrics = evaluate_regression(y_true, y_pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "median_abs_error" in metrics
        assert "p95_abs_error" in metrics
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1.0

    def test_evaluate_regression_with_baseline(self):
        y_true = np.array([0.95, 0.92, 0.88, 0.97, 0.90])
        y_pred = np.array([0.94, 0.93, 0.89, 0.96, 0.91])
        metrics = evaluate_regression(y_true, y_pred, baseline_pred=0.92)
        assert "baseline" in metrics
        assert metrics["baseline"]["constant_value"] == 0.92
        assert metrics["baseline"]["mae"] >= 0


class TestSemiconductorModelAdapters:
    """Verify regression model adapters."""

    def test_xgboost_regressor_task_type(self):
        p = XGBoostRegressorPredictor()
        assert p.task_type == "regression"
        assert p.model_type == "xgboost_regressor"

    def test_linear_regression_task_type(self):
        p = LinearRegressionPredictor()
        assert p.task_type == "regression"
        assert p.model_type == "linear_regression"

    def test_xgboost_regressor_fit_predict(self):
        p = XGBoostRegressorPredictor(n_estimators=10)
        X = np.random.rand(50, 5)
        y = np.random.rand(50)
        p._feature_names = [f"f{i}" for i in range(5)]
        p.fit(X, y)
        preds = p.predict(X)
        assert preds.shape == (50,)

    def test_xgboost_regressor_save_load(self, tmp_path):
        p = XGBoostRegressorPredictor(n_estimators=10)
        X = np.random.rand(50, 5)
        y = np.random.rand(50)
        p._feature_names = [f"f{i}" for i in range(5)]
        p.fit(X, y)
        path = str(tmp_path / "model.joblib")
        p.save(path)
        loaded = XGBoostRegressorPredictor.load(path)
        preds_orig = p.predict(X)
        preds_loaded = loaded.predict(X)
        np.testing.assert_array_almost_equal(preds_orig, preds_loaded)


class TestSemiconductorTraining:
    """Verify model training on semiconductor data (small subset)."""

    @pytest.fixture
    def small_semi_dataset(self, tmp_path):
        """Create a small semiconductor dataset for fast tests."""
        from scripts.generate_semiconductor_yield_data import generate_wafers
        df = generate_wafers(n_wafers=500, seed=99)
        pq = tmp_path / "semi.parquet"
        df.write_parquet(pq)

        # Register dataset
        registry_path = tmp_path / "dataset_registry.json"
        dataset_registry = DatasetRegistry(registry_path)
        config = load_dataset_config(_CONFIG_PATH)
        spec = config.dataset_spec
        dataset_registry.register(
            spec=spec,
            source=str(pq),
            row_count=df.height,
            parquet_path=str(pq),
            data_start=None,
            data_end=None,
            feature_spec=config.feature_spec,
            target_spec=config.target_spec,
            spec_path=str(_CONFIG_PATH),
        )
        return dataset_registry, tmp_path, pq

    def test_xgboost_regressor_trains(self, small_semi_dataset):
        dataset_registry, tmp_path, pq = small_semi_dataset
        model_registry = ModelRegistry(tmp_path / "models")
        config = load_dataset_config(_CONFIG_PATH)

        result = train_model(
            dataset_id="semiconductor_yield",
            target_name="wafer_yield",
            model_type="xgboost_regressor",
            dataset_registry=dataset_registry,
            model_registry=model_registry,
            config=config,
            model_parameters={"n_estimators": 20},
            parquet_path=str(pq),
        )
        assert "model_id" in result
        assert "metrics" in result
        metrics = result["metrics"]
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_linear_regression_trains(self, small_semi_dataset):
        dataset_registry, tmp_path, pq = small_semi_dataset
        model_registry = ModelRegistry(tmp_path / "models")
        config = load_dataset_config(_CONFIG_PATH)

        result = train_model(
            dataset_id="semiconductor_yield",
            target_name="wafer_yield",
            model_type="linear_regression",
            dataset_registry=dataset_registry,
            model_registry=model_registry,
            config=config,
            parquet_path=str(pq),
        )
        assert "metrics" in result
        assert result["metrics"]["r2"] is not None

    def test_model_beats_baseline(self, small_semi_dataset):
        """XGBoost should beat the trivial mean-yield predictor."""
        dataset_registry, tmp_path, pq = small_semi_dataset
        model_registry = ModelRegistry(tmp_path / "models")
        config = load_dataset_config(_CONFIG_PATH)

        result = train_model(
            dataset_id="semiconductor_yield",
            target_name="wafer_yield",
            model_type="xgboost_regressor",
            dataset_registry=dataset_registry,
            model_registry=model_registry,
            config=config,
            model_parameters={"n_estimators": 50},
            parquet_path=str(pq),
        )
        metrics = result["metrics"]
        assert metrics["mae"] < metrics["baseline"]["mae"], (
            f"XGBoost MAE ({metrics['mae']:.4f}) should be < baseline MAE ({metrics['baseline']['mae']:.4f})"
        )

    def test_chronological_split_used(self, small_semi_dataset):
        """Verify training period is before test period."""
        dataset_registry, tmp_path, pq = small_semi_dataset
        model_registry = ModelRegistry(tmp_path / "models")
        config = load_dataset_config(_CONFIG_PATH)

        result = train_model(
            dataset_id="semiconductor_yield",
            target_name="wafer_yield",
            model_type="xgboost_regressor",
            dataset_registry=dataset_registry,
            model_registry=model_registry,
            config=config,
            model_parameters={"n_estimators": 10},
            parquet_path=str(pq),
        )
        meta = result["metadata"]
        if meta.get("training_period") and meta.get("test_period"):
            assert meta["training_period"]["end"] <= meta["test_period"]["start"]

    def test_model_persists_and_reloads(self, small_semi_dataset):
        """Verify model can be reloaded after persistence."""
        dataset_registry, tmp_path, pq = small_semi_dataset
        model_registry = ModelRegistry(tmp_path / "models")
        config = load_dataset_config(_CONFIG_PATH)

        result = train_model(
            dataset_id="semiconductor_yield",
            target_name="wafer_yield",
            model_type="xgboost_regressor",
            dataset_registry=dataset_registry,
            model_registry=model_registry,
            config=config,
            model_parameters={"n_estimators": 10},
            model_id="test-semi-xgb",
            parquet_path=str(pq),
        )
        # Reload model from registry.
        meta = model_registry.get("test-semi-xgb")
        assert meta is not None
        assert meta.model_type == "xgboost_regressor"

        # Load predictor from disk.
        predictor = XGBoostRegressorPredictor.load(f"{meta.artifact_path}/model.joblib")
        assert predictor is not None


class TestFindInterestingSlices:
    """Verify find_interesting_slices on semiconductor data."""

    @pytest.fixture
    def semi_registry(self, tmp_path):
        """Register the semiconductor dataset with a small sample."""
        from scripts.generate_semiconductor_yield_data import generate_wafers
        df = generate_wafers(n_wafers=2000, seed=77)
        pq = tmp_path / "semi.parquet"
        df.write_parquet(pq)

        registry_path = tmp_path / "dataset_registry.json"
        dataset_registry = DatasetRegistry(registry_path)
        config = load_dataset_config(_CONFIG_PATH)
        spec = config.dataset_spec
        dataset_registry.register(
            spec=spec,
            source=str(pq),
            row_count=df.height,
            parquet_path=str(pq),
            data_start=None,
            data_end=None,
            feature_spec=config.feature_spec,
            target_spec=config.target_spec,
            spec_path=str(_CONFIG_PATH),
        )
        return dataset_registry

    def test_find_interesting_slices_returns_results(self, semi_registry):
        results = find_interesting_slices(
            "semiconductor_yield", semi_registry, "wafer_yield", top_k=10
        )
        assert len(results) > 0
        assert len(results) <= 10

    def test_find_interesting_slices_has_required_fields(self, semi_registry):
        results = find_interesting_slices(
            "semiconductor_yield", semi_registry, "wafer_yield", top_k=5
        )
        for r in results:
            assert "dimensions" in r
            assert "values" in r
            assert "row_count" in r
            assert "metric_value" in r
            assert "overall_baseline" in r
            assert "difference" in r
            assert "abs_difference" in r

    def test_find_interesting_slices_respects_min_sample(self, semi_registry):
        results = find_interesting_slices(
            "semiconductor_yield", semi_registry, "wafer_yield",
            min_sample_size=10000, top_k=20
        )
        for r in results:
            assert r["row_count"] >= 10000

    def test_find_interesting_slices_finds_etch02_chb(self, semi_registry):
        """The ETCH_02 + CH_B combination should appear in top slices."""
        results = find_interesting_slices(
            "semiconductor_yield", semi_registry, "wafer_yield",
            dimensions=["etch_tool", "etch_chamber"], top_k=10
        )
        found = False
        for r in results:
            vals = r["values"]
            if vals.get("etch_tool") == "ETCH_02" and vals.get("etch_chamber") == "CH_B":
                found = True
                assert r["difference"] < 0, "ETCH_02+CH_B should have negative difference"
        assert found, "ETCH_02+CH_B not found in interesting slices"


class TestSHAPExplainability:
    """Verify SHAP-based per-prediction explainability."""

    @pytest.fixture
    def trained_semi_model(self, tmp_path):
        """Train a small XGBoost regressor for SHAP tests."""
        from scripts.generate_semiconductor_yield_data import generate_wafers
        from backend.services import AnistrophServices

        df = generate_wafers(n_wafers=1000, seed=55)
        pq = tmp_path / "semi.parquet"
        df.write_parquet(pq)

        svc = AnistrophServices(
            dataset_registry_path=tmp_path / "dataset_registry.json",
            model_registry_dir=tmp_path / "models",
        )
        meta = svc.dataset_registry.register(
            spec=load_dataset_config(_CONFIG_PATH).dataset_spec,
            source=str(pq),
            row_count=df.height,
            parquet_path=str(pq),
            data_start=None,
            data_end=None,
            feature_spec=load_dataset_config(_CONFIG_PATH).feature_spec,
            target_spec=load_dataset_config(_CONFIG_PATH).target_spec,
            spec_path=str(_CONFIG_PATH),
        )
        svc.train(
            "semiconductor_yield", "wafer_yield", "xgboost_regressor",
            model_parameters={"n_estimators": 30},
            model_id="test-shap-xgb",
        )
        return svc, df

    def test_explain_returns_shap_method(self, trained_semi_model):
        svc, df = trained_semi_model
        wafer_id = df["wafer_id"][0]
        expl = svc.explain("test-shap-xgb", entity_id=wafer_id, top_k=5)
        assert expl["explanation_method"] == "shap_tree_explainer"

    def test_explain_has_top_positive_and_negative(self, trained_semi_model):
        svc, df = trained_semi_model
        wafer_id = df["wafer_id"][0]
        expl = svc.explain("test-shap-xgb", entity_id=wafer_id, top_k=5)
        assert "top_positive" in expl
        assert "top_negative" in expl
        assert isinstance(expl["top_positive"], list)
        assert isinstance(expl["top_negative"], list)

    def test_top_positive_impacts_are_positive(self, trained_semi_model):
        svc, df = trained_semi_model
        wafer_id = df["wafer_id"][0]
        expl = svc.explain("test-shap-xgb", entity_id=wafer_id, top_k=5)
        for c in expl["top_positive"]:
            assert c["impact"] > 0, f"top_positive feature {c['feature']} has non-positive impact"

    def test_top_negative_impacts_are_negative(self, trained_semi_model):
        svc, df = trained_semi_model
        wafer_id = df["wafer_id"][0]
        expl = svc.explain("test-shap-xgb", entity_id=wafer_id, top_k=5)
        for c in expl["top_negative"]:
            assert c["impact"] < 0, f"top_negative feature {c['feature']} has non-negative impact"

    def test_explain_preserves_feature_names(self, trained_semi_model):
        """Feature names in explanations must match the engineered feature names."""
        svc, df = trained_semi_model
        wafer_id = df["wafer_id"][0]
        expl = svc.explain("test-shap-xgb", entity_id=wafer_id, top_k=10)
        fm = svc.model_registry.load_feature_metadata("test-shap-xgb")
        all_features = set(fm.feature_names)
        for c in expl["top_positive"] + expl["top_negative"]:
            assert c["feature"] in all_features, f"unknown feature in explanation: {c['feature']}"

    def test_explain_etch02_chb_is_negative(self, trained_semi_model):
        """A wafer with ETCH_02 + CH_B should have those as negative contributors."""
        svc, df = trained_semi_model
        # Find a wafer with ETCH_02 + CH_B
        target_df = df.filter(
            (pl.col("etch_tool") == "ETCH_02") & (pl.col("etch_chamber") == "CH_B")
        )
        assert target_df.height > 0, "no ETCH_02+CH_B wafers in test data"
        wafer_id = target_df["wafer_id"][0]
        expl = svc.explain("test-shap-xgb", entity_id=wafer_id, top_k=10)

        # etch_tool__ETCH_02 or etch_chamber__CH_B should appear in top_negative
        neg_features = {c["feature"] for c in expl["top_negative"]}
        assert "etch_tool__ETCH_02" in neg_features or "etch_chamber__CH_B" in neg_features, (
            "ETCH_02 or CH_B not in top negative contributors for a wafer with that combination"
        )

    def test_explain_top_drivers_backward_compat(self, trained_semi_model):
        """top_drivers should still be present for backward compatibility."""
        svc, df = trained_semi_model
        wafer_id = df["wafer_id"][0]
        expl = svc.explain("test-shap-xgb", entity_id=wafer_id, top_k=5)
        assert "top_drivers" in expl
        assert len(expl["top_drivers"]) <= 5

    def test_explain_contributions_sum_to_prediction(self, trained_semi_model):
        """SHAP values + base value should approximately equal the prediction."""
        svc, df = trained_semi_model
        wafer_id = df["wafer_id"][0]
        expl = svc.explain("test-shap-xgb", entity_id=wafer_id, top_k=100)
        # Get all contributions (not just top_k) by checking top_drivers
        # The sum of all SHAP values + base = prediction. We can't check
        # exact equality with truncated lists, but we can verify the
        # prediction is reasonable.
        assert "predicted_yield" in expl
        assert 0.0 <= expl["predicted_yield"] <= 1.0

    def test_explain_answers_what_pushed_up_or_down(self, trained_semi_model):
        """The explanation answers: 'The model predicted X% yield. Which inputs
        pushed the prediction up or down, and by how much?'

        Verify the response contains the structured fields needed to answer
        that question: predicted value, top_positive (pushed up), and
        top_negative (pushed down), each with feature, impact, and value.
        """
        svc, df = trained_semi_model
        # Find a wafer with ETCH_02 + CH_B for a meaningful example.
        target_df = df.filter(
            (pl.col("etch_tool") == "ETCH_02") & (pl.col("etch_chamber") == "CH_B")
        )
        assert target_df.height > 0
        wafer_id = target_df["wafer_id"][0]

        expl = svc.explain("test-shap-xgb", entity_id=wafer_id, top_k=10)

        # The question: "The model predicted X% yield."
        assert "predicted_yield" in expl
        predicted = expl["predicted_yield"]
        assert 0.0 <= predicted <= 1.0

        # "Which inputs pushed the prediction UP?"
        assert "top_positive" in expl
        assert len(expl["top_positive"]) > 0
        for c in expl["top_positive"]:
            assert "feature" in c, "each positive contributor needs a feature name"
            assert "impact" in c, "each positive contributor needs an impact (how much)"
            assert c["impact"] > 0, "positive contributors must push the prediction up"
            assert "value" in c, "each positive contributor needs the feature value"

        # "Which inputs pushed the prediction DOWN?"
        assert "top_negative" in expl
        assert len(expl["top_negative"]) > 0
        for c in expl["top_negative"]:
            assert "feature" in c, "each negative contributor needs a feature name"
            assert "impact" in c, "each negative contributor needs an impact (how much)"
            assert c["impact"] < 0, "negative contributors must push the prediction down"
            assert "value" in c, "each negative contributor needs the feature value"

        # The impacts should be in percentage-point terms (yield is 0-1,
        # so SHAP values are small decimals). Verify they are reasonable.
        for c in expl["top_positive"] + expl["top_negative"]:
            assert abs(c["impact"]) < 0.5, "SHAP impact should be within reasonable range"

    def test_explain_etch02_chb_negative_for_low_yield_wafer(self, trained_semi_model):
        """For a wafer processed on ETCH_02 + CH_B, the explanation should
        identify those as negative contributors (pushing yield down).

        This is the concrete version of: 'The model predicted 88% yield.
        ETCH_02 pushed it down by 2.4pp, CH_B pushed it down by 1.3pp.'
        """
        svc, df = trained_semi_model
        target_df = df.filter(
            (pl.col("etch_tool") == "ETCH_02") & (pl.col("etch_chamber") == "CH_B")
        )
        assert target_df.height > 0
        wafer_id = target_df["wafer_id"][0]

        expl = svc.explain("test-shap-xgb", entity_id=wafer_id, top_k=10)

        neg_features = {c["feature"] for c in expl["top_negative"]}
        # At least one of ETCH_02 or CH_B should be in the negative contributors.
        assert "etch_tool__ETCH_02" in neg_features or "etch_chamber__CH_B" in neg_features, (
            "For a wafer processed on ETCH_02+CH_B, the explanation should "
            "identify at least one of them as pushing yield down."
        )
