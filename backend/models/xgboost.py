"""XGBoost predictor — primary nonlinear model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from backend.ml.base import Predictor


class XGBoostPredictor(Predictor):
    """XGBoost binary classifier."""

    model_type = "xgboost"
    task_type = "classification"

    def __init__(self, **params: object) -> None:
        self.params = dict(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 3),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=params.get("random_state", 42),
            eval_metric=params.get("eval_metric", "aucpr"),
        )
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] | None = None
        self._shap_explainer = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        self._model = xgb.XGBClassifier(**self.params)
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self._model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "params": self.params, "model_type": self.model_type},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "XGBoostPredictor":
        data = joblib.load(path)
        obj = cls(**data["params"])
        obj._model = data["model"]
        return obj

    def feature_importance(self) -> dict[str, float] | None:
        if self._model is None or self._feature_names is None:
            return None
        scores = self._model.feature_importances_
        return {name: float(score) for name, score in zip(self._feature_names, scores)}

    def explain_instance(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values for a single instance (or batch).

        Uses TreeExplainer (TreeSHAP) for exact, model-derived contributions.
        Returns an array of shape (n_samples, n_features) with signed
        contributions: positive values increase the prediction, negative
        values decrease it.
        """
        if self._model is None:
            raise ValueError("model not fitted")
        if self._shap_explainer is None:
            import shap
            self._shap_explainer = shap.TreeExplainer(self._model)
        shap_values = self._shap_explainer.shap_values(X)
        # For binary classification, shap may return a list of two arrays
        # (one per class) or a 3D array. We want the positive-class (index 1).
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        return np.asarray(shap_values)
