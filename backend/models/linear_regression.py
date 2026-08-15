"""Linear Regression predictor — interpretable regression baseline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from backend.ml.base import Predictor


class LinearRegressionPredictor(Predictor):
    """Ridge regression with standard scaling — simple baseline."""

    model_type = "linear_regression"
    task_type = "regression"

    def __init__(self, **params: object) -> None:
        self.params = dict(
            alpha=params.get("alpha", 1.0),
            random_state=params.get("random_state", 42),
        )
        self._scaler: StandardScaler | None = None
        self._model: Ridge | None = None
        self._feature_names: list[str] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._model = Ridge(**self.params)
        self._model.fit(X_scaled, y)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        assert self._scaler is not None, "model not fitted"
        return self._scaler.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scale(X))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"scaler": self._scaler, "model": self._model, "params": self.params, "model_type": self.model_type},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LinearRegressionPredictor":
        data = joblib.load(path)
        obj = cls(**data["params"])
        obj._scaler = data["scaler"]
        obj._model = data["model"]
        return obj

    def feature_importance(self) -> dict[str, float] | None:
        if self._model is None or self._feature_names is None:
            return None
        coefs = self._model.coef_
        return {name: float(coef) for name, coef in zip(self._feature_names, coefs)}
