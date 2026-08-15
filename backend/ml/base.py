"""Predictor — common abstraction for all models.

Both LogisticRegression and XGBoost predictors implement this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Predictor(ABC):
    """Abstract predictor contract."""

    model_type: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        """Fit the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities (positive class in column 1)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the model to a path."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Predictor":
        """Load a model from a path."""

    def feature_importance(self) -> dict[str, float] | None:
        """Return feature importance if available (feature_name -> score)."""
        return None
