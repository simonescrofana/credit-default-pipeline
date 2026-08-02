"""Structural type contracts shared across model implementations.

Define the Estimator protocol so ml.training.trainer can type-hint its
model_builder parameter generically, accepting any object that implements
fit/predict_proba — whether it is a real scikit-learn estimator or a custom
wrapper like ml.models.mlp.MLPClassifier — without requiring either to
inherit from a common base class.

"""

from typing import Protocol

import numpy as np
import pandas as pd


class Estimator(Protocol):
    """Structural contract for any fit/predict_proba-compatible model."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Estimator":
        """Fit the model on the given features and target."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given features."""
        ...
