"""SHAP explanations: top-5 signed feature contributions per cell and prediction date."""

import logging

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

log = logging.getLogger(__name__)


def shap_values(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """TreeExplainer SHAP values (log-odds units), shape (n_rows, n_features)."""
    return np.asarray(shap.TreeExplainer(model).shap_values(X))


def top_k_contributions(
    model: XGBClassifier, X: pd.DataFrame, keys: pd.DataFrame, k: int = 5
) -> pd.DataFrame:
    """Long table of the ``k`` largest |SHAP| features for each row of ``X``.

    ``keys`` holds ``cell_id`` and ``week_start`` aligned with ``X``.
    """
    sv = shap_values(model, X)
    order = np.argsort(-np.abs(sv), axis=1)[:, :k]
    rows = np.repeat(np.arange(len(X)), k)
    cols = order.ravel()
    out = pd.DataFrame({
        "cell_id": keys["cell_id"].to_numpy()[rows],
        "week_start": keys["week_start"].to_numpy()[rows],
        "rank": np.tile(np.arange(1, k + 1), len(X)),
        "feature": np.asarray(X.columns)[cols],
        "value": X.to_numpy()[rows, cols],
        "contribution": sv[rows, cols],
    })
    log.info("Computed SHAP top-%d for %d rows", k, len(X))
    return out
