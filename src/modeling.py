"""Reusable modeling components for the v2 (honest, calibrated) churn model.

Kept separate from the training script so both the trainer and the test suite
import the *same* preprocessing + estimator definitions — no drift between them.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from features import ENRICHED_NUMERIC, ENRICHED_CATEGORICAL

NUMERIC = ENRICHED_NUMERIC
CATEGORICAL = ENRICHED_CATEGORICAL
ALL_FEATURES = NUMERIC + CATEGORICAL


def make_preprocessor() -> ColumnTransformer:
    """Scale numerics, one-hot categoricals."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def base_pipeline(**params) -> Pipeline:
    """Preprocessor + HistGradientBoosting.

    NOTE: no class_weight='balanced' — that distorts probabilities, and the
    campaign ROI tool depends on calibrated probabilities. Imbalance is handled
    at the decision threshold (see economics.py), not by reweighting.
    """
    return Pipeline([
        ("pre", make_preprocessor()),
        ("clf", HistGradientBoostingClassifier(random_state=42, **params)),
    ])


def bootstrap_auc_ci(y_true, proba, n_boot=1000, alpha=0.05, seed=42):
    """Percentile bootstrap 95% CI for test ROC-AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], proba[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def slice_auc(y_true, proba, groups) -> dict:
    """ROC-AUC within each group value (skips tiny / single-class slices)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    g = pd.Series(np.asarray(groups)).reset_index(drop=True)
    out = {}
    for val in g.dropna().unique():
        m = (g == val).to_numpy()
        if m.sum() > 50 and len(np.unique(y_true[m])) == 2:
            out[str(val)] = round(float(roc_auc_score(y_true[m], proba[m])), 4)
    return out
