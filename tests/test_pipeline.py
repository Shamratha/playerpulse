"""PlayerPulse test suite — guards the pipeline's correctness invariants.

Run:  .venv/Scripts/python.exe -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_clean, add_target, OUTLIER_ROUNDS_THRESHOLD  # noqa: E402
from simulate import simulate, NUMERIC_SIM  # noqa: E402
from features import (  # noqa: E402
    build_enriched_features, make_xy_enriched, ENRICHED_NUMERIC, ENRICHED_CATEGORICAL,
)
from modeling import ALL_FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def clean():
    return load_clean()


@pytest.fixture(scope="module")
def enriched(clean):
    return simulate(clean)


# ---- label definition -----------------------------------------------------
def test_label_is_inverse_of_day7(clean):
    assert (clean["churned"] == (~clean["retention_7"]).astype(int)).all()


def test_add_target_idempotent_values(clean):
    again = add_target(clean.drop(columns=["churned"]))
    assert (again["churned"] == clean["churned"]).all()


# ---- cleaning invariants --------------------------------------------------
def test_no_missing(clean):
    assert clean.isna().sum().sum() == 0


def test_outlier_removed(clean):
    assert clean["sum_gamerounds"].max() < OUTLIER_ROUNDS_THRESHOLD


def test_zero_round_players_kept(clean):
    assert (clean["sum_gamerounds"] == 0).sum() > 0


# ---- simulator: reproducibility + NO label leakage ------------------------
def test_simulate_reproducible(clean):
    a = simulate(clean, seed=42)[NUMERIC_SIM]
    b = simulate(clean, seed=42)[NUMERIC_SIM]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("col", ["friends_invited", "crashes", "n_purchases",
                                 "total_spend_usd", "sessions_count"])
def test_simulated_features_do_not_leak_label(enriched, col):
    # after removing label conditioning, correlation with churn must be modest
    # (it may be non-zero via shared dependence on engagement, but not extreme).
    corr = abs(np.corrcoef(enriched[col], enriched["churned"])[0, 1])
    assert corr < 0.35, f"{col} correlates {corr:.2f} with label — possible leakage"


# ---- feature contract (train == serve) ------------------------------------
def test_played_zero_dropped():
    assert "played_zero" not in ENRICHED_NUMERIC


def test_feature_matrix_columns_exact(enriched):
    X, y = make_xy_enriched(enriched)
    assert list(X.columns) == ENRICHED_NUMERIC + ENRICHED_CATEGORICAL == ALL_FEATURES
    assert set(y.unique()) <= {0, 1}


def test_enriched_features_present(enriched):
    feat = build_enriched_features(enriched)
    for c in ALL_FEATURES:
        assert c in feat.columns


# ---- served model matches training contract -------------------------------
def test_saved_model_scores_single_row(enriched):
    joblib = pytest.importorskip("joblib")
    path = ROOT / "models" / "churn_model_v2.joblib"
    if not path.exists():
        pytest.skip("v2 model not trained yet")
    model = joblib.load(path)
    X, _ = make_xy_enriched(enriched)
    p = model.predict_proba(X.iloc[[0]][ALL_FEATURES])[:, 1]
    assert 0.0 <= float(p[0]) <= 1.0
