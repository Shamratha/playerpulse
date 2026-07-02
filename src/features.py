"""Feature engineering for the PlayerPulse churn model.

We only have a handful of raw columns, so the goal here is to extract every
bit of honest signal without inventing data:

    log_gamerounds   log1p of total rounds — tames the 185-skew so linear
                     models see a usable gradient instead of one giant tail
    retention_1      did the player return on day 1? This is known BEFORE the
                     day-7 churn label, so it is a legitimate early-warning
                     signal, not leakage.
    is_gate_40       A/B variant flag (gate_40 = 1). Captures the small but
                     real retention effect of the level-40 gate.
    played_zero      installed but never played a single round (4.4% of users)
                     — the highest-risk segment, given its own flag.

NOTE ON LEAKAGE: sum_gamerounds is counted over the first 14 days, which
partly postdates the 7-day retention label. This is the standard (if imperfect)
convention for this public dataset; we surface it rather than hide it. A truly
day-7-truncated round count would be the clean fix if raw event logs existed.
"""
import numpy as np
import pandas as pd

FEATURES = ["log_gamerounds", "retention_1", "is_gate_40", "played_zero"]
TARGET = "churned"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with engineered feature columns added."""
    out = df.copy()
    out["log_gamerounds"] = np.log1p(out["sum_gamerounds"])
    out["retention_1"] = out["retention_1"].astype(int)
    out["is_gate_40"] = (out["version"] == "gate_40").astype(int)
    out["played_zero"] = (out["sum_gamerounds"] == 0).astype(int)
    return out


def make_xy(df: pd.DataFrame):
    """Return (X, y) ready for scikit-learn."""
    feat = build_features(df)
    return feat[FEATURES].copy(), feat[TARGET].copy()


# ---------------------------------------------------------------------------
# Enriched feature space (real baseline + simulated behavioral telemetry).
# Heavy-tailed counts/money are log1p-compressed so the linear model sees a
# usable gradient; tree models are unaffected by the monotonic transform.
# ---------------------------------------------------------------------------
# NB: played_zero dropped in v2 — it is collinear with log_gamerounds
# (log1p(0)=0) and contributed ~0 importance. Redundant feature removed.
ENRICHED_NUMERIC = [
    "log_gamerounds", "retention_1", "is_gate_40",
    "log_sessions", "avg_session_min", "days_since_install", "log_level",
    "n_purchases", "log_spend", "log_ads", "friends_invited", "crashes",
]
ENRICHED_CATEGORICAL = ["platform", "region"]


def build_enriched_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered enriched columns; expects simulated columns present."""
    out = build_features(df)
    out["log_sessions"] = np.log1p(out["sessions_count"])
    out["log_level"] = np.log1p(out["level_reached"])
    out["log_spend"] = np.log1p(out["total_spend_usd"])
    out["log_ads"] = np.log1p(out["ads_watched"])
    return out


def make_xy_enriched(df: pd.DataFrame):
    """Return (X, y) for the enriched model — numeric + categorical columns."""
    feat = build_enriched_features(df)
    cols = ENRICHED_NUMERIC + ENRICHED_CATEGORICAL
    return feat[cols].copy(), feat[TARGET].copy()
