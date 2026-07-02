"""Augment the real Cookie Cats data with SIMULATED behavioral features.

Why: the raw dataset has essentially one strong predictor (engagement). To turn
PlayerPulse into a realistic *multi-driver* pipeline we synthesise the kind of
telemetry a real mobile game would log — sessions, monetization, progression,
social, and UX signals.

INTEGRITY NOTE (v2): an earlier version conditioned some features on the churn
label (payers/social skewed "retained", crashes skewed "churned"). That is
circular — it injects signal and then "discovers" it, inflating measured lift.
This version generates EVERY feature from a player's REAL engagement plus
independent noise, with NO reference to the label. As a result any lift the
enriched model shows is honest (and, as expected, small — these features are
mostly proxies for engagement, which already drives churn). The value of the
enriched pipeline is the production machinery (categorical encoding, calibration,
CV, slice analysis) and its readiness for real telemetry, not a headline AUC gain.

Fully reproducible: fixed RNG seed, no wall-clock randomness.
"""
import numpy as np
import pandas as pd

SEED = 42

NUMERIC_SIM = [
    "sessions_count", "avg_session_min", "days_since_install", "level_reached",
    "n_purchases", "total_spend_usd", "ads_watched", "friends_invited", "crashes",
]
CATEGORICAL_SIM = ["platform", "region"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Return df with simulated behavioral columns added (no label leakage)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    rounds = out["sum_gamerounds"].to_numpy()
    ret1 = out["retention_1"].astype(int).to_numpy()   # a real pre-day-7 signal, OK to use
    played = (rounds > 0).astype(int)

    # standardized engagement — the only shared latent driver (real behaviour)
    logr = np.log1p(rounds)
    z = (logr - logr.mean()) / logr.std()

    # SESSIONS: real rounds split into play sessions
    rounds_per_session = rng.gamma(shape=2.0, scale=2.5, size=n)
    sessions = np.where(rounds == 0, 0, np.maximum(1, np.round(rounds / rounds_per_session)))
    out["sessions_count"] = sessions.astype(int)

    # SESSION LENGTH: engaged players play longer
    avg_len = 3 + 5 * _sigmoid(z) + rng.normal(0, 1.0, n)
    out["avg_session_min"] = (np.clip(avg_len, 0.5, None).round(2)) * played

    # TENURE: days since install, weakly tied to engagement
    days = np.round(30 + 18 * z + rng.normal(0, 22, n))
    out["days_since_install"] = np.clip(days, 1, 180).astype(int)

    # PROGRESSION: level grows with rounds (noisy)
    level = np.floor(rounds / (1.4 * rng.lognormal(0, 0.25, n)))
    out["level_reached"] = np.clip(level, 0, 500).astype(int)

    # MONETIZATION: zero-inflated; payer likelihood rises with engagement & day-1
    # return only (both real signals) — NOT with the churn label.
    payer_logit = -3.6 + 1.1 * z + 0.4 * ret1
    is_payer = rng.random(n) < _sigmoid(payer_logit)
    n_purch = np.where(is_payer, 1 + rng.poisson(np.maximum(0.2, 2 + z), n), 0)
    out["n_purchases"] = n_purch.astype(int)
    per_purchase = rng.lognormal(mean=1.5, sigma=0.9, size=n)
    out["total_spend_usd"] = np.round(n_purch * per_purchase, 2)

    # ADS: non-payers monetize via ads; scales with sessions
    ads_lam = np.maximum(0.1, 0.15 * sessions * (1 - 0.6 * is_payer))
    out["ads_watched"] = rng.poisson(ads_lam).astype(int)

    # SOCIAL: invites scale with engagement (no label term)
    friends_lam = np.maximum(0.05, 0.25 + 0.5 * np.maximum(z, 0))
    out["friends_invited"] = (rng.poisson(friends_lam) * played).astype(int)

    # UX: crashes scale with usage only (no label term)
    crash_lam = np.exp(-1.2 + 0.3 * np.log1p(sessions))
    out["crashes"] = rng.poisson(crash_lam).astype(int)

    # CATEGORICAL
    out["platform"] = rng.choice(["iOS", "Android"], size=n, p=[0.4, 0.6])
    out["region"] = rng.choice(
        ["NA", "EU", "LATAM", "APAC", "Other"], size=n,
        p=[0.30, 0.28, 0.15, 0.20, 0.07],
    )
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import load_clean, DATA_DIR

    enriched = simulate(load_clean())
    path = DATA_DIR / "playerpulse_enriched.csv"
    enriched.to_csv(path, index=False)
    print(f"wrote {len(enriched):,} rows, {enriched.shape[1]} cols -> {path}")
    print("\ncorrelation of simulated features with churn (should be modest / via engagement):")
    corr = enriched[NUMERIC_SIM + ["churned"]].corr()["churned"].drop("churned")
    print(corr.round(3).sort_values())
