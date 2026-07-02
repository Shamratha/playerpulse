"""A/B experiment analysis — the gate_30 vs gate_40 randomized test (change #3).

Players were randomly assigned at install to keep the first gate at level 30
(control) or move it to level 40 (treatment). Because assignment is RANDOM, the
difference in retention between groups is a genuine causal effect — the Average
Treatment Effect (ATE).

Why we do the ATE and NOT a personalized uplift / CATE model:
    A CATE ("which individual should get which gate") requires PRE-treatment
    covariates to personalize on. In this dataset players are randomized at
    install with NO prior history — every feature we have (rounds, sessions,
    spend, retention_1...) is measured AFTER treatment. Conditioning an uplift
    model on post-treatment variables is not causally identified; it would be a
    descriptive artifact dressed up as personalization. So the honest, defensible
    analysis this experiment supports is the ATE with a significance test and a
    confidence interval, plus a clearly-labelled *descriptive* heterogeneity view.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_clean  # noqa: E402

CONTROL, TREATMENT = "gate_30", "gate_40"


def two_proportion_ztest(succ_c, n_c, succ_t, n_t):
    """Two-sided z-test for a difference in proportions (treatment - control)."""
    p_c, p_t = succ_c / n_c, succ_t / n_t
    p_pool = (succ_c + succ_t) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se
    p_value = 2 * stats.norm.sf(abs(z))
    return float(p_t - p_c), float(z), float(p_value)


def bootstrap_diff_ci(a_control, a_treat, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for mean(treat) - mean(control)."""
    rng = np.random.default_rng(seed)
    a_c, a_t = np.asarray(a_control), np.asarray(a_treat)
    diffs = [rng.choice(a_t, a_t.size, replace=True).mean()
             - rng.choice(a_c, a_c.size, replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def ate(df, outcome):
    """Average treatment effect of gate_40 vs gate_30 on a 0/1 outcome column."""
    c = df.loc[df["version"] == CONTROL, outcome].astype(int)
    t = df.loc[df["version"] == TREATMENT, outcome].astype(int)
    diff, z, p = two_proportion_ztest(c.sum(), len(c), t.sum(), len(t))
    lo, hi = bootstrap_diff_ci(c.values, t.values)
    return {
        "outcome": outcome,
        "control_rate": round(float(c.mean()), 4),
        "treatment_rate": round(float(t.mean()), 4),
        "effect_pp": round(diff * 100, 3),          # percentage points
        "ci95_pp": [round(lo * 100, 3), round(hi * 100, 3)],
        "z": round(z, 3), "p_value": round(p, 5),
        "significant_0.05": bool(p < 0.05),
    }


def heterogeneity(df, outcome="retention_7"):
    """DESCRIPTIVE effect by engagement bucket (NOT causal — post-treatment split)."""
    d = df.copy()
    d["bucket"] = pd.cut(d["sum_gamerounds"], [-1, 5, 30, 100, 1e9],
                         labels=["0-5", "6-30", "31-100", "100+"])
    rows = []
    for b, g in d.groupby("bucket", observed=True):
        c = g.loc[g.version == CONTROL, outcome].astype(int)
        t = g.loc[g.version == TREATMENT, outcome].astype(int)
        rows.append({"bucket": str(b), "players": len(g),
                     "control_rate": round(float(c.mean()), 4),
                     "treatment_rate": round(float(t.mean()), 4),
                     "effect_pp": round((t.mean() - c.mean()) * 100, 3)})
    return rows


def analyze():
    df = load_clean()
    return {
        "retention_1": ate(df, "retention_1"),
        "retention_7": ate(df, "retention_7"),
        "heterogeneity_day7": heterogeneity(df, "retention_7"),
    }


if __name__ == "__main__":
    import json
    res = analyze()
    for horizon in ("retention_1", "retention_7"):
        a = res[horizon]
        arrow = "lower" if a["effect_pp"] < 0 else "higher"
        print(f"\n{horizon}:  control {a['control_rate']:.1%}  ->  treatment "
              f"{a['treatment_rate']:.1%}")
        print(f"  effect (gate_40 - gate_30): {a['effect_pp']:+.2f} pp "
              f"(95% CI [{a['ci95_pp'][0]:+.2f}, {a['ci95_pp'][1]:+.2f}] pp)")
        print(f"  z={a['z']}, p={a['p_value']}  -> "
              f"{'SIGNIFICANT' if a['significant_0.05'] else 'not significant'} "
              f"({arrow} retention under gate_40)")
    print("\nDescriptive heterogeneity by engagement (day-7, NOT causal):")
    print(pd.DataFrame(res["heterogeneity_day7"]).to_string(index=False))

    out = Path(__file__).resolve().parents[1] / "reports" / "metrics_experiment.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {out}")
