"""Retention-campaign economics — turn churn scores into targeting decisions.

The model gives each player a churn probability p. A retention campaign (push
notification, offer, bonus currency) costs `cost` per targeted player and, for a
player who *would* have churned, wins them back with probability `effectiveness`
(the campaign's uplift). A retained player is worth `ltv` (lifetime value).

Decision for one player with churn prob p:
    do nothing        -> expected value = (1 - p) * ltv
    run campaign       -> expected value = (1 - p) * ltv + p * eff * ltv - cost
    uplift of campaign -> p * eff * ltv - cost

So targeting is profitable exactly when
    p * eff * ltv - cost > 0   <=>   p > cost / (eff * ltv)
which is the break-even churn threshold. We rank players by churn risk and target
everyone above it. (Simplifying assumptions: the campaign only affects would-be
churners and does not annoy loyal players; both are noted, not hidden.)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Campaign:
    ltv: float = 25.0            # value of a retained player ($)
    cost: float = 1.0           # cost per targeted player ($)
    effectiveness: float = 0.15  # P(win back | would-be churner)

    @property
    def break_even_threshold(self) -> float:
        """Churn prob above which targeting a player is profitable."""
        denom = self.effectiveness * self.ltv
        return 1.0 if denom <= 0 else min(1.0, self.cost / denom)


def plan(proba: np.ndarray, camp: Campaign) -> dict:
    """Summarise targeting the players whose churn prob clears break-even."""
    p = np.asarray(proba, dtype=float)
    uplift = p * camp.effectiveness * camp.ltv - camp.cost  # per-player $ uplift
    target = uplift > 0
    n = len(p)
    n_t = int(target.sum())

    expected_saved = float((p[target] * camp.effectiveness).sum())  # players won back
    spend = n_t * camp.cost
    gross = expected_saved * camp.ltv
    net = float(uplift[target].sum())                               # gross - spend
    return {
        "threshold": camp.break_even_threshold,
        "targeted": n_t,
        "targeted_share": n_t / n,
        "expected_players_saved": expected_saved,
        "campaign_spend": spend,
        "gross_value_recovered": gross,
        "net_value": net,
        "roi": (net / spend) if spend > 0 else 0.0,
    }


def roi_curve(proba: np.ndarray, camp: Campaign) -> pd.DataFrame:
    """Net value as a function of how many players we target (ranked by risk).

    Targeting players highest-risk first, cumulative net uplift traces out a curve
    that peaks exactly at the break-even threshold — useful to show the optimum.
    """
    p = np.sort(np.asarray(proba, dtype=float))[::-1]     # high risk first
    per_player_uplift = p * camp.effectiveness * camp.ltv - camp.cost
    cum_net = np.cumsum(per_player_uplift)
    k = np.arange(1, len(p) + 1)
    return pd.DataFrame({
        "n_targeted": k,
        "share_targeted": k / len(p),
        "cumulative_net_value": cum_net,
    })


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import joblib
    from data import load_clean
    from simulate import simulate
    from features import build_enriched_features, ENRICHED_NUMERIC, ENRICHED_CATEGORICAL

    ROOT = Path(__file__).resolve().parents[1]
    df = build_enriched_features(simulate(load_clean()))
    model = joblib.load(ROOT / "models" / "churn_model_enriched.joblib")
    proba = model.predict_proba(df[ENRICHED_NUMERIC + ENRICHED_CATEGORICAL])[:, 1]

    camp = Campaign(ltv=25, cost=1.0, effectiveness=0.15)
    res = plan(proba, camp)
    print(f"break-even churn threshold: {res['threshold']:.3f}")
    for k, v in res.items():
        if k == "threshold":
            continue
        print(f"  {k:24s} {v:,.2f}")

    # compare: naive "target everyone" vs optimal
    all_camp = plan(proba, Campaign(ltv=25, cost=1.0, effectiveness=0.15))
    naive_net = float((proba * 0.15 * 25 - 1.0).sum())
    print(f"\ntarget EVERYONE net value: ${naive_net:,.0f}")
    print(f"target OPTIMAL  net value: ${res['net_value']:,.0f}  "
          f"(+${res['net_value'] - naive_net:,.0f} by not wasting spend on safe players)")
