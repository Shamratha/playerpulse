# PlayerPulse

**🔗 Live demo: [playerpulse-xlyl.onrender.com](https://playerpulse-xlyl.onrender.com)**
*(free tier — the first visit after idle takes ~1 min to wake up)*

Player **churn prediction & retention analytics** for mobile games, built on the
real *Cookie Cats* A/B-test dataset (90,189 players).

## Pipeline

| Stage | File | What it does |
|-------|------|--------------|
| Load / label / clean | `src/data.py` | Reads raw CSV, defines `churned = not day-7 retention`, drops the lone 49,854-round outlier |
| Feature engineering | `src/features.py` | Baseline + enriched feature builders |
| EDA | `notebooks/01_eda.py` | Integrity, target, engagement, A/B, figures |
| Cleaning table | `notebooks/02_clean.py` | Writes `data/playerpulse_clean.csv` |
| Model (baseline) | `notebooks/03_model.py` | LogReg vs HistGradientBoosting on 4 real features |
| Simulated telemetry | `src/simulate.py` | Coherent sessions/monetization/progression/social/UX — **no label leakage** |
| Modeling components | `src/modeling.py` | Shared preprocessor, estimator, bootstrap CI, slice-AUC (train == serve) |
| Model (v2) | `notebooks/05_model_v2.py` | Tuned + isotonic-calibrated + cross-validated → `models/churn_model_v2.joblib` |
| Leakage ablation | `notebooks/06_leakage_test.py` | Quantifies how much AUC rides on the 14-day (leaky) feature |
| A/B causal analysis | `src/experiment.py` | ATE of gate_40 with z-test + bootstrap CI; why CATE isn't identifiable |
| Campaign economics | `src/economics.py` | Break-even targeting: `p × effectiveness × LTV > cost`, ROI curve |
| Tests | `tests/test_pipeline.py` | Label, cleaning, feature-contract, leakage-regression, serving-parity |
| Runner | `run_pipeline.py` | Runs every stage in order |
| Dashboard | `app.py` | Streamlit: retention + A/B causal readout, model comparison, risk predictor with **per-player SHAP**, at-risk, campaign planner. Interactive tools run the **full enriched model** as a day-14 monitoring view (see [Which model the dashboard uses](#which-model-the-dashboard-uses)) |

## Run

```bash
# full pipeline, in order
.venv/Scripts/python.exe run_pipeline.py

# tests
.venv/Scripts/python.exe -m pytest -q

# dashboard
.venv/Scripts/streamlit run app.py     # http://localhost:8501
```

## Model

**Headline (leakage-free) predictive result:**

| Model | Features | ROC-AUC | 95% CI |
|-------|----------|--------:|:------:|
| **Leakage-safe (reported result)** | `retention_1`, `version` — provably pre-day-7 | **0.716** | 0.708–0.723 |
| Engagement-inclusive | + 14-day `sum_gamerounds` | 0.891 | 0.886–0.896 |
| Full enriched | + 11 simulated features | 0.890 | 0.885–0.896 |

*Methodology: all three use HistGradientBoosting on the **same** 25% held-out test
split (`random_state=42`) with the **same** bootstrap seed (42), so the numbers are
directly comparable.* Leakage-safe adds CV 0.710 ± 0.003 and calibrated Brier 0.135.

**Why 0.716 is the number I lead with.** The label is 7-day retention, but
`sum_gamerounds` counts the first **14 days** — it partly postdates the label, so a
model that uses it is not a fair *predictive* number (it's peeking). This dataset
ships only the 14-day aggregate, so the feature can't be truncated to ≤7 days;
the honest predictive result is therefore the model built from features that
provably predate the label. **0.716 AUC from day-1 signals alone is a solid,
deployable early-warning result** — and it's the number that would survive contact
with production.

**Only two leakage-safe features exist in this dataset.** The raw data ships just
five columns — `userid`, `version`, `sum_gamerounds`, `retention_1`, `retention_7`
— of which `retention_7` is the label and the only fields knowable before day 7 are
`retention_1` (day-1 return) and `version` (assigned at install). The leakage-safe
model uses those two; there are simply no other day-≤7 features to add. `retention_1`
carries most of that signal — and that is *expected*, not a problem: early returners
are genuinely more likely to be retained. It's measured days before the label, so it
is a **legitimate, powerful early indicator, not a circular proxy**. Strong
early-signal features are a normal and desirable property of retention data.

**Two honest findings from the comparison:**
1. Dropping the 14-day feature moves AUC 0.89 → 0.72. That gap is the engagement
   feature's contribution — *part legitimate day-0–6 play, part day-8–14 leakage*
   we can't separate without event logs. So 0.716 is a conservative floor and the
   true ≤7-day model sits between the two.
2. The engagement (0.891) and full-enriched (0.890) confidence intervals overlap
   almost entirely → the simulated features add **no real lift**. An earlier version
   reported a +0.037 "lift", but that came from a simulator that conditioned
   features on the label — circular, and now fixed.

### Which model the dashboard uses

The interactive tools — **risk predictor + per-player SHAP, at-risk list, and
campaign planner** — run the **full enriched model** (`churn_model_v2.joblib`,
14 features including the 14-day window), *not* the leakage-safe headline model.
That is deliberate, and it is not a contradiction of the methodology above:

- Those tools are a **day-14 monitoring / what-if view**. By the time you score a
  player, the 14-day window has already closed, so using `sum_gamerounds` there is
  **not leakage** — it would only be leakage if used to predict *before* the window
  closes. The leakage-safe **0.716** model is the honest number for genuine *early*
  prediction; the two answer different questions (early warning vs. retrospective
  characterization), and both are legitimate.
- The full model also makes the SHAP explanations and segment views richer, which
  is the point of an exploration UI.

### What the rigor pass fixed

- **Leakage-safe headline** — the reported number excludes the temporally-leaky
  14-day feature; all models compared on one split (see methodology above).
- **Removed label leakage** in the simulator — features derive only from real
  engagement + noise. A regression test guards against reintroducing it.
- **Calibration over class-weighting** — dropped `class_weight="balanced"` (it
  distorts probabilities the ROI tool depends on) in favour of isotonic
  calibration. Brier (**engagement-inclusive** model, calibrated) improved
  **0.107 → 0.091**. For reference the **leakage-safe** model's calibrated Brier is
  **0.135** — higher because a 2-feature model is less *sharp*, not miscalibrated.
- **Cross-validation + bootstrap 95% CI** — no more single-split point estimates.
- **Hyperparameter tuning** (`RandomizedSearchCV`, CV on train only).
- **Business-cost decision threshold** instead of 0.5.
- **Slice analysis** — consistent across version/payer/platform, but ~0.60
  (near-random) for zero-engagement players: a real weakness the old pipeline hid.
- **Tests + pipeline runner** for reproducibility.
- **Per-player SHAP** explanations in the risk-predictor tab (additivity verified).
- **A/B causal analysis** — ATE of gate_40 with z-test + bootstrap CI
  (−0.82pp on day-7 retention, p=0.0016); documents why CATE isn't identifiable.

## Known caveats / deferred

- **Simulated features are synthetic** (`src/simulate.py`) — they exercise the
  production machinery (encoding, calibration, CV, slices) and are ready for real
  telemetry; they are not real behaviour.
- `sum_gamerounds` spans 14 days, partly postdating the 7-day label — temporal
  leakage inherent to this public dataset. A true fix needs day-≤N event logs.
- **CATE (personalized uplift) is not identifiable here** — the gate A/B has no
  *pre-treatment* covariates (players are randomized at install with no prior
  history), so only the average effect (ATE) is estimable, not an individual
  treatment effect. This is a data limitation, not a scoping choice — see
  `src/experiment.py`.
- **Deferred by scope** (analysis project, not a service): survival (time-to-churn)
  framing, MLflow experiment tracking, containerization.

## If I had production telemetry

The simulated features stand in for a real event log. In production I would ingest
raw, timestamped events and rebuild the feature layer from them:

| Event | Fields | Enables |
|-------|--------|---------|
| `session_start` / `session_end` | user_id, ts, duration | real sessions/day, session length, **day-≤N** engagement windows |
| `round_complete` | user_id, ts, level, result | rounds *bounded to the prediction horizon* — closes the temporal leakage properly |
| `purchase` | user_id, ts, sku, amount | real monetization (spend, first-purchase latency, whale flags) |
| `level_complete` | user_id, ts, level | true progression / difficulty walls |
| `ad_impression` | user_id, ts, placement, reward | ad engagement without the pay/ad confound |
| `crash` / `error` | user_id, ts, device, build | real UX friction |

Only the **feature-building layer** (`src/features.py`) changes — it would compute
windowed aggregates strictly from events at or before day *N* (e.g. `rounds_0_3`,
`sessions_0_7`). Everything downstream (`modeling.py`, calibration, CV, the
dashboard, the economics) stays identical, because they all consume a feature
matrix, not raw data. This is exactly why the pipeline is split the way it is.

## Deployment

See [DEPLOY.md](DEPLOY.md) — one-command push to **Streamlit Community Cloud** for
a public, shareable URL.

## Data source

Cookie Cats — [ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats](https://github.com/ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats)
· [Kaggle mirror](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)
