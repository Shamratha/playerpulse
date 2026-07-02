"""PlayerPulse — v2 churn model: honest, tuned, calibrated, cross-validated.

Run:  .venv/Scripts/python.exe notebooks/05_model_v2.py

Improvements over 03/04:
  * simulator no longer leaks the label (see src/simulate.py)
  * no class_weight='balanced'; isotonic CALIBRATION instead (ROI needs it)
  * hyperparameter tuning (RandomizedSearchCV, CV inside train only)
  * k-fold CV mean +/- std and a bootstrap 95% CI on test AUC
  * decision threshold from campaign economics, not 0.5
  * per-segment (slice) AUC
Saves models/churn_model_v2.joblib and reports/metrics_v2.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix,
)
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_clean, FIG_DIR  # noqa: E402
from simulate import simulate  # noqa: E402
from features import make_xy_enriched  # noqa: E402
from modeling import base_pipeline, bootstrap_auc_ci, slice_auc, ALL_FEATURES  # noqa: E402
from economics import Campaign  # noqa: E402

MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = ROOT / "reports"
sns.set_theme(style="whitegrid")


def h(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


# ------------------------------------------------------------------ data
enriched = simulate(load_clean())
X, y = make_xy_enriched(enriched)
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, np.arange(len(X)), test_size=0.25, random_state=42, stratify=y)
print(f"features={len(ALL_FEATURES)}  train={len(X_tr):,}  test={len(X_te):,}  "
      f"churn={y.mean():.3f}")

# ------------------------------------------------------------------ HPO (train only)
h("1. HYPERPARAMETER SEARCH (RandomizedSearchCV, 4-fold, on train only)")
param_dist = {
    "clf__learning_rate": loguniform(0.02, 0.2),
    "clf__max_depth": [3, 4, 5, 6, None],
    "clf__max_leaf_nodes": [15, 31, 63],
    "clf__min_samples_leaf": [20, 50, 100],
    "clf__l2_regularization": [0.0, 0.1, 1.0],
    "clf__max_iter": [200, 300, 400],
}
search = RandomizedSearchCV(
    base_pipeline(), param_dist, n_iter=15,
    cv=StratifiedKFold(4, shuffle=True, random_state=42),
    scoring="roc_auc", n_jobs=-1, random_state=42, refit=True,
)
search.fit(X_tr, y_tr)
best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
bi = search.best_index_
cv_mean = float(search.cv_results_["mean_test_score"][bi])
cv_std = float(search.cv_results_["std_test_score"][bi])
print(f"best CV ROC-AUC: {cv_mean:.4f} +/- {cv_std:.4f}")
print("best params:", best_params)

# ------------------------------------------------------------------ calibration
h("2. CALIBRATION (isotonic, 5-fold) vs uncalibrated")
uncal = base_pipeline(**best_params).fit(X_tr, y_tr)
p_uncal = uncal.predict_proba(X_te)[:, 1]

cal = CalibratedClassifierCV(base_pipeline(**best_params), method="isotonic", cv=5)
cal.fit(X_tr, y_tr)
p_cal = cal.predict_proba(X_te)[:, 1]

test = {
    "roc_auc": round(roc_auc_score(y_te, p_cal), 4),
    "pr_auc": round(average_precision_score(y_te, p_cal), 4),
    "brier_uncalibrated": round(brier_score_loss(y_te, p_uncal), 4),
    "brier_calibrated": round(brier_score_loss(y_te, p_cal), 4),
}
print(f"test ROC-AUC (cal): {test['roc_auc']}   PR-AUC: {test['pr_auc']}")
print(f"Brier  uncalibrated: {test['brier_uncalibrated']}  ->  "
      f"calibrated: {test['brier_calibrated']}")

ci = bootstrap_auc_ci(y_te, p_cal)
print(f"test ROC-AUC 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# ------------------------------------------------------------------ threshold
h("3. DECISION THRESHOLD from campaign economics (not 0.5)")
thr = Campaign().break_even_threshold
pred = (p_cal >= thr).astype(int)
cm = confusion_matrix(y_te, pred)
print(f"operating threshold = {thr:.3f}  (LTV=$25, cost=$1, eff=15%)")
print(classification_report(y_te, pred, target_names=["retained", "churned"]))
print("confusion matrix [rows=true, cols=pred]:\n", cm)

# ------------------------------------------------------------------ slice analysis
h("4. SLICE AUC (does it work across segments?)")
te_rows = enriched.iloc[idx_te].reset_index(drop=True)
slices = {
    "version": slice_auc(y_te, p_cal, te_rows["version"]),
    "payer": slice_auc(y_te, p_cal, np.where(te_rows["n_purchases"] > 0, "payer", "free")),
    "platform": slice_auc(y_te, p_cal, te_rows["platform"]),
    "engaged": slice_auc(y_te, p_cal, np.where(te_rows["sum_gamerounds"] > 0, "played", "zero")),
}
for name, d in slices.items():
    print(f"  {name:10s} {d}")

# ------------------------------------------------------------------ importance
h("5. PERMUTATION IMPORTANCE (calibrated model)")
perm = permutation_importance(cal, X_te, y_te, n_repeats=5,
                              random_state=42, scoring="roc_auc", n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=ALL_FEATURES).sort_values(ascending=False)
print(imp.round(4).head(8))

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for p, lbl in [(p_uncal, "uncalibrated"), (p_cal, "calibrated")]:
    frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax[0].plot(mean_pred, frac_pos, marker="o", label=lbl)
ax[0].plot([0, 1], [0, 1], "--", color="gray")
ax[0].set(xlabel="mean predicted prob", ylabel="observed churn rate",
          title="Calibration (reliability) curve"); ax[0].legend()
imp.sort_values().plot(kind="barh", ax=ax[1], color="#5145cd")
ax[1].set_title("Permutation importance (v2)"); ax[1].set_xlabel("ROC-AUC drop")
fig.tight_layout(); fig.savefig(FIG_DIR / "06_v2_diagnostics.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ persist
try:
    baseline_auc = json.loads((REPORT_DIR / "metrics.json").read_text())["scores"]["hgb"]["roc_auc"]
except Exception:
    baseline_auc = None

joblib.dump(cal, MODEL_DIR / "churn_model_v2.joblib")
out = {
    "model": "HistGradientBoosting + isotonic calibration",
    "best_params": best_params,
    "cv_auc_mean": round(cv_mean, 4),
    "cv_auc_std": round(cv_std, 4),
    "test": test,
    "auc_ci95": [round(ci[0], 4), round(ci[1], 4)],
    "operating_threshold": round(float(thr), 4),
    "confusion_at_threshold": cm.tolist(),
    "baseline_roc_auc": baseline_auc,
    "slice_auc": slices,
    "importance": {k: round(float(v), 5) for k, v in imp.items()},
    "features": ALL_FEATURES,
}
(REPORT_DIR / "metrics_v2.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved model  -> {MODEL_DIR/'churn_model_v2.joblib'}")
print(f"saved metrics -> {REPORT_DIR/'metrics_v2.json'}")
