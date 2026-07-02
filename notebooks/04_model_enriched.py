"""PlayerPulse — enriched multi-driver churn model + baseline comparison.

Run:  .venv/Scripts/python.exe notebooks/04_model_enriched.py
Trains on real + simulated features, compares to the 4-feature baseline,
saves models/churn_model_enriched.joblib and reports/metrics_enriched.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, brier_score_loss,
)
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_clean, FIG_DIR  # noqa: E402
from simulate import simulate  # noqa: E402
from features import (  # noqa: E402
    make_xy, make_xy_enriched, ENRICHED_NUMERIC, ENRICHED_CATEGORICAL,
)

MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = ROOT / "reports"
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------ data
base = load_clean()
enriched = simulate(base)

# identical split for a fair baseline-vs-enriched comparison
idx = np.arange(len(enriched))
tr, te = train_test_split(idx, test_size=0.25, random_state=42,
                          stratify=enriched["churned"])

Xb, yb = make_xy(base)                    # baseline 4 features
Xe, ye = make_xy_enriched(enriched)       # enriched 13 num + 2 cat
assert (yb.values == ye.values).all()     # same rows/labels

y_tr, y_te = ye.iloc[tr], ye.iloc[te]

# ------------------------------------------------------------------ preprocessors
pre = ColumnTransformer([
    ("num", StandardScaler(), ENRICHED_NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ENRICHED_CATEGORICAL),
])


def evaluate(model, Xtr, Xte):
    model.fit(Xtr, y_tr)
    p = model.predict_proba(Xte)[:, 1]
    return p, {
        "roc_auc": round(roc_auc_score(y_te, p), 4),
        "pr_auc": round(average_precision_score(y_te, p), 4),
        "brier": round(brier_score_loss(y_te, p), 4),
    }

# ------------------------------------------------------------------ baseline (4 feat)
base_pipe = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
_, base_scores = evaluate(base_pipe, Xb.iloc[tr], Xb.iloc[te])

# ------------------------------------------------------------------ enriched models
enr_logreg = Pipeline([("pre", pre),
                       ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
enr_hgb = Pipeline([("pre", pre),
                    ("clf", HistGradientBoostingClassifier(
                        class_weight="balanced", learning_rate=0.08,
                        max_depth=4, max_iter=300, random_state=42))])

p_lr, lr_scores = evaluate(enr_logreg, Xe.iloc[tr], Xe.iloc[te])
p_hgb, hgb_scores = evaluate(enr_hgb, Xe.iloc[tr], Xe.iloc[te])

print("=" * 62)
print("MODEL COMPARISON  (held-out test, n=%d)" % len(te))
print("=" * 62)
row = "{:<26} {:>8} {:>8} {:>8}"
print(row.format("model", "ROC-AUC", "PR-AUC", "Brier"))
print(row.format("baseline logreg (4 feat)", base_scores["roc_auc"], base_scores["pr_auc"], base_scores["brier"]))
print(row.format("enriched logreg (15)", lr_scores["roc_auc"], lr_scores["pr_auc"], lr_scores["brier"]))
print(row.format("enriched HGB (15)", hgb_scores["roc_auc"], hgb_scores["pr_auc"], hgb_scores["brier"]))

# ------------------------------------------------------------------ pick + report
best_name, best_model, best_p, best_scores = max(
    [("enriched_logreg", enr_logreg, p_lr, lr_scores),
     ("enriched_hgb", enr_hgb, p_hgb, hgb_scores)],
    key=lambda t: t[3]["roc_auc"],
)
lift = round(best_scores["roc_auc"] - base_scores["roc_auc"], 4)
print(f"\n>>> best: {best_name}   ROC-AUC lift over baseline: +{lift}")

pred = (best_p >= 0.5).astype(int)
print("\nclassification report (threshold=0.50):")
print(classification_report(y_te, pred, target_names=["retained", "churned"]))
print("confusion matrix [rows=true, cols=pred]:\n", confusion_matrix(y_te, pred))

# ------------------------------------------------------------------ importance
print("\ncomputing permutation importance (this takes a few seconds)...")
perm = permutation_importance(best_model, Xe.iloc[te], y_te, n_repeats=5,
                              random_state=42, scoring="roc_auc")
feat_names = ENRICHED_NUMERIC + ENRICHED_CATEGORICAL
imp = pd.Series(perm.importances_mean, index=feat_names).sort_values(ascending=False)
print("\ntop drivers (drop in ROC-AUC when shuffled):\n", imp.round(4).head(10))

# figure: importance
fig, ax = plt.subplots(figsize=(8, 5))
imp.sort_values().plot(kind="barh", ax=ax, color="#2c7fb8")
ax.set_title("Enriched model — permutation feature importance")
ax.set_xlabel("mean ROC-AUC drop when shuffled")
fig.tight_layout(); fig.savefig(FIG_DIR / "05_enriched_importance.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ persist
joblib.dump(best_model, MODEL_DIR / "churn_model_enriched.joblib")
out = {
    "best_model": best_name,
    "baseline_scores": base_scores,
    "enriched_scores": {"logreg": lr_scores, "hgb": hgb_scores},
    "roc_auc_lift": lift,
    "numeric_features": ENRICHED_NUMERIC,
    "categorical_features": ENRICHED_CATEGORICAL,
    "importance": {k: round(float(v), 5) for k, v in imp.items()},
}
(REPORT_DIR / "metrics_enriched.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved model -> {MODEL_DIR/'churn_model_enriched.joblib'}")
print(f"saved metrics -> {REPORT_DIR/'metrics_enriched.json'}")
