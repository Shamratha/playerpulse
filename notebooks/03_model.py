"""PlayerPulse — train & evaluate the churn model.

Run:  python notebooks/03_model.py
Saves the fitted pipeline to models/ and metrics to reports/.
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss,
)
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_clean, FIG_DIR  # noqa: E402
from features import make_xy, FEATURES  # noqa: E402

MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = ROOT / "reports"
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------ data split
df = load_clean()
X, y = make_xy(df)
# stratify to preserve the 81% churn ratio in both splits
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"train={len(X_tr):,}  test={len(X_te):,}  features={FEATURES}")
print(f"churn rate  train={y_tr.mean():.3f}  test={y_te.mean():.3f}")

# ------------------------------------------------------------------ two models
# class_weight='balanced' counteracts the 81/19 imbalance so the model does not
# just predict "everyone churns".
models = {
    "logreg": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    ),
    "hgb": HistGradientBoostingClassifier(
        class_weight="balanced", learning_rate=0.1,
        max_depth=4, random_state=42,
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    results[name] = {
        "roc_auc": roc_auc_score(y_te, p),
        "pr_auc": average_precision_score(y_te, p),   # better for imbalance
        "brier": brier_score_loss(y_te, p),           # calibration quality
        "proba": p,
        "model": model,
    }
    print(f"\n[{name}]  ROC-AUC={results[name]['roc_auc']:.4f}  "
          f"PR-AUC={results[name]['pr_auc']:.4f}  Brier={results[name]['brier']:.4f}")

# ------------------------------------------------------------------ pick winner
best_name = max(results, key=lambda n: results[n]["roc_auc"])
best = results[best_name]
print(f"\n>>> best model: {best_name}")

# classification report at the default 0.5 threshold
p = best["proba"]
pred = (p >= 0.5).astype(int)
print("\nclassification report (threshold=0.50):")
print(classification_report(y_te, pred, target_names=["retained", "churned"]))
cm = confusion_matrix(y_te, pred)
print("confusion matrix [rows=true, cols=pred]:\n", cm)

# ------------------------------------------------------------------ feature signal
if best_name == "logreg":
    coefs = best["model"].named_steps["logisticregression"].coef_[0]
    imp = pd.Series(coefs, index=FEATURES).sort_values()
    print("\nlogreg coefficients (standardized):\n", imp)
else:
    # permutation-free proxy: use logreg coefs for interpretability alongside
    lr = models["logreg"]
    coefs = lr.named_steps["logisticregression"].coef_[0]
    imp = pd.Series(coefs, index=FEATURES).sort_values()
    print("\n(logreg coefficients for interpretability):\n", imp)

# ------------------------------------------------------------------ figures
fpr, tpr, _ = roc_curve(y_te, p)
prec, rec, _ = precision_recall_curve(y_te, p)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(fpr, tpr, label=f"{best_name} (AUC={best['roc_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "--", color="gray")
ax[0].set(xlabel="FPR", ylabel="TPR", title="ROC curve"); ax[0].legend()
ax[1].plot(rec, prec, color="#c0392b", label=f"AP={best['pr_auc']:.3f}")
ax[1].axhline(y_te.mean(), ls="--", color="gray", label=f"baseline={y_te.mean():.2f}")
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall"); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG_DIR / "04_model_curves.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ persist
joblib.dump(best["model"], MODEL_DIR / "churn_model.joblib")
metrics = {
    "best_model": best_name,
    "features": FEATURES,
    "test_size": len(X_te),
    "scores": {n: {k: round(float(v[k]), 4) for k in ("roc_auc", "pr_auc", "brier")}
               for n, v in results.items()},
    "coefficients": {f: round(float(c), 4) for f, c in imp.items()},
}
(REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\nsaved model -> {MODEL_DIR/'churn_model.joblib'}")
print(f"saved metrics -> {REPORT_DIR/'metrics.json'}")
