"""PlayerPulse — Exploratory Data Analysis.

Run:  python notebooks/01_eda.py
Produces console report + figures in reports/figures/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load, FIG_DIR  # noqa: E402

sns.set_theme(style="whitegrid")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


df = load()

# ---------------------------------------------------------------- 1. INTEGRITY
section("1. STRUCTURE & INTEGRITY")
print(f"shape: {df.shape}")
print("\ndtypes:\n", df.dtypes)
print("\nmissing values per column:\n", df.isna().sum())
print(f"\nduplicate userids: {df['userid'].duplicated().sum()}")
print(f"fully duplicate rows: {df.duplicated().sum()}")

# ---------------------------------------------------------------- 2. TARGET
section("2. TARGET: churn (no return on day 7)")
print(df["churned"].value_counts().rename({0: "retained", 1: "churned"}))
print(f"\nchurn rate: {df['churned'].mean():.4f}")
print(f"day-1 retention: {df['retention_1'].mean():.4f}")
print(f"day-7 retention: {df['retention_7'].mean():.4f}")

# ---------------------------------------------------------------- 3. ENGAGEMENT
section("3. ENGAGEMENT: sum_gamerounds")
print(df["sum_gamerounds"].describe(percentiles=[.25, .5, .75, .9, .99]))
print(f"\nplayers with 0 rounds: {(df['sum_gamerounds'] == 0).sum():,} "
      f"({(df['sum_gamerounds'] == 0).mean():.2%})")
print(f"max rounds (possible outlier): {df['sum_gamerounds'].max():,}")
print(f"skewness: {df['sum_gamerounds'].skew():.1f}")

# ---------------------------------------------------------------- 4. A/B GROUPS
section("4. A/B GROUPS (version)")
grp = df.groupby("version").agg(
    players=("userid", "size"),
    mean_rounds=("sum_gamerounds", "mean"),
    median_rounds=("sum_gamerounds", "median"),
    ret1=("retention_1", "mean"),
    ret7=("retention_7", "mean"),
    churn=("churned", "mean"),
)
print(grp)

# --------------------------------------------------- 5. ENGAGEMENT vs CHURN
section("5. ENGAGEMENT vs CHURN")
by_churn = df.groupby("churned")["sum_gamerounds"].agg(["mean", "median", "count"])
print(by_churn.rename(index={0: "retained", 1: "churned"}))

# bucket gamerounds and show churn rate per bucket
bins = [-1, 0, 1, 5, 10, 30, 50, 100, 1e9]
labels = ["0", "1", "2-5", "6-10", "11-30", "31-50", "51-100", "100+"]
df["rounds_bucket"] = pd.cut(df["sum_gamerounds"], bins=bins, labels=labels)
churn_by_bucket = df.groupby("rounds_bucket", observed=True).agg(
    players=("userid", "size"), churn_rate=("churned", "mean")
)
print("\nchurn rate by gamerounds bucket:\n", churn_by_bucket)

# ---------------------------------------------------------------- FIGURES
section("6. SAVING FIGURES")

# fig 1: gamerounds distribution (raw vs log), capped for readability
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df.loc[df.sum_gamerounds <= 200, "sum_gamerounds"], bins=60, ax=axes[0])
axes[0].set_title("sum_gamerounds (<=200)")
sns.histplot(np.log1p(df["sum_gamerounds"]), bins=60, ax=axes[1], color="teal")
axes[1].set_title("log1p(sum_gamerounds)")
fig.tight_layout(); fig.savefig(FIG_DIR / "01_gamerounds_dist.png", dpi=110); plt.close(fig)

# fig 2: churn rate by rounds bucket
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=churn_by_bucket.reset_index(), x="rounds_bucket", y="churn_rate", ax=ax, color="#c0392b")
ax.axhline(df["churned"].mean(), ls="--", color="gray", label="overall churn")
ax.set_title("Churn rate by engagement level"); ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR / "02_churn_by_engagement.png", dpi=110); plt.close(fig)

# fig 3: retention & churn by A/B version
fig, ax = plt.subplots(figsize=(7, 4))
grp[["ret1", "ret7", "churn"]].plot(kind="bar", ax=ax)
ax.set_title("Retention / churn by A/B version"); ax.set_ylabel("rate"); ax.tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig(FIG_DIR / "03_ab_version.png", dpi=110); plt.close(fig)

print(f"saved 3 figures to {FIG_DIR}")
print("\nEDA complete.")
