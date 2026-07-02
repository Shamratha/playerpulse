"""Data loading and label definition for PlayerPulse.

The raw dataset is the Cookie Cats mobile-game A/B test (90,189 players):
    userid          unique player id
    version         A/B group: gate_30 (control) or gate_40 (variant)
    sum_gamerounds  total game rounds played in the first 14 days
    retention_1     player came back 1 day after install  (bool)
    retention_7     player came back 7 days after install  (bool)

For churn modelling we define the target as the inverse of 7-day retention:
a player who did NOT return on day 7 is treated as churned.
"""
from pathlib import Path
import pandas as pd

# Project paths (resolved relative to this file, so it works from anywhere)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"
RAW_CSV = DATA_DIR / "cookie_cats.csv"


def load_raw() -> pd.DataFrame:
    """Load the raw CSV with correct dtypes."""
    df = pd.read_csv(RAW_CSV)
    # retention_1 / retention_7 arrive as the strings "True"/"False" or as bools
    for col in ("retention_1", "retention_7"):
        if df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False})
        df[col] = df[col].astype(bool)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the churn label: churned = did not return on day 7."""
    df = df.copy()
    df["churned"] = (~df["retention_7"]).astype(int)
    return df


def load() -> pd.DataFrame:
    """Convenience: raw data + churn label."""
    return add_target(load_raw())


# Single lone outlier in the raw data: one player has 49,854 rounds while the
# next-highest is 2,961 (and that player shows retention_1=False, which is
# implausible for ~3,500 rounds/day). Treated as a data-quality artifact.
OUTLIER_ROUNDS_THRESHOLD = 10_000

PROCESSED_CSV = DATA_DIR / "playerpulse_clean.csv"


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a cleaned modeling table plus a log of what was changed.

    Decisions:
      * drop the single 49,854-round outlier (see OUTLIER_ROUNDS_THRESHOLD)
      * keep 0-round players — they are real installs that never engaged and
        are the most important churn segment, not noise
    """
    log = {"rows_in": len(df)}
    out = df.copy()

    outliers = out["sum_gamerounds"] >= OUTLIER_ROUNDS_THRESHOLD
    log["outliers_dropped"] = int(outliers.sum())
    out = out.loc[~outliers].reset_index(drop=True)

    log["rows_out"] = len(out)
    log["zero_round_players"] = int((out["sum_gamerounds"] == 0).sum())
    log["churn_rate"] = round(float(out["churned"].mean()), 4)
    return out, log


def load_clean() -> pd.DataFrame:
    """Cleaned modeling table (raw + label + outlier removal)."""
    clean_df, _ = clean(load())
    return clean_df


if __name__ == "__main__":
    df = load()
    print(df.head())
    print(f"\nrows={len(df):,}  churn_rate={df['churned'].mean():.3f}")
