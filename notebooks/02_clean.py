"""PlayerPulse — data cleaning / build the modeling table.

Run:  python notebooks/02_clean.py
Writes data/playerpulse_clean.csv and prints a cleaning log.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load, clean, PROCESSED_CSV  # noqa: E402

df = load()
clean_df, log = clean(df)

print("cleaning log:")
for k, v in log.items():
    print(f"  {k:20s} {v}")

# sanity: the extreme outlier must be gone, 0-round players must remain
assert clean_df["sum_gamerounds"].max() < 10_000, "outlier not removed"
assert (clean_df["sum_gamerounds"] == 0).any(), "0-round players wrongly dropped"

clean_df.to_csv(PROCESSED_CSV, index=False)
print(f"\nwrote {len(clean_df):,} rows -> {PROCESSED_CSV}")
print("\npreview:\n", clean_df.head())
