"""Run the full PlayerPulse pipeline end to end, in order.

Usage:  .venv/Scripts/python.exe run_pipeline.py
Each stage is a standalone script; we run them with the current interpreter so
the same environment (and trained artifacts) is used throughout.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("Clean / build modeling table", "notebooks/02_clean.py"),
    ("EDA report + figures", "notebooks/01_eda.py"),
    ("Baseline model (4 features)", "notebooks/03_model.py"),
    ("Simulate telemetry", "src/simulate.py"),
    ("v2 model: tuned + calibrated + CV", "notebooks/05_model_v2.py"),
    ("Temporal-leakage ablation", "notebooks/06_leakage_test.py"),
    ("A/B causal analysis (ATE)", "src/experiment.py"),
]


def main() -> int:
    for i, (name, script) in enumerate(STAGES, 1):
        print(f"\n{'#' * 70}\n# [{i}/{len(STAGES)}] {name}\n#   {script}\n{'#' * 70}")
        r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if r.returncode != 0:
            print(f"\n!! stage failed: {script} (exit {r.returncode})")
            return r.returncode
    print("\nPipeline complete. Launch the dashboard with:\n"
          "  .venv/Scripts/streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
