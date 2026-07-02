# Deploying PlayerPulse (Streamlit Community Cloud — free)

The dashboard is a single Streamlit app, so it deploys in ~15 minutes and gives
you a public URL to put on a resume (recruiters click links; they rarely clone
repos and run pipelines).

## 1. Push to GitHub

Commit the code **and the artifacts the app reads** so every tab works on first
load without re-running the pipeline:

```
git init && git add .
git commit -m "PlayerPulse dashboard"
git remote add origin https://github.com/<you>/playerpulse.git
git push -u origin main
```

Make sure these are committed (they are small):
- `app.py`, `src/`, everything under version control
- `models/churn_model_v2.joblib`  (the trained model)
- `reports/metrics_v2.json`, `reports/metrics_leakage.json`, `reports/metrics_experiment.json`
- `reports/figures/06_v2_diagnostics.png`
- `data/cookie_cats.csv`  (~2 MB — fine to commit)
- `requirements.txt`, `.streamlit/config.toml`

`.gitignore` already excludes `.venv/` and caches. If the model artifact is
missing, the app trains a quick calibrated model on startup as a fallback — but
committing it is faster and keeps the exact tested model.

## 2. Deploy

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → pick your repo, branch `main`, main file `app.py`.
3. (Optional) Advanced settings → Python version **3.13** to match local.
4. **Deploy.** First build installs `requirements.txt` (a few minutes).

Your app goes live at `https://<you>-playerpulse.streamlit.app`.

## Notes
- `requirements.txt` pins the exact tested versions so `joblib.load` matches the
  scikit-learn the model was trained on — this avoids version-skew load errors.
- First load computes scores for all 90k players; Streamlit caches it, so only the
  first visitor waits.
- To update the live app, just `git push` — Streamlit redeploys automatically.
