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

---

# Alternative: deploy on Render (free)

The repo includes a **`render.yaml` Blueprint**, so Render can configure the whole
service automatically.

1. Go to <https://dashboard.render.com> and sign in with GitHub.
2. **New +** → **Blueprint** → connect `Shamratha/playerpulse` → **Apply**.
   Render reads `render.yaml` and provisions a free web service.
3. Wait for the first build (installs `requirements.txt`, a few minutes).

Your app goes live at `https://playerpulse.onrender.com` (or a similar URL Render
assigns).

### What `render.yaml` sets
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
  (plus `--server.enableCORS false --server.enableXsrfProtection false` so Streamlit's
  WebSocket works behind Render's proxy).
- **Health check:** `/_stcore/health` (Streamlit's built-in endpoint).
- **Python:** pinned via `PYTHON_VERSION` to match the trained model.

### Render gotchas
- **Free tier spins down** after ~15 min idle; the next visit cold-starts in ~1 min.
- **512 MB RAM** is a bit tight for the full stack. The model is committed (no
  training at startup) and `shap` is imported lazily (only when "Explain this
  prediction" is clicked), which keeps baseline memory down. If a page still OOMs,
  bump to the paid Starter instance in the service settings.
- If the build fails on the Python version, edit `PYTHON_VERSION` in `render.yaml`
  to a 3.13.x patch Render currently offers (or remove it to use Render's default).

### Manual setup (without the Blueprint)
New **Web Service** → connect repo → **Build:** `pip install -r requirements.txt` →
**Start:** the `streamlit run ...` command above → **Instance:** Free.
