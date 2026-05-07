# Sales Forecasting System

Production-grade time series forecasting for weekly sales by state.

Overview
-
This repository contains a production-ready backend service that:

- loads historical weekly sales per state from `data/Forecasting Case- Study.xlsx`;
- trains multiple model families (SARIMA, Prophet, XGBoost, LSTM);
- evaluates models using RMSE/MAE/MAPE and selects the best model per state;
- exposes forecasts via a FastAPI HTTP API (`api/app.py`).

Client-focused quick start
-
1) Create a Python 3.12 virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Ensure the dataset is at:

```
data/Forecasting Case- Study.xlsx
```

3) Start the API (uses the pre-built registry at `models/saved/real_registry.pkl`):

```bash
set DATA_PATH=%CD%\data\Forecasting Case- Study.xlsx
set REGISTRY_PATH=%CD%\models\saved\real_registry.pkl
.\.venv\Scripts\python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

4) Example requests:

```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/forecast?state=Gujarat&horizon=8"
```

Response example:

```json
{
  "state": "Gujarat",
  "forecast": [
    {"week": "2026-06-01", "sales": 1200},
    {"week": "2026-06-08", "sales": 1350}
  ],
  "model_used": "XGBoost"
}
```

Project layout (client bundle)
-
- `api/` — FastAPI app (`api/app.py`).
- `data/` — `preprocessor.py` and the official dataset `Forecasting Case- Study.xlsx`.
- `models/` — model implementations and `models/saved/real_registry.pkl` (pre-trained registry used by the API).
- `requirements.txt` — Python dependencies.
- `README.md` — this file.

Files excluded from client bundle
-
To keep the client distribution small and focused on inference we removed development-only helpers (demo/generate scripts and local logs). If you need to retrain, I can restore training scripts on a separate `dev` branch.

Developer notes (optional/retraining)
-
- Python 3.12 is recommended for TensorFlow (LSTM) compatibility.
- When retraining, use a strict chronological train/validation split (last 8 weeks holdout).
- For large-scale retraining, build registries offline and place the final `*.pkl` in `models/saved/` for API use.

Publishing to GitHub
-
1. Create a new GitHub repository.
2. Add a sensible `.gitignore` (see provided `.gitignore`).
3. Commit and push the client bundle to the repo:

```bash
git init
git add .
git commit -m "Client-facing release: API + pretrained registry"
git remote add origin <git-url>
git push -u origin main
```

Want me to push? If you provide your GitHub personal access token and target repo url I can prepare the branch and push the client bundle for you (or I can give step-by-step commands you can run locally).

Generated on 2026-05-07
