# Deployment Guide

This project is now prepared for a split deployment:

- `Render` for the Flask API and model inference
- `Vercel` for the React frontend

## 1. Before You Deploy

Make sure these files exist in your repo before pushing:

- `Models/logistic_model.joblib`
- `Models/model_metrics.json`

Without those two files, the backend can start but predictions will fail.

## 2. Deploy the Backend on Render

Create a new Web Service on Render and connect your repo.

Use these settings if Render does not auto-detect them:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

Set this environment variable in Render:

- `FRONTEND_URL`
  Use your Vercel frontend URL after the frontend is deployed.

Optional environment variables:

- `MODEL_PATH=Models/logistic_model.joblib`
- `MODEL_METRICS_PATH=Models/model_metrics.json`
- `SBERT_MODEL_NAME=all-MiniLM-L6-v2`
- `THRESHOLD_GENUINE=0.30`
- `THRESHOLD_FAKE=0.75`

Render will provide a public backend URL such as:

`https://your-backend-name.onrender.com`

Test these endpoints after deploy:

- `/health`
- `/api/metrics`

## 3. Deploy the Frontend on Vercel

Import the `frontend` directory as the Vercel project root.

Set this environment variable in Vercel:

- `VITE_API_BASE_URL=https://your-backend-name.onrender.com`

Use these defaults if Vercel asks:

- Framework preset: `Vite`
- Build command: `npm run build`
- Output directory: `dist`

## 4. Local Development

Backend:

```bash
pip install -r requirements.txt
python app.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

For local frontend API calls, create `frontend/.env` with:

```bash
VITE_API_BASE_URL=http://localhost:5000
```

## 5. Important Notes

- The backend now uses portable relative paths instead of Windows-only absolute paths.
- The frontend no longer hardcodes `localhost:5000`; it reads the API base URL from `VITE_API_BASE_URL`.
- Flask still runs locally with `python app.py`, but production hosting should use `gunicorn app:app`.
- The first prediction request may feel slower on free hosting because the model has to load into memory.
