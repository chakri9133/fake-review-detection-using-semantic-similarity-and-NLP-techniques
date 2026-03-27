"""
Central configuration for local development and cloud deployment.
"""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Datasets"
EMBEDDINGS_DIR = BASE_DIR / "Embeddings"
MODELS_DIR = BASE_DIR / "Models"
CACHE_DIR = BASE_DIR / "cache"


BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "5000")))
BACKEND_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
BACKEND_URL = os.getenv("BACKEND_URL", f"http://localhost:{BACKEND_PORT}")


DEFAULT_FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
extra_origins = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = DEFAULT_FRONTEND_ORIGINS + [FRONTEND_URL]
if extra_origins.strip():
    CORS_ORIGINS.extend(origin.strip() for origin in extra_origins.split(",") if origin.strip())

CORS_ORIGINS = list(dict.fromkeys(CORS_ORIGINS))


MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "logistic_model.joblib"))
MODEL_METRICS_PATH = Path(os.getenv("MODEL_METRICS_PATH", MODELS_DIR / "model_metrics.json"))
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")


TEXT_MIN_LENGTH = int(os.getenv("TEXT_MIN_LENGTH", "5"))
TEXT_MAX_LENGTH = int(os.getenv("TEXT_MAX_LENGTH", "5000"))


THRESHOLD_GENUINE = float(os.getenv("THRESHOLD_GENUINE", "0.30"))
THRESHOLD_FAKE = float(os.getenv("THRESHOLD_FAKE", "0.75"))


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_DETAILED_LOGGING = os.getenv("ENABLE_DETAILED_LOGGING", "true").lower() == "true"
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "30"))
