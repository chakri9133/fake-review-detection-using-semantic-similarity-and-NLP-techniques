# Problem Statement

Fake reviews on e-commerce platforms can mislead customers and reduce trust in online marketplaces. This project aims to identify whether a review is likely fake or genuine using machine learning and natural language processing. The system preprocesses review text, generates semantic embeddings, trains classification models, and serves predictions through a Flask backend with a React frontend.

# Project Code

## train_classifier.py

```py
import numpy as np
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from config import EMBEDDINGS_DIR, MODELS_DIR

# =====================================================
# 1. LOAD EMBEDDINGS
# =====================================================
EMB_PATH = EMBEDDINGS_DIR / "sbert_embeddings.npy"
LABEL_PATH = EMBEDDINGS_DIR / "labels.npy"

X = np.load(EMB_PATH)
y = np.load(LABEL_PATH)

print("Embeddings loaded:", X.shape)

# =====================================================
# 2. TRAIN-TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 3. TRAIN CLASSIFIER
# =====================================================
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

clf.fit(X_train, y_train)

# =====================================================
# 3.5 SAVE TRAINED MODEL
# =====================================================
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = MODELS_DIR / "logistic_model.joblib"
joblib.dump(clf, MODEL_PATH)

print("âœ… Trained model saved at:", MODEL_PATH)

# =====================================================
# 4. EVALUATION
# =====================================================
y_pred = clf.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===\n")
print(classification_report(y_test, y_pred))

print("\n=== CONFUSION MATRIX ===\n")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# 5. SAVE REAL METRICS (AUTO-GENERATED)
# =====================================================
report = classification_report(y_test, y_pred, output_dict=True)

metrics = {
    "precision_fake": round(report["1"]["precision"], 3),
    "recall_fake": round(report["1"]["recall"], 3),
    "f1_fake": round(report["1"]["f1-score"], 3),
    "support_fake": int(report["1"]["support"]),
    "accuracy": round(report["accuracy"], 3),
    "dataset_size": int(len(y)),
    "balanced_dataset": True
}

METRICS_PATH = MODELS_DIR / "model_metrics.json"
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("âœ… Model metrics updated and saved at:", METRICS_PATH)
```

## semantic_similarity.py

```py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
from config import EMBEDDINGS_DIR

# =====================================================
# 1. LOAD EMBEDDINGS & LABELS
# =====================================================
EMB_PATH = EMBEDDINGS_DIR / "sbert_embeddings.npy"
LABEL_PATH = EMBEDDINGS_DIR / "labels.npy"

X = np.load(EMB_PATH)
y = np.load(LABEL_PATH)

print("Total embeddings:", X.shape)

# =====================================================
# 2. SEPARATE FAKE & GENUINE
# =====================================================
fake_indices = np.where(y == 1)[0]
genuine_indices = np.where(y == 0)[0]

print("Fake reviews:", len(fake_indices))
print("Genuine reviews:", len(genuine_indices))

# =====================================================
# 3. SAMPLE REVIEWS (IMPORTANT)
# =====================================================
SAMPLE_SIZE = 2000   # Safe & sufficient for analysis

fake_sample_idx = np.random.choice(fake_indices, SAMPLE_SIZE, replace=False)
genuine_sample_idx = np.random.choice(genuine_indices, SAMPLE_SIZE, replace=False)

fake_embeddings = X[fake_sample_idx]
genuine_embeddings = X[genuine_sample_idx]

# =====================================================
# 4. COSINE SIMILARITY COMPUTATION
# =====================================================
fake_fake_sim = cosine_similarity(fake_embeddings, fake_embeddings)
genuine_genuine_sim = cosine_similarity(genuine_embeddings, genuine_embeddings)
fake_genuine_sim = cosine_similarity(fake_embeddings, genuine_embeddings)

# =====================================================
# 5. REMOVE SELF-SIMILARITY (DIAGONAL)
# =====================================================
fake_fake_sim = fake_fake_sim[np.triu_indices_from(fake_fake_sim, k=1)]
genuine_genuine_sim = genuine_genuine_sim[np.triu_indices_from(genuine_genuine_sim, k=1)]

# =====================================================
# 6. CALCULATE MEAN SIMILARITIES
# =====================================================
print("\n=== SEMANTIC SIMILARITY RESULTS ===")

print(f"Average Fakeâ€“Fake similarity     : {fake_fake_sim.mean():.4f}")
print(f"Average Genuineâ€“Genuine similarity: {genuine_genuine_sim.mean():.4f}")
print(f"Average Fakeâ€“Genuine similarity   : {fake_genuine_sim.mean():.4f}")
```

## preprocess_kindle_reviews.py

```py
import pandas as pd
import re
import os
import json
from config import DATASETS_DIR

# =====================================================
# 1. BASE DIRECTORY
# =====================================================
DATASET_ROOT = str(DATASETS_DIR)

print("Searching datasets inside:", DATASET_ROOT)

# =====================================================
# 2. AMAZON REVIEW COLUMN SIGNATURE
# =====================================================
REQUIRED_COLUMNS = {
    "asin",
    "overall",
    "reviewText"
}

csv_file = None
json_file = None

# =====================================================
# 3. FIND CORRECT AMAZON DATASET
# =====================================================
for root, dirs, files in os.walk(DATASET_ROOT):
    for file in files:
        full_path = os.path.join(root, file)
        file_lower = file.lower()

        try:
            # Check CSV
            if file_lower.endswith(".csv") or file_lower == "kindle_reviews":
                temp_df = pd.read_csv(full_path, nrows=5)
                if REQUIRED_COLUMNS.issubset(set(temp_df.columns)):
                    csv_file = full_path
                    break

            # Check JSON
            if file_lower.endswith(".json"):
                with open(full_path, "r", encoding="utf-8") as f:
                    sample = json.loads(next(f))
                if REQUIRED_COLUMNS.issubset(sample.keys()):
                    json_file = full_path
                    break
        except Exception:
            continue

    if csv_file or json_file:
        break

# =====================================================
# 4. LOAD DATASET
# =====================================================
if csv_file:
    print("âœ… Amazon Kindle CSV found:", csv_file)
    df = pd.read_csv(csv_file)

elif json_file:
    print("âš ï¸ CSV not found, loading Amazon Kindle JSON:", json_file)
    data = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

else:
    raise FileNotFoundError(
        "âŒ No Amazon Kindle review dataset found inside Datasets"
    )

print("Initial dataset shape:", df.shape)

# =====================================================
# 5. STANDARD AMAZON COLUMNS
# =====================================================
required_columns = [
    "asin",
    "reviewerID",
    "overall",
    "summary",
    "reviewText",
    "reviewTime",
    "unixReviewTime"
]

# Some datasets may miss reviewerID or summary
existing_columns = [col for col in required_columns if col in df.columns]
df = df[existing_columns]

# =====================================================
# 6. HANDLE MISSING VALUES
# =====================================================
df["reviewText"] = df["reviewText"].fillna("")
if "summary" in df.columns:
    df["summary"] = df["summary"].fillna("")
else:
    df["summary"] = ""

df = df[df["reviewText"].str.strip() != ""]
print("After removing empty reviews:", df.shape)

# =====================================================
# 7. COMBINE TEXT
# =====================================================
df["text"] = df["summary"] + ". " + df["reviewText"]

# =====================================================
# 8. CLEAN TEXT (SBERT SAFE)
# =====================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"&quot;|&#34;", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# =====================================================
# 9. REMOVE DUPLICATES
# =====================================================
df = df.drop_duplicates(subset=["clean_text"])
print("After removing duplicates:", df.shape)

# =====================================================
# 10. LABEL CREATION (WEAK SUPERVISION)
# =====================================================
df["label"] = df["overall"].apply(lambda x: 1 if x <= 2 else 0)

# =====================================================
# 11. FINAL DATASET
# =====================================================
final_df = df[
    ["asin", "overall", "clean_text", "label"]
]

print("Final dataset shape:", final_df.shape)

# =====================================================
# 12. SAVE OUTPUT
# =====================================================
OUTPUT_PATH = os.path.join(DATASET_ROOT, "kindle_reviews_clean.csv")
final_df.to_csv(OUTPUT_PATH, index=False)

print("==============================================")
print("âœ… PREPROCESSING COMPLETED SUCCESSFULLY")
print("ðŸ“ Output saved at:")
print(OUTPUT_PATH)
print("==============================================")
```

## predict_review.py

```py
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
from config import MODEL_PATH, SBERT_MODEL_NAME, THRESHOLD_FAKE, THRESHOLD_GENUINE

# =====================================================
# 2. TEXT CLEANING (SAME AS TRAINING)
# =====================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"&quot;|&#34;", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# 3. LOAD MODELS
# =====================================================
print("Loading SBERT model...")
embedder = SentenceTransformer(SBERT_MODEL_NAME)

print("Loading classifier...")
clf = joblib.load(MODEL_PATH)

# =====================================================
# 4. DECISION POLICY
# =====================================================
def decision_policy(p_fake):
    if p_fake < THRESHOLD_GENUINE:
        return "LIKELY GENUINE"
    elif p_fake > THRESHOLD_FAKE:
        return "LIKELY FAKE"
    else:
        return "UNCERTAIN"

# =====================================================
# 5. MAIN PREDICTION FUNCTION
# =====================================================
def predict_review(review_text):
    """
    Input  : Raw review text (string)
    Output : Dictionary with label + probability
    """

    # Clean text
    cleaned = clean_text(review_text)

    # Embed
    embedding = embedder.encode(
        cleaned,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).reshape(1, -1)

    # Predict probability
    p_fake = clf.predict_proba(embedding)[0, 1]

    # Decision
    label = decision_policy(p_fake)

    return {
    "review": review_text,
    "prediction": label,
    "fake_probability": round(float(p_fake), 4),
    "confidence_percent": round(float(p_fake) * 100, 2)
}


# =====================================================
# 6. CLI INTERFACE (FOR TESTING)
# =====================================================
if __name__ == "__main__":
    print("\nEnter a review (type 'exit' to quit):\n")

    while True:
        user_review = input("> ")

        if user_review.lower() == "exit":
            break

        result = predict_review(user_review)

        print("\n--- RESULT ---")
        print("Prediction      :", result["prediction"])
        print("Fake Probability:", result["fake_probability"])
        print("----------------\n")
```

## package-lock.json

```json
{
  "name": "Fake Review Detection",
  "lockfileVersion": 3,
  "requires": true,
  "packages": {}
}
```

## config.py

```py
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
```

## sbert_embeddings.py

```py
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import DATASETS_DIR, EMBEDDINGS_DIR, SBERT_MODEL_NAME

# =====================================================
# 1. PATHS
# =====================================================
DATA_PATH = DATASETS_DIR / "kindle_reviews_clean.csv"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

EMB_PATH = EMBEDDINGS_DIR / "sbert_embeddings.npy"
LABEL_PATH = EMBEDDINGS_DIR / "labels.npy"

# =====================================================
# 2. LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

texts = df["clean_text"].astype(str).tolist()
labels = df["label"].values

print("Total reviews:", len(texts))

# =====================================================
# 3. LOAD SBERT MODEL
# =====================================================
print("Loading SBERT model...")
model = SentenceTransformer(SBERT_MODEL_NAME)

# =====================================================
# 4. BATCH-WISE EMBEDDING (CRITICAL)
# =====================================================
BATCH_SIZE = 512   # Safe for CPU, increase to 1024 if RAM allows
embeddings = []

print("Generating SBERT embeddings...")
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(
        batch_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)

print("Embeddings shape:", embeddings.shape)

# =====================================================
# 5. SAVE EMBEDDINGS
# =====================================================
np.save(EMB_PATH, embeddings)
np.save(LABEL_PATH, labels)

print("==============================================")
print("âœ… SBERT EMBEDDINGS GENERATED & SAVED")
print("ðŸ“ Embeddings:", EMB_PATH)
print("ðŸ“ Labels:", LABEL_PATH)
print("==============================================")
```

## run_other_models_and_save_results.py

```py
import json
import subprocess
import os
from config import BASE_DIR

MODELS_DIR = BASE_DIR / "All Other Models"
RESULTS_DIR = BASE_DIR / "model_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "other_models_results.json"

models = {
    "SBERT": "train_classifier.py"
}

results = {}

def run_model(model_name, script_name):
    print(f"\nðŸš€ Running {model_name}...\n")

    process = subprocess.run(
        ["python", str(MODELS_DIR / script_name)],
        capture_output=True,
        text=True
    )

    output = process.stdout

    # ---- Parse metrics from classification report ----
    lines = output.split("\n")
    for line in lines:
        if "accuracy" in line:
            accuracy = float(line.split()[-2])
        if line.strip().startswith("1"):
            precision = float(line.split()[1])
            recall = float(line.split()[2])
            f1 = float(line.split()[3])

    results[model_name] = {
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1
    }

    print(f"âœ… {model_name} done")

# ==========================
# RUN ALL MODELS
# ==========================
for name, script in models.items():
    run_model(name, script)

# ==========================
# SAVE JSON
# ==========================
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\n==============================")
print("ðŸŽ‰ ALL MODEL RESULTS SAVED")
print("ðŸ“„", RESULTS_PATH)
print("==============================")
```

## requirements.txt

```txt
Flask==2.3.0
Flask-CORS==4.0.0
sentence-transformers==2.2.2
scikit-learn==1.3.0
joblib==1.3.1
numpy==1.24.3
pandas==2.2.3
tqdm==4.67.1
gunicorn==23.0.0
```

## render.yaml

```yaml
services:
  - type: web
    name: fake-review-detection-api
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.11
      - key: FRONTEND_URL
        sync: false
```

## app.py

```py
"""
Flask Backend API for Fake Review Detection
Connects React Frontend to SBERT Model + Logistic Regression Classifier
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_review import predict_review
import json
from config import (
    BACKEND_DEBUG,
    BACKEND_HOST,
    BACKEND_PORT,
    CORS_ORIGINS,
    MODEL_METRICS_PATH,
    TEXT_MAX_LENGTH,
    TEXT_MIN_LENGTH,
)

app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, resources={
    r"/api/*": {
        "origins": CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load model metrics
if MODEL_METRICS_PATH.exists():
    with MODEL_METRICS_PATH.open("r", encoding="utf-8") as f:
        model_metrics = json.load(f)
else:
    model_metrics = {"status": "Model metrics not found"}


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API Endpoint: Predict if a review is fake or genuine
    
    Request:
    {
        "review": "Review text here"
    }
    
    Response:
    {
        "review": "Review text here",
        "prediction": "LIKELY GENUINE|LIKELY FAKE|UNCERTAIN",
        "fake_probability": 0.XX,
        "confidence_percent": XX.XX
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        review_text = data.get("review", "").strip()
        
        if not review_text:
            return jsonify({"error": "Review text is required"}), 400
        
        if len(review_text) < TEXT_MIN_LENGTH:
            return jsonify({"error": f"Review must be at least {TEXT_MIN_LENGTH} characters long"}), 400

        if len(review_text) > TEXT_MAX_LENGTH:
            return jsonify({"error": f"Review must be at most {TEXT_MAX_LENGTH} characters long"}), 400
        
        # Get prediction from SBERT model
        result = predict_review(review_text)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """API Endpoint: Get model metrics and statistics"""
    try:
        return jsonify(model_metrics), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "backend": "Flask",
        "model": "SBERT + Logistic Regression",
        "api_version": "1.0"
    }), 200


@app.route("/", methods=["GET"])
def index():
    """Root endpoint"""
    return jsonify({
        "message": "Fake Review Detection API",
        "endpoints": {
            "POST /api/predict": "Analyze a review",
            "GET /api/metrics": "Get model metrics",
            "GET /health": "Health check"
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Fake Review Detection - Backend API")
    print("=" * 60)
    print("SBERT Model: all-MiniLM-L6-v2")
    print("Classifier: Logistic Regression")
    print("Frontend origins:", ", ".join(CORS_ORIGINS))
    print(f"Backend: Flask ({BACKEND_HOST}:{BACKEND_PORT})")
    print("=" * 60)
    print("Starting server...")
    print("=" * 60)
    
    app.run(
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        debug=BACKEND_DEBUG,
        use_reloader=True
    )
```

## model_results\models_results.json

```json
{
    "BoW + Naive Bayes": {
        "accuracy": 0.86,
        "precision_fake": 0.85,
        "recall_fake": 0.89,
        "f1_fake": 0.87
    },
    "TF-IDF + Naive Bayes": {
        "accuracy": 0.88,
        "precision_fake": 0.96,
        "recall_fake": 0.28,
        "f1_fake": 0.44
    },
    "GRU (from-scratch embeddings)": {
        "accuracy": 0.85,
        "precision_fake": 0.8218,
        "recall_fake": 0.8029,
        "f1_fake": 0.8323
    },
    "SBERT + Logistic Regression": {
    "accuracy": 0.89,
    "precision_fake": 0.88,
    "recall_fake": 0.90,
    "f1_fake": 0.89
    }

}
```

## beauty_model\sbert_embeddings_beauty.py

```py
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import BASE_DIR, EMBEDDINGS_DIR, SBERT_MODEL_NAME

# =====================================================
# 1. PATHS (BEAUTY DATASET ONLY)
# =====================================================
DATA_PATH = os.path.join(
    BASE_DIR,
    "Datasets",
    "Beauty Products Review Rating Prediction",
    "dataset",
    "beauty_train_clean.csv"
)

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

EMB_PATH = EMBEDDINGS_DIR / "beauty_sbert_embeddings.npy"
LABEL_PATH = EMBEDDINGS_DIR / "beauty_labels.npy"

# =====================================================
# 2. LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

texts = df["clean_text"].astype(str).tolist()
labels = df["label"].values

print("Total Beauty reviews:", len(texts))

# =====================================================
# 3. LOAD SBERT MODEL
# =====================================================
print("Loading SBERT model...")
model = SentenceTransformer(SBERT_MODEL_NAME)

# =====================================================
# 4. BATCH-WISE EMBEDDING (CPU SAFE)
# =====================================================
BATCH_SIZE = 512   # Same as Kindle for consistency
embeddings = []

print("Generating SBERT embeddings for Beauty dataset...")
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(
        batch_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)

print("Embeddings shape:", embeddings.shape)

# =====================================================
# 5. SAVE EMBEDDINGS (NEW NAMES)
# =====================================================
np.save(EMB_PATH, embeddings)
np.save(LABEL_PATH, labels)

print("==============================================")
print("âœ… BEAUTY SBERT EMBEDDINGS GENERATED & SAVED")
print("ðŸ“ Embeddings:", EMB_PATH)
print("ðŸ“ Labels:", LABEL_PATH)
print("==============================================")
```

## beauty_model\preprocess_beauty_reviews.py

```py
import pandas as pd
import re
import os
from config import DATASETS_DIR

# =====================================================
# 1. PATHS
# =====================================================
DATASET_DIR = os.path.join(
    DATASETS_DIR,
    "Beauty Products Review Rating Prediction",
    "dataset"
)

TRAIN_PATH = os.path.join(DATASET_DIR, "Train.csv")
OUTPUT_PATH = os.path.join(DATASET_DIR, "beauty_train_clean.csv")

print("Loading Beauty Train dataset from:")
print(TRAIN_PATH)

# =====================================================
# 2. LOAD DATASET
# =====================================================
df = pd.read_csv(TRAIN_PATH)
print("Initial dataset shape:", df.shape)

# =====================================================
# 3. STANDARDIZE COLUMNS
# =====================================================
# Dataset columns:
# rating, title, text, asin, parent_asin, timestamp, helpful_vote, verified_purchase

df = df.rename(columns={"rating": "overall"})

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# Remove rows where both title and text are empty
df = df[(df["title"].str.strip() != "") | (df["text"].str.strip() != "")]
print("After removing empty reviews:", df.shape)

# =====================================================
# 4. COMBINE TITLE + TEXT
# =====================================================
df["combined_text"] = df["title"] + ". " + df["text"]

# =====================================================
# 5. CLEAN TEXT (SBERT SAFE)
# =====================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"&quot;|&#34;", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["combined_text"].apply(clean_text)

# =====================================================
# 6. REMOVE DUPLICATES
# =====================================================
df = df.drop_duplicates(subset=["clean_text"])
print("After removing duplicates:", df.shape)

# =====================================================
# 7. WEAK SUPERVISION LABEL CREATION
# =====================================================
# 1 â†’ unreliable-like
# 0 â†’ reliable-like
df["label"] = df["overall"].apply(lambda x: 1 if x <= 2 else 0)

# =====================================================
# 8. FINAL DATASET
# =====================================================
final_df = df[
    ["asin", "overall", "clean_text", "label"]
]

print("Final dataset shape:", final_df.shape)

# =====================================================
# 9. SAVE OUTPUT (NEW, NON-CONFLICTING NAME)
# =====================================================
final_df.to_csv(OUTPUT_PATH, index=False)

print("==============================================")
print("âœ… BEAUTY TRAIN PREPROCESSING COMPLETED")
print("ðŸ“ Output saved at:")
print(OUTPUT_PATH)
print("==============================================")
```

## beauty_model\train_classifier_beauty.py

```py
import numpy as np
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from config import EMBEDDINGS_DIR, MODELS_DIR

# =====================================================
# 1. LOAD BEAUTY EMBEDDINGS
# =====================================================
EMB_PATH = EMBEDDINGS_DIR / "beauty_sbert_embeddings.npy"
LABEL_PATH = EMBEDDINGS_DIR / "beauty_labels.npy"

X = np.load(EMB_PATH)
y = np.load(LABEL_PATH)

print("Beauty embeddings loaded:", X.shape)

# =====================================================
# 2. TRAIN-TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 3. TRAIN CLASSIFIER (IMBALANCE-AWARE)
# =====================================================
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

clf.fit(X_train, y_train)

# =====================================================
# 4. SAVE TRAINED BEAUTY MODEL
# =====================================================
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = MODELS_DIR / "logistic_model_beauty.joblib"
joblib.dump(clf, MODEL_PATH)

print("âœ… Beauty model saved at:", MODEL_PATH)

# =====================================================
# 5. EVALUATION
# =====================================================
y_pred = clf.predict(X_test)

print("\n=== BEAUTY CLASSIFICATION REPORT ===\n")
print(classification_report(y_test, y_pred))

print("\n=== BEAUTY CONFUSION MATRIX ===\n")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# 6. SAVE MODEL METRICS (BEAUTY-SPECIFIC)
# =====================================================
report = classification_report(y_test, y_pred, output_dict=True)

metrics = {
    "dataset": "beauty",
    "precision_fake": report["1"]["precision"],
    "recall_fake": report["1"]["recall"],
    "f1_fake": report["1"]["f1-score"],
    "support_fake": report["1"]["support"],
    "accuracy": report["accuracy"],
    "dataset_size": len(y)
}

METRICS_PATH = MODELS_DIR / "model_metrics_beauty.json"
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("âœ… Beauty model metrics saved at:", METRICS_PATH)
```

## .env.example

```example
BACKEND_HOST=0.0.0.0
BACKEND_PORT=5000
FLASK_DEBUG=true
FRONTEND_URL=http://localhost:5173
CORS_ORIGINS=
MODEL_PATH=Models/logistic_model.joblib
MODEL_METRICS_PATH=Models/model_metrics.json
SBERT_MODEL_NAME=all-MiniLM-L6-v2
THRESHOLD_GENUINE=0.30
THRESHOLD_FAKE=0.75
```

## Models\model_metrics.json

```json
{
    "precision_fake": 0.881,
    "recall_fake": 0.895,
    "f1_fake": 0.888,
    "support_fake": 11429,
    "accuracy": 0.887,
    "dataset_size": 114286,
    "balanced_dataset": true
}
```

## Models\model_metrics_beauty.json

```json
{
    "dataset": "beauty",
    "precision_fake": 0.5881488736532811,
    "recall_fake": 0.9016516516516516,
    "f1_fake": 0.7119146413752223,
    "support_fake": 1332.0,
    "accuracy": 0.8756397134083931,
    "dataset_size": 39080
}
```

## All Other Models\train_gru_torch.py

```py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =====================================================
# 1. CONFIG
# =====================================================
BASE_DIR = r"D:\Fake Review Detection"
DATA_PATH = BASE_DIR + r"\Datasets\kindle_reviews_clean.csv"

MAX_LEN = 120
BATCH_SIZE = 64
EPOCHS = 5
EMB_DIM = 100
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =====================================================
# 2. LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)
texts = df["clean_text"].astype(str).tolist()
labels = df["label"].values

# =====================================================
# 3. TOKENIZER + VOCAB
# =====================================================
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

vocab = {"<PAD>": 0, "<UNK>": 1}
for text in texts:
    for word in tokenize(text):
        if word not in vocab:
            vocab[word] = len(vocab)

# =====================================================
# 4. DATASET
# =====================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def encode(self, text):
        tokens = tokenize(text)[:MAX_LEN]
        ids = [vocab.get(w, 1) for w in tokens]
        return ids + [0] * (MAX_LEN - len(ids))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encode(self.texts[idx])),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# =====================================================
# 5. SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

train_loader = DataLoader(
    ReviewDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    ReviewDataset(X_test, y_test),
    batch_size=BATCH_SIZE
)

# =====================================================
# 6. GRU MODEL
# =====================================================
class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), EMB_DIM, padding_idx=0)
        self.gru = nn.GRU(
            EMB_DIM,
            HIDDEN_DIM,
            batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_DIM, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

model = GRUClassifier().to(DEVICE)

# =====================================================
# 7. TRAINING
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training GRU...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f}")

# =====================================================
# 8. EVALUATION
# =====================================================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        preds = torch.argmax(model(x.to(DEVICE)), dim=1)
        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n=== GRU CLASSIFICATION REPORT ===\n")
print(classification_report(y_true, y_pred))
```

## All Other Models\tfidf_nb.py

```py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# =====================================================
# 1. LOAD DATA
# =====================================================
DATA_PATH = r"D:\Fake Review Detection\Datasets\Beauty Products Review Rating Prediction\dataset\beauty_train_clean.csv"

df = pd.read_csv(DATA_PATH)

texts = df["clean_text"].astype(str)
labels = df["label"].astype(int)

print("Total samples:", len(df))

# =====================================================
# 2. TRAIN-TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# =====================================================
# 3. TF-IDF VECTORIZE
# =====================================================
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=5
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =====================================================
# 4. TRAIN NAIVE BAYES
# =====================================================
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# =====================================================
# 5. EVALUATION
# =====================================================
y_pred = clf.predict(X_test_tfidf)

print("\n=== TF-IDF + Naive Bayes ===\n")
print(classification_report(y_test, y_pred))
```

## All Other Models\tempCodeRunnerFile.py

```py
feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# =====================================================
# 1. LOAD DATA
# =====================================================
BASE_DIR = r"D:\Fake Review Detection"
DATA_PATH = os.path.join(BASE_DIR, "All Other Models", "kindle_reviews_augmented_clean.csv")

df = pd.read_csv(DATA_PATH)

texts = df["clean_text"].astype(str)
labels = df["label"].values

print("Total samples:", len(texts))
```

## All Other Models\bow_nb.py

```py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# =====================================================
# 1. LOAD DATA
# =====================================================
DATA_PATH = r"D:\Fake Review Detection\All Other Models\kindle_reviews_augmented_clean.csv"
df = pd.read_csv(DATA_PATH)

texts = df["clean_text"].astype(str)
labels = df["label"]

print("Total samples:", len(df))

# =====================================================
# 2. TRAIN-TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# =====================================================
# 3. BAG-OF-WORDS VECTORIZATION
# =====================================================
vectorizer = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =====================================================
# 4. TRAIN NAIVE BAYES
# =====================================================
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# =====================================================
# 5. EVALUATION
# =====================================================
y_pred = clf.predict(X_test_vec)

print("\n=== Bag-of-Words + Naive Bayes ===\n")
print(classification_report(y_test, y_pred))
```

## frontend\vite.config.ts

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
})
```

## frontend\tsconfig.node.json

```json
{
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.node.tsbuildinfo",
    "target": "ES2023",
    "lib": ["ES2023"],
    "module": "ESNext",
    "types": ["node"],
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "verbatimModuleSyntax": true,
    "moduleDetection": "force",
    "noEmit": true,

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "erasableSyntaxOnly": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  },
  "include": ["vite.config.ts"]
}
```

## frontend\tsconfig.json

```json
{
  "files": [],
  "references": [
    { "path": "./tsconfig.app.json" },
    { "path": "./tsconfig.node.json" }
  ]
}
```

## frontend\tsconfig.app.json

```json
{
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.app.tsbuildinfo",
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "types": ["vite/client"],
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "verbatimModuleSyntax": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "erasableSyntaxOnly": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  },
  "include": ["src"]
}
```

## frontend\src\main.tsx

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

## frontend\src\index.css

```css
:root {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  line-height: 1.5;
  font-weight: 400;
  color: #213547;
  background-color: #ffffff;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  margin: 0;
  min-height: 100vh;
}

h1, h2, h3 {
  margin-top: 0;
}

a {
  color: #007bff;
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

button {
  font-family: inherit;
}
```

## frontend\src\components\Technologies.tsx

```tsx
import React from 'react';
import './Technologies.css';

const Technologies: React.FC = () => {
  const technologies = [
    {
      category: 'Machine Learning & NLP',
      items: [
        'Python 3.x',
        'Sentence Transformers (SBERT)',
        'Scikit-learn',
        'NumPy',
        'Pandas',
        'Joblib (Model Serialization)'
      ]
    },
    {
      category: 'Backend',
      items: [
        'Flask',
        'Flask-CORS',
        'JSON API',
        'Python Virtual Environment'
      ]
    },
    {
      category: 'Frontend',
      items: [
        'React 19.2.0',
        'TypeScript 5.9.3',
        'Vite 7.2.4',
        'CSS3',
        'Responsive Design'
      ]
    },
    {
      category: 'Data & Datasets',
      items: [
        'Amazon Kindle Reviews',
        'CSV Processing',
        'NumPy Arrays (.npy)'
      ]
    }
  ];

  return (
    <section className="technologies" id="technologies">
      <div className="container">
        <h2>Technologies Used</h2>
        <div className="tech-list">
          {technologies.map((tech, index) => (
            <div className="tech-category" key={index}>
              <h3>{tech.category}</h3>
              <ul>
                {tech.items.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Technologies;
```

## frontend\src\components\Technologies.css

```css
.technologies {
  padding: 3rem 0;
  background-color: white;
}

.technologies h2 {
  text-align: center;
  margin-bottom: 2rem;
  color: #007bff;
  font-size: 1.8rem;
  font-weight: bold;
}

.tech-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  max-width: 1000px;
  margin: 0 auto;
}

.tech-category {
  background-color: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  border-left: 4px solid #007bff;
  transition: all 0.3s ease;
}

.tech-category:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.tech-category h3 {
  color: #007bff;
  margin-bottom: 1rem;
  text-align: center;
}

.tech-category ul {
  list-style-type: none;
  padding: 0;
}

.tech-category li {
  padding: 0.5rem 0;
  border-bottom: 1px solid #eee;
  color: #555;
}

.tech-category li:last-child {
  border-bottom: none;
}
```

## frontend\src\components\Team.tsx

```tsx
import React from 'react';
import './Team.css';

const Team: React.FC = () => {
  const teamMembers = [
    { name: 'Chakri K', role: 'Team Leader' },
    { name: 'Delan Pinto', role: 'Project Team Member' },
    { name: 'Hruday Kumar', role: 'Project Team Member' },
    { name: 'Mohammed Mukram', role: 'Project Team Member' }
  ];

  return (
    <section className="team">
      <div className="container">
        <h2>Project Team</h2>
        
        <div className="guide-section">
          <h3>Under the Guidance of</h3>
          <div className="guide-card">
            <p className="guide-name">Dr Sangeetha R</p>
            <p className="guide-title">Faculty Advisor</p>
          </div>
        </div>

        <div className="team-members">
          {teamMembers.map((member, index) => (
            <div className="member" key={index}>
              <h3>{member.name}</h3>
              <p>{member.role}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Team;
```

## frontend\src\components\Team.css

```css
.team {
  padding: 3rem 0;
  background-color: white;
}

.team h2 {
  text-align: center;
  margin-bottom: 3rem;
  color: #333;
  font-size: 2rem;
  font-weight: bold;
}

.guide-section {
  text-align: center;
  margin-bottom: 3rem;
}

.guide-section h3 {
  color: #007bff;
  margin-bottom: 1.5rem;
  font-size: 1.3rem;
}

.guide-card {
  background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
  color: white;
  padding: 2rem;
  border-radius: 8px;
  max-width: 400px;
  margin: 0 auto;
  box-shadow: 0 4px 8px rgba(0, 123, 255, 0.3);
}

.guide-name {
  font-size: 1.3rem;
  font-weight: bold;
  margin: 0 0 0.5rem 0;
}

.guide-title {
  font-size: 0.95rem;
  opacity: 0.9;
  margin: 0;
}

.team-members {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
  max-width: 1000px;
  margin: 0 auto;
}

.member {
  background-color: white;
  padding: 2rem 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  text-align: center;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  border-top: 4px solid #007bff;
}

.member:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.member h3 {
  color: #007bff;
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
  font-weight: bold;
}

.member p {
  color: #555;
  line-height: 1.5;
  font-size: 0.95rem;
  margin: 0.3rem 0;
}

@media (max-width: 768px) {
  .team {
    padding: 2rem 0;
  }

  .team h2 {
    font-size: 1.5rem;
  }

  .guide-card {
    padding: 1.5rem;
  }

  .guide-name {
    font-size: 1.1rem;
  }

  .team-members {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1.5rem;
  }

  .member {
    padding: 1.5rem 1rem;
  }
}
```

## frontend\src\components\Results.tsx

```tsx
import React from 'react';
import './Results.css';

const Results: React.FC = () => {
  return (
    <section className="results-section" id="results">
      <div className="container">
        <div className="results">
          <h2>Results & Evaluation</h2>
          <table className="results-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>BoW + Naive Bayes</td>
                <td>0.86</td>
                <td>0.85</td>
                <td>0.89</td>
                <td>0.87</td>
              </tr>
              <tr>
                <td>TF-IDF + Naive Bayes</td>
                <td>0.88</td>
                <td>0.96</td>
                <td>0.28</td>
                <td>0.44</td>
              </tr>
              <tr>
                <td>GRU (from-scratch embeddings)</td>
                <td>0.85</td>
                <td>0.8218</td>
                <td>0.8029</td>
                <td>0.8323</td>
              </tr>
              <tr className="highlight">
                <td><strong>SBERT + Logistic Regression</strong></td>
                <td><strong>0.89</strong></td>
                <td><strong>0.88</strong></td>
                <td><strong>0.90</strong></td>
                <td><strong>0.89</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};

export default Results;
```

## frontend\src\components\Results.css

```css
.results-section {
  padding: 3rem 0;
  background-color: white;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}

.results {
  background-color: white;
  border-left: 5px solid #007bff;
  padding: 2rem;
  border-radius: 8px;
  text-align: left;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.results h2 {
  font-size: 1.8rem;
  margin-top: 0;
  margin-bottom: 1.5rem;
  font-weight: bold;
  color: #007bff;
  border-bottom: 2px solid #007bff;
  padding-bottom: 0.5rem;
}

.results-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
}

.results-table thead {
  background-color: #f8f9fa;
}

.results-table th {
  padding: 1rem;
  text-align: center;
  font-weight: bold;
  color: #007bff;
  border-bottom: 2px solid #007bff;
  font-size: 0.95rem;
}

.results-table td {
  padding: 0.9rem 1rem;
  text-align: center;
  border-bottom: 1px solid #e0e0e0;
  color: #555;
}

.results-table tbody tr:hover {
  background-color: #f5f5f5;
}

.results-table tbody tr.highlight {
  background-color: #e7f3ff;
  font-weight: bold;
}

.results-table tbody tr.highlight td {
  color: #007bff;
  border-bottom: 2px solid #007bff;
}

@media (max-width: 768px) {
  .results-section {
    padding: 2rem 0;
  }

  .results {
    padding: 1.5rem;
  }

  .results h2 {
    font-size: 1.4rem;
  }

  .results-table th,
  .results-table td {
    padding: 0.7rem 0.5rem;
    font-size: 0.85rem;
  }
}
```

## frontend\src\components\Navigation.tsx

```tsx
import React, { useState } from 'react';
import './Navigation.css';

const Navigation: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setIsOpen(false);
    }
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <span className="brand-text">Fake Review Detection</span>
        </div>

        <button
          className={`hamburger ${isOpen ? 'active' : ''}`}
          onClick={() => setIsOpen(!isOpen)}
        >
          <span></span>
          <span></span>
          <span></span>
        </button>

        <ul className={`nav-menu ${isOpen ? 'active' : ''}`}>
          <li className="nav-item">
            <button onClick={() => scrollToSection('problem-statement')} className="nav-link">
              Problem Statement
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('objectives')} className="nav-link">
              Objectives
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('methodology')} className="nav-link">
              Methodology
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('features')} className="nav-link">
              Key Features
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('technologies')} className="nav-link">
              Technologies Used
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('results')} className="nav-link">
              Results & Evaluation
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('demo')} className="nav-link">
              Demo
            </button>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navigation;
```

## frontend\src\components\Navigation.css

```css
.navbar {
  position: sticky;
  top: 0;
  z-index: 1000;
  background-color: white;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  padding: 0.5rem 0;
}

.navbar-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 60px;
}

.navbar-brand {
  font-weight: bold;
  font-size: 1.2rem;
}

.brand-text {
  color: #007bff;
  font-weight: 700;
}

.nav-menu {
  display: flex;
  list-style: none;
  gap: 0;
  margin: 0;
  padding: 0;
  align-items: center;
}

.nav-item {
  margin: 0;
}

.nav-link {
  background: none;
  border: none;
  color: #333;
  padding: 0.75rem 1rem;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 500;
}

.nav-link:hover {
  color: #007bff;
  background-color: #f0f8ff;
  border-radius: 4px;
}

.hamburger {
  display: none;
  flex-direction: column;
  background: none;
  border: none;
  cursor: pointer;
  gap: 5px;
}

.hamburger span {
  width: 25px;
  height: 3px;
  background-color: #333;
  transition: all 0.3s ease;
  border-radius: 2px;
}

.hamburger.active span:nth-child(1) {
  transform: rotate(45deg) translate(8px, 8px);
}

.hamburger.active span:nth-child(2) {
  opacity: 0;
}

.hamburger.active span:nth-child(3) {
  transform: rotate(-45deg) translate(7px, -7px);
}

@media (max-width: 768px) {
  .hamburger {
    display: flex;
  }

  .nav-menu {
    position: absolute;
    top: 60px;
    left: 0;
    right: 0;
    background-color: white;
    flex-direction: column;
    width: 100%;
    text-align: left;
    transition: 0.3s;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    max-height: 0;
    overflow: hidden;
  }

  .nav-menu.active {
    max-height: 500px;
  }

  .nav-item {
    width: 100%;
  }

  .nav-link {
    display: block;
    width: 100%;
    padding: 1rem;
    text-align: left;
  }

  .nav-link:hover {
    background-color: #f0f8ff;
    border-radius: 0;
  }

  .navbar-container {
    height: auto;
  }
}

/* Smooth scroll behavior with offset for sticky nav */
html {
  scroll-behavior: smooth;
  scroll-padding-top: 70px;
}
```

## frontend\src\components\Methodology.tsx

```tsx
import React, { useState } from 'react';
import './Methodology.css';

const Methodology: React.FC = () => {
  const [hoveredStep, setHoveredStep] = useState<string | null>(null);

  const pipelineSteps = [
    { id: 'dataset', label: 'Review\nDataset', icon: 'ðŸ“Š', tooltip: 'Collect reviews from e-commerce platforms' },
    { id: 'preprocessing', label: 'Data\nPreprocessing', icon: 'ðŸ§¹', tooltip: 'Clean, normalize, tokenize text' },
    { id: 'features', label: 'Feature\nExtraction', icon: 'ðŸ”', tooltip: 'BoW, TF-IDF, SBERT Embeddings' },
    { id: 'training', label: 'Model\nTraining', icon: 'âš™ï¸', tooltip: '4 models trained independently' },
    { id: 'similarity', label: 'Semantic\nSimilarity', icon: 'ðŸ”—', tooltip: 'Cosine similarity analysis' },
    { id: 'evaluation', label: 'Model\nEvaluation', icon: 'ðŸ“ˆ', tooltip: 'Accuracy, Precision, Recall, F1' },
    { id: 'prediction', label: 'Fake/Genuine\nPrediction', icon: 'âœ…', tooltip: 'Web app delivers prediction' }
  ];

  const features = [
    {
      icon: 'ðŸŸ¦',
      title: 'Bag of Words',
      description: 'Word frequency-based representation'
    },
    {
      icon: 'ðŸŸ¨',
      title: 'TF-IDF',
      description: 'Weighted word importance'
    },
    {
      icon: 'ðŸŸ©',
      title: 'SBERT Embeddings',
      description: 'Context-aware semantic vectors'
    }
  ];

  const models = [
    {
      name: 'BoW + NB',
      strength: 'Simple baseline',
      icon: 'ðŸ“¦',
      color: '#FF6B6B'
    },
    {
      name: 'TF-IDF + NB',
      strength: 'Better word weighting',
      icon: 'âš–ï¸',
      color: '#4ECDC4'
    },
    {
      name: 'GRU',
      strength: 'Sequential learning',
      icon: 'ðŸ§ ',
      color: '#95E1D3'
    },
    {
      name: 'SBERT + LR',
      strength: 'Best semantic understanding',
      icon: 'â­',
      color: '#FFD93D',
      isFinal: true
    }
  ];

  const metrics = [
    { icon: 'ðŸ“Š', label: 'Accuracy', key: 'accuracy' },
    { icon: 'ðŸŽ¯', label: 'Precision', key: 'precision' },
    { icon: 'ðŸ”', label: 'Recall', key: 'recall' },
    { icon: 'âš–ï¸', label: 'F1-Score', key: 'f1' },
    { icon: 'ðŸ§©', label: 'Confusion Matrix', key: 'matrix' }
  ];

  return (
    <section className="methodology-visual" id="methodology">
      <div className="container">
        <h2>Methodology</h2>
        <p className="subtitle">End-to-End Fake Review Detection Pipeline</p>

        {/* Main Pipeline */}
        <div className="pipeline-section">
          <h3>System Pipeline</h3>
          <div className="pipeline">
            {pipelineSteps.map((step, index) => (
              <React.Fragment key={step.id}>
                <div
                  className={`pipeline-step ${hoveredStep === step.id ? 'active' : ''}`}
                  onMouseEnter={() => setHoveredStep(step.id)}
                  onMouseLeave={() => setHoveredStep(null)}
                  title={step.tooltip}
                >
                  <div className="step-icon">{step.icon}</div>
                  <div className="step-label">{step.label}</div>
                  {hoveredStep === step.id && (
                    <div className="tooltip">{step.tooltip}</div>
                  )}
                </div>
                {index < pipelineSteps.length - 1 && (
                  <div className="pipeline-arrow">â†’</div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Feature Extraction */}
        <div className="features-section">
          <h3>Feature Extraction Techniques</h3>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="feature-icon">{feature.icon}</div>
                <h4>{feature.title}</h4>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Model Comparison */}
        <div className="models-section">
          <h3>Model Comparison</h3>
          <div className="models-grid">
            {models.map((model, index) => (
              <div key={index} className={`model-card ${model.isFinal ? 'final-model' : ''}`}
                   style={{ borderTopColor: model.color }}>
                <div className="model-icon">{model.icon}</div>
                <h4>{model.name}</h4>
                <p className="model-strength">{model.strength}</p>
                {model.isFinal && <div className="final-badge">FINAL MODEL</div>}
              </div>
            ))}
          </div>
        </div>

        {/* Semantic Similarity */}
        <div className="similarity-section">
          <h3>Semantic Similarity Analysis</h3>
          <div className="similarity-diagram">
            <div className="review-box">
              <p>Review A</p>
            </div>
            <div className="similarity-connection">
              <div className="similarity-line"></div>
              <div className="similarity-label">High Cosine Similarity</div>
            </div>
            <div className="review-box">
              <p>Review B</p>
            </div>
            <div className="similarity-insight">â†’ Possible Fake Review Detected</div>
          </div>
          <p className="similarity-caption">High semantic similarity indicates duplicated or manipulated reviews</p>
        </div>

        {/* Evaluation Metrics */}
        <div className="metrics-section">
          <h3>Model Evaluation Metrics</h3>
          <div className="metrics-grid">
            {metrics.map((metric, index) => (
              <div key={index} className="metric-badge">
                <div className="metric-icon">{metric.icon}</div>
                <p>{metric.label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Deployment */}
        <div className="deployment-section">
          <h3>Web Application Deployment</h3>
          <div className="deployment-flow">
            <div className="deployment-box">
              <p>User Input</p>
              <p className="small">Review Text</p>
            </div>
            <div className="deployment-arrow">â†“</div>
            <div className="deployment-box highlight">
              <p>SBERT + Logistic Regression</p>
              <p className="small">Final Model</p>
            </div>
            <div className="deployment-arrow">â†“</div>
            <div className="deployment-box result">
              <p>ðŸŽ¯ Prediction Output</p>
              <p className="small">Fake / Genuine</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Methodology;
```

## frontend\src\components\Methodology.css

```css
.methodology-visual {
  padding: 3rem 0;
  background-color: #f8f9fa;
}

.methodology-visual .container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}

.methodology-visual h2 {
  font-size: 1.8rem;
  color: #007bff;
  margin-top: 0;
  margin-bottom: 1.5rem;
  font-weight: bold;
  border-bottom: 2px solid #007bff;
  padding-bottom: 0.5rem;
  text-align: left;
}

.methodology-visual .subtitle {
  text-align: left;
  color: #666;
  font-size: 1rem;
  margin-bottom: 2rem;
  margin-top: -1rem;
}

.methodology-visual h3 {
  font-size: 1.3rem;
  color: #007bff;
  margin: 2rem 0 1.5rem 0;
  text-align: left;
  font-weight: 600;
}

/* ==================== PIPELINE SECTION ==================== */
.pipeline-section {
  background: white;
  border-left: 5px solid #007bff;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
  overflow-x: auto;
}

.pipeline {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
  min-height: 150px;
}

.pipeline-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 100px;
  height: 120px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 12px;
  padding: 1rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  font-weight: 600;
  font-size: 0.9rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.pipeline-step:hover {
  transform: translateY(-8px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
  background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

.pipeline-step.active {
  transform: scale(1.1);
  box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
}

.step-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.step-label {
  font-size: 0.85rem;
  line-height: 1.3;
  white-space: pre-wrap;
}

.tooltip {
  position: absolute;
  bottom: -50px;
  left: 50%;
  transform: translateX(-50%);
  background: #333;
  color: white;
  padding: 0.7rem 1rem;
  border-radius: 8px;
  font-size: 0.8rem;
  white-space: nowrap;
  z-index: 10;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.pipeline-arrow {
  font-size: 1.5rem;
  color: #667eea;
  font-weight: bold;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* ==================== FEATURES SECTION ==================== */
.features-section {
  background: white;
  border-left: 5px solid #007bff;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
}

.feature-card {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  padding: 2rem;
  border-radius: 12px;
  text-align: center;
  transition: all 0.3s ease;
  border-left: 5px solid #667eea;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
  border-left-color: #764ba2;
}

.feature-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.feature-card h4 {
  font-size: 1.3rem;
  color: #333;
  margin-bottom: 0.5rem;
  font-weight: 600;
}

.feature-card p {
  color: #666;
  font-size: 0.95rem;
}

/* ==================== MODELS SECTION ==================== */
.models-section {
  background: white;
  border-left: 5px solid #007bff;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
}

.model-card {
  background: white;
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  transition: all 0.3s ease;
  border-top: 6px solid #667eea;
  position: relative;
}

.model-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
  border-top-color: #764ba2;
}

.model-card.final-model {
  background: linear-gradient(135deg, #fff5e1 0%, #fffbf0 100%);
  border-top-color: #FFD93D;
  box-shadow: 0 8px 20px rgba(255, 217, 61, 0.2);
}

.model-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
}

.model-card h4 {
  font-size: 1.2rem;
  color: #333;
  margin-bottom: 0.5rem;
  font-weight: 700;
}

.model-strength {
  color: #666;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}

.final-badge {
  display: inline-block;
  background: #FFD93D;
  color: #333;
  padding: 0.4rem 0.8rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 700;
  margin-top: 0.5rem;
  letter-spacing: 1px;
}

/* ==================== SIMILARITY SECTION ==================== */
.similarity-section {
  background: white;
  border-left: 5px solid #007bff;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.similarity-diagram {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1.5rem;
  margin: 2rem 0;
  flex-wrap: wrap;
  padding: 2rem;
  background: #f9f9f9;
  border-radius: 12px;
}

.review-box {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1.5rem 2rem;
  border-radius: 10px;
  font-weight: 600;
  min-width: 120px;
  text-align: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.review-box p {
  margin: 0;
  font-size: 1rem;
}

.similarity-connection {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.similarity-line {
  width: 80px;
  height: 3px;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  position: relative;
}

.similarity-line::before,
.similarity-line::after {
  content: '';
  position: absolute;
  top: -4px;
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
}

.similarity-line::before {
  left: -8px;
  border-top: 6px solid #667eea;
}

.similarity-line::after {
  right: -8px;
  border-top: 6px solid #764ba2;
}

.similarity-label {
  font-size: 0.9rem;
  color: #667eea;
  font-weight: 600;
  white-space: nowrap;
}

.similarity-insight {
  font-weight: 600;
  color: #e74c3c;
  font-size: 1rem;
  margin-left: auto;
  white-space: nowrap;
}

.similarity-caption {
  text-align: center;
  color: #666;
  font-size: 0.95rem;
  margin-top: 1rem;
  font-style: italic;
}

/* ==================== METRICS SECTION ==================== */
.metrics-section {
  background: white;
  border-left: 5px solid #007bff;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
}

.metric-badge {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 2rem 1rem;
  border-radius: 12px;
  text-align: center;
  transition: all 0.3s ease;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.metric-badge:hover {
  transform: scale(1.08);
  box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
}

.metric-icon {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

.metric-badge p {
  font-size: 0.95rem;
  margin: 0;
  font-weight: 600;
}

/* ==================== DEPLOYMENT SECTION ==================== */
.deployment-section {
  background: white;
  border-left: 5px solid #007bff;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.deployment-flow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  margin: 2rem 0;
  flex-wrap: wrap;
  padding: 2rem;
  background: #f9f9f9;
  border-radius: 12px;
}

.deployment-box {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1.5rem 2rem;
  border-radius: 10px;
  text-align: center;
  font-weight: 600;
  min-width: 150px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transition: all 0.3s ease;
}

.deployment-box:hover {
  transform: scale(1.05);
}

.deployment-box.highlight {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  min-width: 180px;
}

.deployment-box.result {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  min-width: 160px;
}

.deployment-box p {
  margin: 0;
}

.deployment-box p:first-child {
  font-size: 1rem;
  margin-bottom: 0.3rem;
}

.deployment-box p.small {
  font-size: 0.8rem;
  opacity: 0.9;
}

.deployment-arrow {
  font-size: 1.5rem;
  color: #667eea;
  font-weight: bold;
  animation: pulse 2s infinite;
}

/* ==================== RESPONSIVE DESIGN ==================== */
@media (max-width: 768px) {
  .methodology-visual h2 {
    font-size: 2rem;
  }

  .methodology-visual h3 {
    font-size: 1.5rem;
  }

  .pipeline {
    gap: 0.5rem;
  }

  .pipeline-step {
    min-width: 80px;
    height: 100px;
    font-size: 0.8rem;
  }

  .step-icon {
    font-size: 1.5rem;
  }

  .features-grid,
  .models-grid,
  .metrics-grid {
    grid-template-columns: 1fr;
  }

  .similarity-diagram {
    flex-direction: column;
    gap: 1rem;
  }

  .similarity-line {
    width: 3px;
    height: 40px;
    transform: rotate(90deg);
  }

  .similarity-insight {
    margin-left: 0;
  }

  .deployment-flow {
    flex-direction: column;
    gap: 0.5rem;
  }

  .tooltip {
    bottom: -60px;
    font-size: 0.7rem;
  }
}

@media (max-width: 480px) {
  .methodology-visual .container {
    padding: 0 1rem;
  }

  .pipeline-section,
  .features-section,
  .models-section,
  .similarity-section,
  .metrics-section,
  .deployment-section {
    padding: 1.5rem 1rem;
  }

  .methodology-visual h2 {
    font-size: 1.5rem;
  }

  .methodology-visual h3 {
    font-size: 1.2rem;
  }

  .features-grid,
  .models-grid,
  .metrics-grid {
    gap: 1rem;
  }

  .pipeline-step {
    min-width: 70px;
    height: 90px;
    font-size: 0.7rem;
  }
}
```

## frontend\src\components\Header.tsx

```tsx
import React from 'react';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="container">
        <h1 className="title">Detection of Fake Reviews Using Semantic Similarity and Natural Language Processing Techniques</h1>
        <div className="problem-statement" id="problem-statement">
          <h2>Problem Statement</h2>
          <p>
            Online reviews play a crucial role in shaping consumer decisions across ecommerce, travel, food delivery, and service platforms. However, the growing presence of fake reviews that mislead potential customers poses a serious challenge to the credibility of these platforms. Traditional review moderation systems often rely on manual checks or simple keyword filtering, which are ineffective in identifying fake reviews.
          </p>
          <p>
            There is a need for an intelligent system that can automatically detect fake reviews by analyzing their semantic content and writing patterns. This project aims to develop a solution that uses Natural Language Processing (NLP) techniques such as semantic similarity analysis to evaluate the authenticity of reviews. By comparing reviews across products and users and identifying unnatural repetition, emotional exaggeration, or linguistic anomalies, the system can detect potentially fake reviews. This will enhance the trustworthiness and reliability of online review platforms.
          </p>
        </div>
        <div className="objectives" id="objectives">
          <h2>Objectives</h2>
          <ul>
            <li>To design and develop an intelligent system that automatically detects fake reviews using Natural Language Processing techniques.</li>
            <li>To apply semantic similarity analysis for identifying repetitive, exaggerated, or semantically inconsistent reviews across products and users.</li>
            <li>To extract linguistic and sentiment features (e.g., emotional tone, exaggeration, unnatural patterns) that help distinguish between genuine and deceptive reviews.</li>
            <li>Using only the S-BERT and Logistic regression (for classification).</li>
            <li>To improve detection accuracy and reduce false positives by integrating semantic similarity with advanced NLP-based feature extraction.</li>
          </ul>
        </div>
      </div>
    </header>
  );
};

export default Header;
```

## frontend\src\components\Header.css

```css
.header {
  background-color: white;
  color: #333;
  padding: 3rem 0;
  text-align: center;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}

.title {
  background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
  color: white;
  font-size: 2.5rem;
  margin-bottom: 2rem;
  font-weight: bold;
  line-height: 1.2;
  padding: 2rem 1rem;
  border-radius: 8px;
}

.subtitle {
  font-size: 1.2rem;
  opacity: 0.9;
}

.problem-statement {
  background-color: white;
  border-left: 5px solid #007bff;
  padding: 2rem;
  margin-top: 2rem;
  border-radius: 8px;
  text-align: left;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.problem-statement h2 {
  font-size: 1.8rem;
  margin-top: 0;
  margin-bottom: 1.5rem;
  font-weight: bold;
  color: #007bff;
  border-bottom: 2px solid #007bff;
  padding-bottom: 0.5rem;
}

.problem-statement p {
  font-size: 1rem;
  line-height: 1.8;
  margin-bottom: 1.2rem;
  text-align: justify;
  color: #555;
}

.problem-statement p:last-child {
  margin-bottom: 0;
}

.objectives {
  background-color: white;
  border-left: 5px solid #007bff;
  padding: 2rem;
  margin-top: 2rem;
  border-radius: 8px;
  text-align: left;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.objectives h2 {
  font-size: 1.8rem;
  margin-top: 0;
  margin-bottom: 1.5rem;
  font-weight: bold;
  color: #007bff;
  border-bottom: 2px solid #007bff;
  padding-bottom: 0.5rem;
}

.objectives ul {
  list-style-position: outside;
  padding-left: 2rem;
  margin: 0;
}

.objectives li {
  font-size: 1rem;
  line-height: 1.8;
  margin-bottom: 1rem;
  text-align: justify;
  color: #555;
}

.objectives li:last-child {
  margin-bottom: 0;
}

.methodology {
  background-color: white;
  border-left: 5px solid #007bff;
  padding: 2rem;
  margin-top: 2rem;
  border-radius: 8px;
  text-align: left;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.methodology h2 {
  font-size: 1.8rem;
  margin-top: 0;
  margin-bottom: 1.5rem;
  font-weight: bold;
  color: #007bff;
  border-bottom: 2px solid #007bff;
  padding-bottom: 0.5rem;
}

.methodology > p {
  font-size: 1rem;
  line-height: 1.8;
  color: #555;
  margin-bottom: 1.5rem;
}

.methodology-section {
  margin-bottom: 1.5rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid #e0e0e0;
}

.methodology-section:last-child {
  border-bottom: none;
  padding-bottom: 0;
  margin-bottom: 0;
}

.methodology-section h3 {
  font-size: 1.2rem;
  margin-top: 0;
  margin-bottom: 0.8rem;
  font-weight: bold;
  color: #333;
}

.methodology-section h4 {
  font-size: 1rem;
  margin-top: 0.8rem;
  margin-bottom: 0.5rem;
  font-weight: bold;
  color: #555;
}

.methodology-section p {
  font-size: 1rem;
  line-height: 1.8;
  color: #555;
  text-align: justify;
  margin-bottom: 0.8rem;
}

.methodology-section ul {
  list-style-position: outside;
  padding-left: 2rem;
  margin: 0.8rem 0;
}

.methodology-section li {
  font-size: 0.95rem;
  line-height: 1.6;
  color: #555;
  margin-bottom: 0.5rem;
}

@media (max-width: 768px) {
  .header h1 {
    font-size: 1.8rem;
  }
  
  .problem-statement {
    text-align: left;
    padding: 1.5rem;
  }
  
  .problem-statement h2 {
    font-size: 1.4rem;
  }
  
  .problem-statement p {
    font-size: 0.95rem;
  }
}
```

## frontend\src\components\Footer.tsx

```tsx
import React from 'react';
import './Footer.css';

const Footer: React.FC = () => {
  return (
    <footer className="footer">
      <div className="container">
        <p>&copy; 2026 Fake Review Detection Project. All rights reserved.</p>
        <p>College Engineering Project - Department of Artificial Intelligence and Data Science - GITAM University</p>
      </div>
    </footer>
  );
};

export default Footer;
```

## frontend\src\components\Footer.css

```css
.footer {
  background-color: #333;
  color: white;
  padding: 2rem 0;
  text-align: center;
}

.footer p {
  margin: 0.5rem 0;
  opacity: 0.8;
}
```

## frontend\src\components\Features.tsx

```tsx
import React from 'react';
import './Features.css';

const Features: React.FC = () => {
  return (
    <section className="features">
      <div className="container">
        <h2>Key Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <h3>Machine Learning Model</h3>
            <p>Trained classifier using SBERT embeddings for accurate fake review detection.</p>
          </div>
          <div className="feature-card">
            <h3>Semantic Analysis</h3>
            <p>Advanced semantic similarity analysis to identify review patterns and anomalies.</p>
          </div>
          <div className="feature-card">
            <h3>Data Preprocessing</h3>
            <p>Comprehensive preprocessing pipeline for cleaning and preparing review data.</p>
          </div>
          <div className="feature-card">
            <h3>High Accuracy</h3>
            <p>Achieves high precision and recall in detecting deceptive reviews.</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;
```

## frontend\src\components\Features.css

```css
.features {
  padding: 3rem 0;
  background-color: white;
}

.features h2 {
  text-align: center;
  margin-bottom: 2rem;
  color: #333;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  max-width: 1000px;
  margin: 0 auto;
}

.feature-card {
  background-color: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  text-align: center;
}

.feature-card h3 {
  color: #007bff;
  margin-bottom: 1rem;
}

.feature-card p {
  color: #555;
  line-height: 1.5;
}
```

## frontend\src\components\Demo.tsx

```tsx
import React, { useState } from 'react';
import './Demo.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000';

interface PredictionResult {
  review: string;
  prediction: string;
  fake_probability: number;
  confidence_percent: number;
}

const Demo: React.FC = () => {
  const [review, setReview] = useState('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to predict');
      }
      
      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="demo" id="demo">
      <div className="container">
        <h2>Try the Demo</h2>
        <p>Enter a review text below to check if it's likely to be fake or genuine.</p>
        <form onSubmit={handleSubmit} className="demo-form">
          <textarea
            value={review}
            onChange={(e) => setReview(e.target.value)}
            placeholder="Enter your review text here..."
            rows={5}
            required
            disabled={loading}
          />
          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze Review'}
          </button>
        </form>
        {loading && <div className="loading">Analyzing review...</div>}
        {error && <div className="error">Error: {error}</div>}
        {result && (
          <div className="result">
            <h3>Prediction Result</h3>
            <div className="prediction-card">
              <div className={`prediction-status ${result.prediction.toLowerCase().replace(' ', '-')}`}>
                <span className="status-emoji">
                  {result.prediction.toLowerCase() === 'likely genuine' && 'ðŸ˜Š'}
                  {result.prediction.toLowerCase() === 'likely fake' && 'ðŸ˜ '}
                  {result.prediction.toLowerCase() === 'uncertain' && 'ðŸ¤”'}
                </span>
                <strong>{result.prediction}</strong>
              </div>
              <div className="prediction-details">
                <p><strong>Fake Probability:</strong> {result.fake_probability.toFixed(4)} ({result.confidence_percent.toFixed(2)}%)</p>
                <p><strong>Confidence:</strong> {result.confidence_percent > 50 ? 'High' : result.confidence_percent > 30 ? 'Medium' : 'Low'}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default Demo;
```

## frontend\src\components\Demo.css

```css
.demo {
  padding: 0;
  background-color: white;
}

.demo .container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 3rem 2rem;
  border-left: 5px solid #007bff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.demo h2 {
  text-align: center;
  margin-bottom: 1rem;
  color: #007bff;
  font-size: 1.8rem;
  font-weight: bold;
  border-bottom: 2px solid #007bff;
  padding-bottom: 0.5rem;
}

.demo p {
  text-align: center;
  color: #555;
  margin-bottom: 2rem;
}

.demo-form {
  max-width: 600px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.demo-form textarea {
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  resize: vertical;
}

.submit-btn {
  background-color: #007bff;
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 4px;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.3s;
}

.submit-btn:hover {
  background-color: #0056b3;
}

.submit-btn:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}

.loading {
  margin-top: 2rem;
  text-align: center;
  color: #007bff;
  font-weight: bold;
  padding: 1rem;
}

.error {
  margin-top: 2rem;
  text-align: center;
  color: #dc3545;
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  border-radius: 4px;
  padding: 1rem;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

.result {
  margin-top: 2rem;
  text-align: center;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

.result h3 {
  color: #333;
  margin-bottom: 1.5rem;
}

.prediction-card {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  color: #333;
  border-radius: 8px;
  padding: 2rem;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
  border: 2px solid #e0e6ed;
}

.prediction-status {
  font-size: 1.5rem;
  font-weight: bold;
  padding: 1rem;
  border-radius: 6px;
  margin-bottom: 1rem;
  background-color: rgba(100, 100, 100, 0.1);
}

.prediction-status.likely-genuine {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: white;
  box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
}

.prediction-status.likely-fake {
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
  color: white;
  box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3);
}

.prediction-status.uncertain {
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  color: white;
  box-shadow: 0 4px 10px rgba(245, 158, 11, 0.3);
}

.prediction-details {
  text-align: left;
  background-color: rgba(255, 255, 255, 0.7);
  padding: 1rem;
  border-radius: 6px;
  border: 1px solid #e0e6ed;
}

.prediction-details p {
  margin: 0.5rem 0;
  color: #333;
}

.prediction-details strong {
  color: #1f2937;
}

/* Hover Effects and Animations */
.prediction-card {
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.prediction-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 25px rgba(0, 0, 0, 0.2);
}

.status-emoji {
  display: inline-block;
  font-size: 2rem;
  margin-right: 0.5rem;
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-10px);
  }
}

.prediction-status:hover .status-emoji {
  animation: bounce 0.6s ease-in-out;
}

@keyframes bounce {
  0%, 100% {
    transform: translateY(0px);
  }
  25% {
    transform: translateY(-15px);
  }
  50% {
    transform: translateY(0px);
  }
  75% {
    transform: translateY(-8px);
  }
}
```

## frontend\src\components\About.tsx

```tsx
import React from 'react';
import './About.css';

const About: React.FC = () => {
  return (
    <section className="about">
      <div className="container">
        <h2>About the Project</h2>
        <p>
          This project aims to develop a robust system for detecting fake reviews in e-commerce platforms,
          particularly focusing on Amazon Kindle Store reviews. By leveraging machine learning techniques
          and natural language processing, we analyze review patterns to distinguish between genuine and
          deceptive feedback.
        </p>
        <p>
          The system uses SBERT embeddings for semantic similarity analysis and a trained classifier
          to predict the authenticity of reviews with high accuracy.
        </p>
      </div>
    </section>
  );
};

export default About;
```

## frontend\src\components\About.css

```css
.about {
  padding: 3rem 0;
  background-color: #f8f9fa;
}

.about h2 {
  text-align: center;
  margin-bottom: 2rem;
  color: #333;
}

.about p {
  font-size: 1.1rem;
  line-height: 1.6;
  color: #555;
  max-width: 800px;
  margin: 0 auto 1.5rem;
}
```

## frontend\src\assets\react.svg

```svg
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="35.93" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 228"><path fill="#00D8FF" d="M210.483 73.824a171.49 171.49 0 0 0-8.24-2.597c.465-1.9.893-3.777 1.273-5.621c6.238-30.281 2.16-54.676-11.769-62.708c-13.355-7.7-35.196.329-57.254 19.526a171.23 171.23 0 0 0-6.375 5.848a155.866 155.866 0 0 0-4.241-3.917C100.759 3.829 77.587-4.822 63.673 3.233C50.33 10.957 46.379 33.89 51.995 62.588a170.974 170.974 0 0 0 1.892 8.48c-3.28.932-6.445 1.924-9.474 2.98C17.309 83.498 0 98.307 0 113.668c0 15.865 18.582 31.778 46.812 41.427a145.52 145.52 0 0 0 6.921 2.165a167.467 167.467 0 0 0-2.01 9.138c-5.354 28.2-1.173 50.591 12.134 58.266c13.744 7.926 36.812-.22 59.273-19.855a145.567 145.567 0 0 0 5.342-4.923a168.064 168.064 0 0 0 6.92 6.314c21.758 18.722 43.246 26.282 56.54 18.586c13.731-7.949 18.194-32.003 12.4-61.268a145.016 145.016 0 0 0-1.535-6.842c1.62-.48 3.21-.974 4.76-1.488c29.348-9.723 48.443-25.443 48.443-41.52c0-15.417-17.868-30.326-45.517-39.844Zm-6.365 70.984c-1.4.463-2.836.91-4.3 1.345c-3.24-10.257-7.612-21.163-12.963-32.432c5.106-11 9.31-21.767 12.459-31.957c2.619.758 5.16 1.557 7.61 2.4c23.69 8.156 38.14 20.213 38.14 29.504c0 9.896-15.606 22.743-40.946 31.14Zm-10.514 20.834c2.562 12.94 2.927 24.64 1.23 33.787c-1.524 8.219-4.59 13.698-8.382 15.893c-8.067 4.67-25.32-1.4-43.927-17.412a156.726 156.726 0 0 1-6.437-5.87c7.214-7.889 14.423-17.06 21.459-27.246c12.376-1.098 24.068-2.894 34.671-5.345a134.17 134.17 0 0 1 1.386 6.193ZM87.276 214.515c-7.882 2.783-14.16 2.863-17.955.675c-8.075-4.657-11.432-22.636-6.853-46.752a156.923 156.923 0 0 1 1.869-8.499c10.486 2.32 22.093 3.988 34.498 4.994c7.084 9.967 14.501 19.128 21.976 27.15a134.668 134.668 0 0 1-4.877 4.492c-9.933 8.682-19.886 14.842-28.658 17.94ZM50.35 144.747c-12.483-4.267-22.792-9.812-29.858-15.863c-6.35-5.437-9.555-10.836-9.555-15.216c0-9.322 13.897-21.212 37.076-29.293c2.813-.98 5.757-1.905 8.812-2.773c3.204 10.42 7.406 21.315 12.477 32.332c-5.137 11.18-9.399 22.249-12.634 32.792a134.718 134.718 0 0 1-6.318-1.979Zm12.378-84.26c-4.811-24.587-1.616-43.134 6.425-47.789c8.564-4.958 27.502 2.111 47.463 19.835a144.318 144.318 0 0 1 3.841 3.545c-7.438 7.987-14.787 17.08-21.808 26.988c-12.04 1.116-23.565 2.908-34.161 5.309a160.342 160.342 0 0 1-1.76-7.887Zm110.427 27.268a347.8 347.8 0 0 0-7.785-12.803c8.168 1.033 15.994 2.404 23.343 4.08c-2.206 7.072-4.956 14.465-8.193 22.045a381.151 381.151 0 0 0-7.365-13.322Zm-45.032-43.861c5.044 5.465 10.096 11.566 15.065 18.186a322.04 322.04 0 0 0-30.257-.006c4.974-6.559 10.069-12.652 15.192-18.18ZM82.802 87.83a323.167 323.167 0 0 0-7.227 13.238c-3.184-7.553-5.909-14.98-8.134-22.152c7.304-1.634 15.093-2.97 23.209-3.984a321.524 321.524 0 0 0-7.848 12.897Zm8.081 65.352c-8.385-.936-16.291-2.203-23.593-3.793c2.26-7.3 5.045-14.885 8.298-22.6a321.187 321.187 0 0 0 7.257 13.246c2.594 4.48 5.28 8.868 8.038 13.147Zm37.542 31.03c-5.184-5.592-10.354-11.779-15.403-18.433c4.902.192 9.899.29 14.978.29c5.218 0 10.376-.117 15.453-.343c-4.985 6.774-10.018 12.97-15.028 18.486Zm52.198-57.817c3.422 7.8 6.306 15.345 8.596 22.52c-7.422 1.694-15.436 3.058-23.88 4.071a382.417 382.417 0 0 0 7.859-13.026a347.403 347.403 0 0 0 7.425-13.565Zm-16.898 8.101a358.557 358.557 0 0 1-12.281 19.815a329.4 329.4 0 0 1-23.444.823c-7.967 0-15.716-.248-23.178-.732a310.202 310.202 0 0 1-12.513-19.846h.001a307.41 307.41 0 0 1-10.923-20.627a310.278 310.278 0 0 1 10.89-20.637l-.001.001a307.318 307.318 0 0 1 12.413-19.761c7.613-.576 15.42-.876 23.31-.876H128c7.926 0 15.743.303 23.354.883a329.357 329.357 0 0 1 12.335 19.695a358.489 358.489 0 0 1 11.036 20.54a329.472 329.472 0 0 1-11 20.722Zm22.56-122.124c8.572 4.944 11.906 24.881 6.52 51.026c-.344 1.668-.73 3.367-1.15 5.09c-10.622-2.452-22.155-4.275-34.23-5.408c-7.034-10.017-14.323-19.124-21.64-27.008a160.789 160.789 0 0 1 5.888-5.4c18.9-16.447 36.564-22.941 44.612-18.3ZM128 90.808c12.625 0 22.86 10.235 22.86 22.86s-10.235 22.86-22.86 22.86s-22.86-10.235-22.86-22.86s10.235-22.86 22.86-22.86Z"></path></svg>
```

## frontend\src\App.tsx

```tsx
import Navigation from './components/Navigation';
import Header from './components/Header';
import Methodology from './components/Methodology';
import Features from './components/Features';
import Technologies from './components/Technologies';
import Results from './components/Results';
import Demo from './components/Demo';
import Team from './components/Team';
import Footer from './components/Footer';
import './App.css';

function App() {
  return (
    <div className="App">
      <Navigation />
      <div id="header">
        <Header />
      </div>
      <div id="methodology">
        <Methodology />
        <Features />
      </div>
      <div id="technologies">
        <Technologies />
      </div>
      <div id="results">
        <Results />
      </div>
      <div id="demo">
        <Demo />
      </div>
      <Team />
      <Footer />
    </div>
  );
}

export default App;
```

## frontend\src\App.css

```css
.App {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  line-height: 1.6;
  color: #333;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
}
```

## frontend\index.html

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Detection of Fake Reviews Using Semantic Similarity and NLP</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

## frontend\eslint.config.js

```js
import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
  },
])
```

## frontend\.env.example

```example
VITE_API_BASE_URL=http://localhost:5000
```

## frontend\package-lock.json

```json
{
  "name": "frontend",
  "version": "0.0.0",
  "lockfileVersion": 3,
  "requires": true,
  "packages": {
    "": {
      "name": "frontend",
      "version": "0.0.0",
      "dependencies": {
        "react": "^19.2.0",
        "react-dom": "^19.2.0"
      },
      "devDependencies": {
        "@eslint/js": "^9.39.1",
        "@types/node": "^24.10.1",
        "@types/react": "^19.2.5",
        "@types/react-dom": "^19.2.3",
        "@vitejs/plugin-react": "^5.1.1",
        "eslint": "^9.39.1",
        "eslint-plugin-react-hooks": "^7.0.1",
        "eslint-plugin-react-refresh": "^0.4.24",
        "globals": "^16.5.0",
        "typescript": "~5.9.3",
        "typescript-eslint": "^8.46.4",
        "vite": "^7.2.4"
      }
    },
    "node_modules/@babel/code-frame": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/code-frame/-/code-frame-7.28.6.tgz",
      "integrity": "sha512-JYgintcMjRiCvS8mMECzaEn+m3PfoQiyqukOMCCVQtoJGYJw8j/8LBJEiqkHLkfwCcs74E3pbAUFNg7d9VNJ+Q==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/helper-validator-identifier": "^7.28.5",
        "js-tokens": "^4.0.0",
        "picocolors": "^1.1.1"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/compat-data": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/compat-data/-/compat-data-7.28.6.tgz",
      "integrity": "sha512-2lfu57JtzctfIrcGMz992hyLlByuzgIk58+hhGCxjKZ3rWI82NnVLjXcaTqkI2NvlcvOskZaiZ5kjUALo3Lpxg==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/core": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/core/-/core-7.28.6.tgz",
      "integrity": "sha512-H3mcG6ZDLTlYfaSNi0iOKkigqMFvkTKlGUYlD8GW7nNOYRrevuA46iTypPyv+06V3fEmvvazfntkBU34L0azAw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/code-frame": "^7.28.6",
        "@babel/generator": "^7.28.6",
        "@babel/helper-compilation-targets": "^7.28.6",
        "@babel/helper-module-transforms": "^7.28.6",
        "@babel/helpers": "^7.28.6",
        "@babel/parser": "^7.28.6",
        "@babel/template": "^7.28.6",
        "@babel/traverse": "^7.28.6",
        "@babel/types": "^7.28.6",
        "@jridgewell/remapping": "^2.3.5",
        "convert-source-map": "^2.0.0",
        "debug": "^4.1.0",
        "gensync": "^1.0.0-beta.2",
        "json5": "^2.2.3",
        "semver": "^6.3.1"
      },
      "engines": {
        "node": ">=6.9.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/babel"
      }
    },
    "node_modules/@babel/generator": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/generator/-/generator-7.28.6.tgz",
      "integrity": "sha512-lOoVRwADj8hjf7al89tvQ2a1lf53Z+7tiXMgpZJL3maQPDxh0DgLMN62B2MKUOFcoodBHLMbDM6WAbKgNy5Suw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/parser": "^7.28.6",
        "@babel/types": "^7.28.6",
        "@jridgewell/gen-mapping": "^0.3.12",
        "@jridgewell/trace-mapping": "^0.3.28",
        "jsesc": "^3.0.2"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/helper-compilation-targets": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/helper-compilation-targets/-/helper-compilation-targets-7.28.6.tgz",
      "integrity": "sha512-JYtls3hqi15fcx5GaSNL7SCTJ2MNmjrkHXg4FSpOA/grxK8KwyZ5bubHsCq8FXCkua6xhuaaBit+3b7+VZRfcA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/compat-data": "^7.28.6",
        "@babel/helper-validator-option": "^7.27.1",
        "browserslist": "^4.24.0",
        "lru-cache": "^5.1.1",
        "semver": "^6.3.1"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/helper-globals": {
      "version": "7.28.0",
      "resolved": "https://registry.npmjs.org/@babel/helper-globals/-/helper-globals-7.28.0.tgz",
      "integrity": "sha512-+W6cISkXFa1jXsDEdYA8HeevQT/FULhxzR99pxphltZcVaugps53THCeiWA8SguxxpSp3gKPiuYfSWopkLQ4hw==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/helper-module-imports": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/helper-module-imports/-/helper-module-imports-7.28.6.tgz",
      "integrity": "sha512-l5XkZK7r7wa9LucGw9LwZyyCUscb4x37JWTPz7swwFE/0FMQAGpiWUZn8u9DzkSBWEcK25jmvubfpw2dnAMdbw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/traverse": "^7.28.6",
        "@babel/types": "^7.28.6"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/helper-module-transforms": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/helper-module-transforms/-/helper-module-transforms-7.28.6.tgz",
      "integrity": "sha512-67oXFAYr2cDLDVGLXTEABjdBJZ6drElUSI7WKp70NrpyISso3plG9SAGEF6y7zbha/wOzUByWWTJvEDVNIUGcA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/helper-module-imports": "^7.28.6",
        "@babel/helper-validator-identifier": "^7.28.5",
        "@babel/traverse": "^7.28.6"
      },
      "engines": {
        "node": ">=6.9.0"
      },
      "peerDependencies": {
        "@babel/core": "^7.0.0"
      }
    },
    "node_modules/@babel/helper-plugin-utils": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/helper-plugin-utils/-/helper-plugin-utils-7.28.6.tgz",
      "integrity": "sha512-S9gzZ/bz83GRysI7gAD4wPT/AI3uCnY+9xn+Mx/KPs2JwHJIz1W8PZkg2cqyt3RNOBM8ejcXhV6y8Og7ly/Dug==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/helper-string-parser": {
      "version": "7.27.1",
      "resolved": "https://registry.npmjs.org/@babel/helper-string-parser/-/helper-string-parser-7.27.1.tgz",
      "integrity": "sha512-qMlSxKbpRlAridDExk92nSobyDdpPijUq2DW6oDnUqd0iOGxmQjyqhMIihI9+zv4LPyZdRje2cavWPbCbWm3eA==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/helper-validator-identifier": {
      "version": "7.28.5",
      "resolved": "https://registry.npmjs.org/@babel/helper-validator-identifier/-/helper-validator-identifier-7.28.5.tgz",
      "integrity": "sha512-qSs4ifwzKJSV39ucNjsvc6WVHs6b7S03sOh2OcHF9UHfVPqWWALUsNUVzhSBiItjRZoLHx7nIarVjqKVusUZ1Q==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/helper-validator-option": {
      "version": "7.27.1",
      "resolved": "https://registry.npmjs.org/@babel/helper-validator-option/-/helper-validator-option-7.27.1.tgz",
      "integrity": "sha512-YvjJow9FxbhFFKDSuFnVCe2WxXk1zWc22fFePVNEaWJEu8IrZVlda6N0uHwzZrUM1il7NC9Mlp4MaJYbYd9JSg==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/helpers": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/helpers/-/helpers-7.28.6.tgz",
      "integrity": "sha512-xOBvwq86HHdB7WUDTfKfT/Vuxh7gElQ+Sfti2Cy6yIWNW05P8iUslOVcZ4/sKbE+/jQaukQAdz/gf3724kYdqw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/template": "^7.28.6",
        "@babel/types": "^7.28.6"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/parser": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/parser/-/parser-7.28.6.tgz",
      "integrity": "sha512-TeR9zWR18BvbfPmGbLampPMW+uW1NZnJlRuuHso8i87QZNq2JRF9i6RgxRqtEq+wQGsS19NNTWr2duhnE49mfQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/types": "^7.28.6"
      },
      "bin": {
        "parser": "bin/babel-parser.js"
      },
      "engines": {
        "node": ">=6.0.0"
      }
    },
    "node_modules/@babel/plugin-transform-react-jsx-self": {
      "version": "7.27.1",
      "resolved": "https://registry.npmjs.org/@babel/plugin-transform-react-jsx-self/-/plugin-transform-react-jsx-self-7.27.1.tgz",
      "integrity": "sha512-6UzkCs+ejGdZ5mFFC/OCUrv028ab2fp1znZmCZjAOBKiBK2jXD1O+BPSfX8X2qjJ75fZBMSnQn3Rq2mrBJK2mw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/helper-plugin-utils": "^7.27.1"
      },
      "engines": {
        "node": ">=6.9.0"
      },
      "peerDependencies": {
        "@babel/core": "^7.0.0-0"
      }
    },
    "node_modules/@babel/plugin-transform-react-jsx-source": {
      "version": "7.27.1",
      "resolved": "https://registry.npmjs.org/@babel/plugin-transform-react-jsx-source/-/plugin-transform-react-jsx-source-7.27.1.tgz",
      "integrity": "sha512-zbwoTsBruTeKB9hSq73ha66iFeJHuaFkUbwvqElnygoNbj/jHRsSeokowZFN3CZ64IvEqcmmkVe89OPXc7ldAw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/helper-plugin-utils": "^7.27.1"
      },
      "engines": {
        "node": ">=6.9.0"
      },
      "peerDependencies": {
        "@babel/core": "^7.0.0-0"
      }
    },
    "node_modules/@babel/template": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/template/-/template-7.28.6.tgz",
      "integrity": "sha512-YA6Ma2KsCdGb+WC6UpBVFJGXL58MDA6oyONbjyF/+5sBgxY/dwkhLogbMT2GXXyU84/IhRw/2D1Os1B/giz+BQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/code-frame": "^7.28.6",
        "@babel/parser": "^7.28.6",
        "@babel/types": "^7.28.6"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/traverse": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/traverse/-/traverse-7.28.6.tgz",
      "integrity": "sha512-fgWX62k02qtjqdSNTAGxmKYY/7FSL9WAS1o2Hu5+I5m9T0yxZzr4cnrfXQ/MX0rIifthCSs6FKTlzYbJcPtMNg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/code-frame": "^7.28.6",
        "@babel/generator": "^7.28.6",
        "@babel/helper-globals": "^7.28.0",
        "@babel/parser": "^7.28.6",
        "@babel/template": "^7.28.6",
        "@babel/types": "^7.28.6",
        "debug": "^4.3.1"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/types": {
      "version": "7.28.6",
      "resolved": "https://registry.npmjs.org/@babel/types/-/types-7.28.6.tgz",
      "integrity": "sha512-0ZrskXVEHSWIqZM/sQZ4EV3jZJXRkio/WCxaqKZP1g//CEWEPSfeZFcms4XeKBCHU0ZKnIkdJeU/kF+eRp5lBg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/helper-string-parser": "^7.27.1",
        "@babel/helper-validator-identifier": "^7.28.5"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@esbuild/aix-ppc64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/aix-ppc64/-/aix-ppc64-0.27.2.tgz",
      "integrity": "sha512-GZMB+a0mOMZs4MpDbj8RJp4cw+w1WV5NYD6xzgvzUJ5Ek2jerwfO2eADyI6ExDSUED+1X8aMbegahsJi+8mgpw==",
      "cpu": [
        "ppc64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "aix"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/android-arm": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/android-arm/-/android-arm-0.27.2.tgz",
      "integrity": "sha512-DVNI8jlPa7Ujbr1yjU2PfUSRtAUZPG9I1RwW4F4xFB1Imiu2on0ADiI/c3td+KmDtVKNbi+nffGDQMfcIMkwIA==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/android-arm64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/android-arm64/-/android-arm64-0.27.2.tgz",
      "integrity": "sha512-pvz8ZZ7ot/RBphf8fv60ljmaoydPU12VuXHImtAs0XhLLw+EXBi2BLe3OYSBslR4rryHvweW5gmkKFwTiFy6KA==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/android-x64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/android-x64/-/android-x64-0.27.2.tgz",
      "integrity": "sha512-z8Ank4Byh4TJJOh4wpz8g2vDy75zFL0TlZlkUkEwYXuPSgX8yzep596n6mT7905kA9uHZsf/o2OJZubl2l3M7A==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/darwin-arm64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/darwin-arm64/-/darwin-arm64-0.27.2.tgz",
      "integrity": "sha512-davCD2Zc80nzDVRwXTcQP/28fiJbcOwvdolL0sOiOsbwBa72kegmVU0Wrh1MYrbuCL98Omp5dVhQFWRKR2ZAlg==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/darwin-x64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/darwin-x64/-/darwin-x64-0.27.2.tgz",
      "integrity": "sha512-ZxtijOmlQCBWGwbVmwOF/UCzuGIbUkqB1faQRf5akQmxRJ1ujusWsb3CVfk/9iZKr2L5SMU5wPBi1UWbvL+VQA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/freebsd-arm64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/freebsd-arm64/-/freebsd-arm64-0.27.2.tgz",
      "integrity": "sha512-lS/9CN+rgqQ9czogxlMcBMGd+l8Q3Nj1MFQwBZJyoEKI50XGxwuzznYdwcav6lpOGv5BqaZXqvBSiB/kJ5op+g==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "freebsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/freebsd-x64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/freebsd-x64/-/freebsd-x64-0.27.2.tgz",
      "integrity": "sha512-tAfqtNYb4YgPnJlEFu4c212HYjQWSO/w/h/lQaBK7RbwGIkBOuNKQI9tqWzx7Wtp7bTPaGC6MJvWI608P3wXYA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "freebsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-arm": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-arm/-/linux-arm-0.27.2.tgz",
      "integrity": "sha512-vWfq4GaIMP9AIe4yj1ZUW18RDhx6EPQKjwe7n8BbIecFtCQG4CfHGaHuh7fdfq+y3LIA2vGS/o9ZBGVxIDi9hw==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-arm64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-arm64/-/linux-arm64-0.27.2.tgz",
      "integrity": "sha512-hYxN8pr66NsCCiRFkHUAsxylNOcAQaxSSkHMMjcpx0si13t1LHFphxJZUiGwojB1a/Hd5OiPIqDdXONia6bhTw==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-ia32": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-ia32/-/linux-ia32-0.27.2.tgz",
      "integrity": "sha512-MJt5BRRSScPDwG2hLelYhAAKh9imjHK5+NE/tvnRLbIqUWa+0E9N4WNMjmp/kXXPHZGqPLxggwVhz7QP8CTR8w==",
      "cpu": [
        "ia32"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-loong64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-loong64/-/linux-loong64-0.27.2.tgz",
      "integrity": "sha512-lugyF1atnAT463aO6KPshVCJK5NgRnU4yb3FUumyVz+cGvZbontBgzeGFO1nF+dPueHD367a2ZXe1NtUkAjOtg==",
      "cpu": [
        "loong64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-mips64el": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-mips64el/-/linux-mips64el-0.27.2.tgz",
      "integrity": "sha512-nlP2I6ArEBewvJ2gjrrkESEZkB5mIoaTswuqNFRv/WYd+ATtUpe9Y09RnJvgvdag7he0OWgEZWhviS1OTOKixw==",
      "cpu": [
        "mips64el"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-ppc64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-ppc64/-/linux-ppc64-0.27.2.tgz",
      "integrity": "sha512-C92gnpey7tUQONqg1n6dKVbx3vphKtTHJaNG2Ok9lGwbZil6DrfyecMsp9CrmXGQJmZ7iiVXvvZH6Ml5hL6XdQ==",
      "cpu": [
        "ppc64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-riscv64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-riscv64/-/linux-riscv64-0.27.2.tgz",
      "integrity": "sha512-B5BOmojNtUyN8AXlK0QJyvjEZkWwy/FKvakkTDCziX95AowLZKR6aCDhG7LeF7uMCXEJqwa8Bejz5LTPYm8AvA==",
      "cpu": [
        "riscv64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-s390x": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-s390x/-/linux-s390x-0.27.2.tgz",
      "integrity": "sha512-p4bm9+wsPwup5Z8f4EpfN63qNagQ47Ua2znaqGH6bqLlmJ4bx97Y9JdqxgGZ6Y8xVTixUnEkoKSHcpRlDnNr5w==",
      "cpu": [
        "s390x"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-x64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-x64/-/linux-x64-0.27.2.tgz",
      "integrity": "sha512-uwp2Tip5aPmH+NRUwTcfLb+W32WXjpFejTIOWZFw/v7/KnpCDKG66u4DLcurQpiYTiYwQ9B7KOeMJvLCu/OvbA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/netbsd-arm64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/netbsd-arm64/-/netbsd-arm64-0.27.2.tgz",
      "integrity": "sha512-Kj6DiBlwXrPsCRDeRvGAUb/LNrBASrfqAIok+xB0LxK8CHqxZ037viF13ugfsIpePH93mX7xfJp97cyDuTZ3cw==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "netbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/netbsd-x64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/netbsd-x64/-/netbsd-x64-0.27.2.tgz",
      "integrity": "sha512-HwGDZ0VLVBY3Y+Nw0JexZy9o/nUAWq9MlV7cahpaXKW6TOzfVno3y3/M8Ga8u8Yr7GldLOov27xiCnqRZf0tCA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "netbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/openbsd-arm64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/openbsd-arm64/-/openbsd-arm64-0.27.2.tgz",
      "integrity": "sha512-DNIHH2BPQ5551A7oSHD0CKbwIA/Ox7+78/AWkbS5QoRzaqlev2uFayfSxq68EkonB+IKjiuxBFoV8ESJy8bOHA==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "openbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/openbsd-x64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/openbsd-x64/-/openbsd-x64-0.27.2.tgz",
      "integrity": "sha512-/it7w9Nb7+0KFIzjalNJVR5bOzA9Vay+yIPLVHfIQYG/j+j9VTH84aNB8ExGKPU4AzfaEvN9/V4HV+F+vo8OEg==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "openbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/openharmony-arm64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/openharmony-arm64/-/openharmony-arm64-0.27.2.tgz",
      "integrity": "sha512-LRBbCmiU51IXfeXk59csuX/aSaToeG7w48nMwA6049Y4J4+VbWALAuXcs+qcD04rHDuSCSRKdmY63sruDS5qag==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "openharmony"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/sunos-x64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/sunos-x64/-/sunos-x64-0.27.2.tgz",
      "integrity": "sha512-kMtx1yqJHTmqaqHPAzKCAkDaKsffmXkPHThSfRwZGyuqyIeBvf08KSsYXl+abf5HDAPMJIPnbBfXvP2ZC2TfHg==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "sunos"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/win32-arm64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/win32-arm64/-/win32-arm64-0.27.2.tgz",
      "integrity": "sha512-Yaf78O/B3Kkh+nKABUF++bvJv5Ijoy9AN1ww904rOXZFLWVc5OLOfL56W+C8F9xn5JQZa3UX6m+IktJnIb1Jjg==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/win32-ia32": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/win32-ia32/-/win32-ia32-0.27.2.tgz",
      "integrity": "sha512-Iuws0kxo4yusk7sw70Xa2E2imZU5HoixzxfGCdxwBdhiDgt9vX9VUCBhqcwY7/uh//78A1hMkkROMJq9l27oLQ==",
      "cpu": [
        "ia32"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/win32-x64": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/@esbuild/win32-x64/-/win32-x64-0.27.2.tgz",
      "integrity": "sha512-sRdU18mcKf7F+YgheI/zGf5alZatMUTKj/jNS6l744f9u3WFu4v7twcUI9vu4mknF4Y9aDlblIie0IM+5xxaqQ==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@eslint-community/eslint-utils": {
      "version": "4.9.1",
      "resolved": "https://registry.npmjs.org/@eslint-community/eslint-utils/-/eslint-utils-4.9.1.tgz",
      "integrity": "sha512-phrYmNiYppR7znFEdqgfWHXR6NCkZEK7hwWDHZUjit/2/U0r6XvkDl0SYnoM51Hq7FhCGdLDT6zxCCOY1hexsQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "eslint-visitor-keys": "^3.4.3"
      },
      "engines": {
        "node": "^12.22.0 || ^14.17.0 || >=16.0.0"
      },
      "funding": {
        "url": "https://opencollective.com/eslint"
      },
      "peerDependencies": {
        "eslint": "^6.0.0 || ^7.0.0 || >=8.0.0"
      }
    },
    "node_modules/@eslint-community/eslint-utils/node_modules/eslint-visitor-keys": {
      "version": "3.4.3",
      "resolved": "https://registry.npmjs.org/eslint-visitor-keys/-/eslint-visitor-keys-3.4.3.tgz",
      "integrity": "sha512-wpc+LXeiyiisxPlEkUzU6svyS1frIO3Mgxj1fdy7Pm8Ygzguax2N3Fa/D/ag1WqbOprdI+uY6wMUl8/a2G+iag==",
      "dev": true,
      "license": "Apache-2.0",
      "engines": {
        "node": "^12.22.0 || ^14.17.0 || >=16.0.0"
      },
      "funding": {
        "url": "https://opencollective.com/eslint"
      }
    },
    "node_modules/@eslint-community/regexpp": {
      "version": "4.12.2",
      "resolved": "https://registry.npmjs.org/@eslint-community/regexpp/-/regexpp-4.12.2.tgz",
      "integrity": "sha512-EriSTlt5OC9/7SXkRSCAhfSxxoSUgBm33OH+IkwbdpgoqsSsUg7y3uh+IICI/Qg4BBWr3U2i39RpmycbxMq4ew==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": "^12.0.0 || ^14.0.0 || >=16.0.0"
      }
    },
    "node_modules/@eslint/config-array": {
      "version": "0.21.1",
      "resolved": "https://registry.npmjs.org/@eslint/config-array/-/config-array-0.21.1.tgz",
      "integrity": "sha512-aw1gNayWpdI/jSYVgzN5pL0cfzU02GT3NBpeT/DXbx1/1x7ZKxFPd9bwrzygx/qiwIQiJ1sw/zD8qY/kRvlGHA==",
      "dev": true,
      "license": "Apache-2.0",
      "dependencies": {
        "@eslint/object-schema": "^2.1.7",
        "debug": "^4.3.1",
        "minimatch": "^3.1.2"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      }
    },
    "node_modules/@eslint/config-helpers": {
      "version": "0.4.2",
      "resolved": "https://registry.npmjs.org/@eslint/config-helpers/-/config-helpers-0.4.2.tgz",
      "integrity": "sha512-gBrxN88gOIf3R7ja5K9slwNayVcZgK6SOUORm2uBzTeIEfeVaIhOpCtTox3P6R7o2jLFwLFTLnC7kU/RGcYEgw==",
      "dev": true,
      "license": "Apache-2.0",
      "dependencies": {
        "@eslint/core": "^0.17.0"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      }
    },
    "node_modules/@eslint/core": {
      "version": "0.17.0",
      "resolved": "https://registry.npmjs.org/@eslint/core/-/core-0.17.0.tgz",
      "integrity": "sha512-yL/sLrpmtDaFEiUj1osRP4TI2MDz1AddJL+jZ7KSqvBuliN4xqYY54IfdN8qD8Toa6g1iloph1fxQNkjOxrrpQ==",
      "dev": true,
      "license": "Apache-2.0",
      "dependencies": {
        "@types/json-schema": "^7.0.15"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      }
    },
    "node_modules/@eslint/eslintrc": {
      "version": "3.3.3",
      "resolved": "https://registry.npmjs.org/@eslint/eslintrc/-/eslintrc-3.3.3.tgz",
      "integrity": "sha512-Kr+LPIUVKz2qkx1HAMH8q1q6azbqBAsXJUxBl/ODDuVPX45Z9DfwB8tPjTi6nNZ8BuM3nbJxC5zCAg5elnBUTQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "ajv": "^6.12.4",
        "debug": "^4.3.2",
        "espree": "^10.0.1",
        "globals": "^14.0.0",
        "ignore": "^5.2.0",
        "import-fresh": "^3.2.1",
        "js-yaml": "^4.1.1",
        "minimatch": "^3.1.2",
        "strip-json-comments": "^3.1.1"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "url": "https://opencollective.com/eslint"
      }
    },
    "node_modules/@eslint/eslintrc/node_modules/globals": {
      "version": "14.0.0",
      "resolved": "https://registry.npmjs.org/globals/-/globals-14.0.0.tgz",
      "integrity": "sha512-oahGvuMGQlPw/ivIYBjVSrWAfWLBeku5tpPE2fOPLi+WHffIWbuh2tCjhyQhTBPMf5E9jDEH4FOmTYgYwbKwtQ==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=18"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/@eslint/js": {
      "version": "9.39.2",
      "resolved": "https://registry.npmjs.org/@eslint/js/-/js-9.39.2.tgz",
      "integrity": "sha512-q1mjIoW1VX4IvSocvM/vbTiveKC4k9eLrajNEuSsmjymSDEbpGddtpfOoN7YGAqBK3NG+uqo8ia4PDTt8buCYA==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "url": "https://eslint.org/donate"
      }
    },
    "node_modules/@eslint/object-schema": {
      "version": "2.1.7",
      "resolved": "https://registry.npmjs.org/@eslint/object-schema/-/object-schema-2.1.7.tgz",
      "integrity": "sha512-VtAOaymWVfZcmZbp6E2mympDIHvyjXs/12LqWYjVw6qjrfF+VK+fyG33kChz3nnK+SU5/NeHOqrTEHS8sXO3OA==",
      "dev": true,
      "license": "Apache-2.0",
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      }
    },
    "node_modules/@eslint/plugin-kit": {
      "version": "0.4.1",
      "resolved": "https://registry.npmjs.org/@eslint/plugin-kit/-/plugin-kit-0.4.1.tgz",
      "integrity": "sha512-43/qtrDUokr7LJqoF2c3+RInu/t4zfrpYdoSDfYyhg52rwLV6TnOvdG4fXm7IkSB3wErkcmJS9iEhjVtOSEjjA==",
      "dev": true,
      "license": "Apache-2.0",
      "dependencies": {
        "@eslint/core": "^0.17.0",
        "levn": "^0.4.1"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      }
    },
    "node_modules/@humanfs/core": {
      "version": "0.19.1",
      "resolved": "https://registry.npmjs.org/@humanfs/core/-/core-0.19.1.tgz",
      "integrity": "sha512-5DyQ4+1JEUzejeK1JGICcideyfUbGixgS9jNgex5nqkW+cY7WZhxBigmieN5Qnw9ZosSNVC9KQKyb+GUaGyKUA==",
      "dev": true,
      "license": "Apache-2.0",
      "engines": {
        "node": ">=18.18.0"
      }
    },
    "node_modules/@humanfs/node": {
      "version": "0.16.7",
      "resolved": "https://registry.npmjs.org/@humanfs/node/-/node-0.16.7.tgz",
      "integrity": "sha512-/zUx+yOsIrG4Y43Eh2peDeKCxlRt/gET6aHfaKpuq267qXdYDFViVHfMaLyygZOnl0kGWxFIgsBy8QFuTLUXEQ==",
      "dev": true,
      "license": "Apache-2.0",
      "dependencies": {
        "@humanfs/core": "^0.19.1",
        "@humanwhocodes/retry": "^0.4.0"
      },
      "engines": {
        "node": ">=18.18.0"
      }
    },
    "node_modules/@humanwhocodes/module-importer": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/@humanwhocodes/module-importer/-/module-importer-1.0.1.tgz",
      "integrity": "sha512-bxveV4V8v5Yb4ncFTT3rPSgZBOpCkjfK0y4oVVVJwIuDVBRMDXrPyXRL988i5ap9m9bnyEEjWfm5WkBmtffLfA==",
      "dev": true,
      "license": "Apache-2.0",
      "engines": {
        "node": ">=12.22"
      },
      "funding": {
        "type": "github",
        "url": "https://github.com/sponsors/nzakas"
      }
    },
    "node_modules/@humanwhocodes/retry": {
      "version": "0.4.3",
      "resolved": "https://registry.npmjs.org/@humanwhocodes/retry/-/retry-0.4.3.tgz",
      "integrity": "sha512-bV0Tgo9K4hfPCek+aMAn81RppFKv2ySDQeMoSZuvTASywNTnVJCArCZE2FWqpvIatKu7VMRLWlR1EazvVhDyhQ==",
      "dev": true,
      "license": "Apache-2.0",
      "engines": {
        "node": ">=18.18"
      },
      "funding": {
        "type": "github",
        "url": "https://github.com/sponsors/nzakas"
      }
    },
    "node_modules/@jridgewell/gen-mapping": {
      "version": "0.3.13",
      "resolved": "https://registry.npmjs.org/@jridgewell/gen-mapping/-/gen-mapping-0.3.13.tgz",
      "integrity": "sha512-2kkt/7niJ6MgEPxF0bYdQ6etZaA+fQvDcLKckhy1yIQOzaoKjBBjSj63/aLVjYE3qhRt5dvM+uUyfCg6UKCBbA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@jridgewell/sourcemap-codec": "^1.5.0",
        "@jridgewell/trace-mapping": "^0.3.24"
      }
    },
    "node_modules/@jridgewell/remapping": {
      "version": "2.3.5",
      "resolved": "https://registry.npmjs.org/@jridgewell/remapping/-/remapping-2.3.5.tgz",
      "integrity": "sha512-LI9u/+laYG4Ds1TDKSJW2YPrIlcVYOwi2fUC6xB43lueCjgxV4lffOCZCtYFiH6TNOX+tQKXx97T4IKHbhyHEQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@jridgewell/gen-mapping": "^0.3.5",
        "@jridgewell/trace-mapping": "^0.3.24"
      }
    },
    "node_modules/@jridgewell/resolve-uri": {
      "version": "3.1.2",
      "resolved": "https://registry.npmjs.org/@jridgewell/resolve-uri/-/resolve-uri-3.1.2.tgz",
      "integrity": "sha512-bRISgCIjP20/tbWSPWMEi54QVPRZExkuD9lJL+UIxUKtwVJA8wW1Trb1jMs1RFXo1CBTNZ/5hpC9QvmKWdopKw==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6.0.0"
      }
    },
    "node_modules/@jridgewell/sourcemap-codec": {
      "version": "1.5.5",
      "resolved": "https://registry.npmjs.org/@jridgewell/sourcemap-codec/-/sourcemap-codec-1.5.5.tgz",
      "integrity": "sha512-cYQ9310grqxueWbl+WuIUIaiUaDcj7WOq5fVhEljNVgRfOUhY9fy2zTvfoqWsnebh8Sl70VScFbICvJnLKB0Og==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/@jridgewell/trace-mapping": {
      "version": "0.3.31",
      "resolved": "https://registry.npmjs.org/@jridgewell/trace-mapping/-/trace-mapping-0.3.31.tgz",
      "integrity": "sha512-zzNR+SdQSDJzc8joaeP8QQoCQr8NuYx2dIIytl1QeBEZHJ9uW6hebsrYgbz8hJwUQao3TWCMtmfV8Nu1twOLAw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@jridgewell/resolve-uri": "^3.1.0",
        "@jridgewell/sourcemap-codec": "^1.4.14"
      }
    },
    "node_modules/@rolldown/pluginutils": {
      "version": "1.0.0-beta.53",
      "resolved": "https://registry.npmjs.org/@rolldown/pluginutils/-/pluginutils-1.0.0-beta.53.tgz",
      "integrity": "sha512-vENRlFU4YbrwVqNDZ7fLvy+JR1CRkyr01jhSiDpE1u6py3OMzQfztQU2jxykW3ALNxO4kSlqIDeYyD0Y9RcQeQ==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/@rollup/rollup-android-arm-eabi": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-android-arm-eabi/-/rollup-android-arm-eabi-4.56.0.tgz",
      "integrity": "sha512-LNKIPA5k8PF1+jAFomGe3qN3bbIgJe/IlpDBwuVjrDKrJhVWywgnJvflMt/zkbVNLFtF1+94SljYQS6e99klnw==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ]
    },
    "node_modules/@rollup/rollup-android-arm64": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-android-arm64/-/rollup-android-arm64-4.56.0.tgz",
      "integrity": "sha512-lfbVUbelYqXlYiU/HApNMJzT1E87UPGvzveGg2h0ktUNlOCxKlWuJ9jtfvs1sKHdwU4fzY7Pl8sAl49/XaEk6Q==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ]
    },
    "node_modules/@rollup/rollup-darwin-arm64": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-darwin-arm64/-/rollup-darwin-arm64-4.56.0.tgz",
      "integrity": "sha512-EgxD1ocWfhoD6xSOeEEwyE7tDvwTgZc8Bss7wCWe+uc7wO8G34HHCUH+Q6cHqJubxIAnQzAsyUsClt0yFLu06w==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ]
    },
    "node_modules/@rollup/rollup-darwin-x64": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-darwin-x64/-/rollup-darwin-x64-4.56.0.tgz",
      "integrity": "sha512-1vXe1vcMOssb/hOF8iv52A7feWW2xnu+c8BV4t1F//m9QVLTfNVpEdja5ia762j/UEJe2Z1jAmEqZAK42tVW3g==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ]
    },
    "node_modules/@rollup/rollup-freebsd-arm64": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-freebsd-arm64/-/rollup-freebsd-arm64-4.56.0.tgz",
      "integrity": "sha512-bof7fbIlvqsyv/DtaXSck4VYQ9lPtoWNFCB/JY4snlFuJREXfZnm+Ej6yaCHfQvofJDXLDMTVxWscVSuQvVWUQ==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "freebsd"
      ]
    },
    "node_modules/@rollup/rollup-freebsd-x64": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-freebsd-x64/-/rollup-freebsd-x64-4.56.0.tgz",
      "integrity": "sha512-KNa6lYHloW+7lTEkYGa37fpvPq+NKG/EHKM8+G/g9WDU7ls4sMqbVRV78J6LdNuVaeeK5WB9/9VAFbKxcbXKYg==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "freebsd"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm-gnueabihf": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm-gnueabihf/-/rollup-linux-arm-gnueabihf-4.56.0.tgz",
      "integrity": "sha512-E8jKK87uOvLrrLN28jnAAAChNq5LeCd2mGgZF+fGF5D507WlG/Noct3lP/QzQ6MrqJ5BCKNwI9ipADB6jyiq2A==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm-musleabihf": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm-musleabihf/-/rollup-linux-arm-musleabihf-4.56.0.tgz",
      "integrity": "sha512-jQosa5FMYF5Z6prEpTCCmzCXz6eKr/tCBssSmQGEeozA9tkRUty/5Vx06ibaOP9RCrW1Pvb8yp3gvZhHwTDsJw==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm64-gnu": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm64-gnu/-/rollup-linux-arm64-gnu-4.56.0.tgz",
      "integrity": "sha512-uQVoKkrC1KGEV6udrdVahASIsaF8h7iLG0U0W+Xn14ucFwi6uS539PsAr24IEF9/FoDtzMeeJXJIBo5RkbNWvQ==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm64-musl": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm64-musl/-/rollup-linux-arm64-musl-4.56.0.tgz",
      "integrity": "sha512-vLZ1yJKLxhQLFKTs42RwTwa6zkGln+bnXc8ueFGMYmBTLfNu58sl5/eXyxRa2RarTkJbXl8TKPgfS6V5ijNqEA==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-loong64-gnu": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-loong64-gnu/-/rollup-linux-loong64-gnu-4.56.0.tgz",
      "integrity": "sha512-FWfHOCub564kSE3xJQLLIC/hbKqHSVxy8vY75/YHHzWvbJL7aYJkdgwD/xGfUlL5UV2SB7otapLrcCj2xnF1dg==",
      "cpu": [
        "loong64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-loong64-musl": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-loong64-musl/-/rollup-linux-loong64-musl-4.56.0.tgz",
      "integrity": "sha512-z1EkujxIh7nbrKL1lmIpqFTc/sr0u8Uk0zK/qIEFldbt6EDKWFk/pxFq3gYj4Bjn3aa9eEhYRlL3H8ZbPT1xvA==",
      "cpu": [
        "loong64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-ppc64-gnu": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-ppc64-gnu/-/rollup-linux-ppc64-gnu-4.56.0.tgz",
      "integrity": "sha512-iNFTluqgdoQC7AIE8Q34R3AuPrJGJirj5wMUErxj22deOcY7XwZRaqYmB6ZKFHoVGqRcRd0mqO+845jAibKCkw==",
      "cpu": [
        "ppc64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-ppc64-musl": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-ppc64-musl/-/rollup-linux-ppc64-musl-4.56.0.tgz",
      "integrity": "sha512-MtMeFVlD2LIKjp2sE2xM2slq3Zxf9zwVuw0jemsxvh1QOpHSsSzfNOTH9uYW9i1MXFxUSMmLpeVeUzoNOKBaWg==",
      "cpu": [
        "ppc64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-riscv64-gnu": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-riscv64-gnu/-/rollup-linux-riscv64-gnu-4.56.0.tgz",
      "integrity": "sha512-in+v6wiHdzzVhYKXIk5U74dEZHdKN9KH0Q4ANHOTvyXPG41bajYRsy7a8TPKbYPl34hU7PP7hMVHRvv/5aCSew==",
      "cpu": [
        "riscv64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-riscv64-musl": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-riscv64-musl/-/rollup-linux-riscv64-musl-4.56.0.tgz",
      "integrity": "sha512-yni2raKHB8m9NQpI9fPVwN754mn6dHQSbDTwxdr9SE0ks38DTjLMMBjrwvB5+mXrX+C0npX0CVeCUcvvvD8CNQ==",
      "cpu": [
        "riscv64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-s390x-gnu": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-s390x-gnu/-/rollup-linux-s390x-gnu-4.56.0.tgz",
      "integrity": "sha512-zhLLJx9nQPu7wezbxt2ut+CI4YlXi68ndEve16tPc/iwoylWS9B3FxpLS2PkmfYgDQtosah07Mj9E0khc3Y+vQ==",
      "cpu": [
        "s390x"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-x64-gnu": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-x64-gnu/-/rollup-linux-x64-gnu-4.56.0.tgz",
      "integrity": "sha512-MVC6UDp16ZSH7x4rtuJPAEoE1RwS8N4oK9DLHy3FTEdFoUTCFVzMfJl/BVJ330C+hx8FfprA5Wqx4FhZXkj2Kw==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-x64-musl": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-x64-musl/-/rollup-linux-x64-musl-4.56.0.tgz",
      "integrity": "sha512-ZhGH1eA4Qv0lxaV00azCIS1ChedK0V32952Md3FtnxSqZTBTd6tgil4nZT5cU8B+SIw3PFYkvyR4FKo2oyZIHA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-openbsd-x64": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-openbsd-x64/-/rollup-openbsd-x64-4.56.0.tgz",
      "integrity": "sha512-O16XcmyDeFI9879pEcmtWvD/2nyxR9mF7Gs44lf1vGGx8Vg2DRNx11aVXBEqOQhWb92WN4z7fW/q4+2NYzCbBA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "openbsd"
      ]
    },
    "node_modules/@rollup/rollup-openharmony-arm64": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-openharmony-arm64/-/rollup-openharmony-arm64-4.56.0.tgz",
      "integrity": "sha512-LhN/Reh+7F3RCgQIRbgw8ZMwUwyqJM+8pXNT6IIJAqm2IdKkzpCh/V9EdgOMBKuebIrzswqy4ATlrDgiOwbRcQ==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "openharmony"
      ]
    },
    "node_modules/@rollup/rollup-win32-arm64-msvc": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-win32-arm64-msvc/-/rollup-win32-arm64-msvc-4.56.0.tgz",
      "integrity": "sha512-kbFsOObXp3LBULg1d3JIUQMa9Kv4UitDmpS+k0tinPBz3watcUiV2/LUDMMucA6pZO3WGE27P7DsfaN54l9ing==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@rollup/rollup-win32-ia32-msvc": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-win32-ia32-msvc/-/rollup-win32-ia32-msvc-4.56.0.tgz",
      "integrity": "sha512-vSSgny54D6P4vf2izbtFm/TcWYedw7f8eBrOiGGecyHyQB9q4Kqentjaj8hToe+995nob/Wv48pDqL5a62EWtg==",
      "cpu": [
        "ia32"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@rollup/rollup-win32-x64-gnu": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-win32-x64-gnu/-/rollup-win32-x64-gnu-4.56.0.tgz",
      "integrity": "sha512-FeCnkPCTHQJFbiGG49KjV5YGW/8b9rrXAM2Mz2kiIoktq2qsJxRD5giEMEOD2lPdgs72upzefaUvS+nc8E3UzQ==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@rollup/rollup-win32-x64-msvc": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-win32-x64-msvc/-/rollup-win32-x64-msvc-4.56.0.tgz",
      "integrity": "sha512-H8AE9Ur/t0+1VXujj90w0HrSOuv0Nq9r1vSZF2t5km20NTfosQsGGUXDaKdQZzwuLts7IyL1fYT4hM95TI9c4g==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@types/babel__core": {
      "version": "7.20.5",
      "resolved": "https://registry.npmjs.org/@types/babel__core/-/babel__core-7.20.5.tgz",
      "integrity": "sha512-qoQprZvz5wQFJwMDqeseRXWv3rqMvhgpbXFfVyWhbx9X47POIA6i/+dXefEmZKoAgOaTdaIgNSMqMIU61yRyzA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/parser": "^7.20.7",
        "@babel/types": "^7.20.7",
        "@types/babel__generator": "*",
        "@types/babel__template": "*",
        "@types/babel__traverse": "*"
      }
    },
    "node_modules/@types/babel__generator": {
      "version": "7.27.0",
      "resolved": "https://registry.npmjs.org/@types/babel__generator/-/babel__generator-7.27.0.tgz",
      "integrity": "sha512-ufFd2Xi92OAVPYsy+P4n7/U7e68fex0+Ee8gSG9KX7eo084CWiQ4sdxktvdl0bOPupXtVJPY19zk6EwWqUQ8lg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/types": "^7.0.0"
      }
    },
    "node_modules/@types/babel__template": {
      "version": "7.4.4",
      "resolved": "https://registry.npmjs.org/@types/babel__template/-/babel__template-7.4.4.tgz",
      "integrity": "sha512-h/NUaSyG5EyxBIp8YRxo4RMe2/qQgvyowRwVMzhYhBCONbW8PUsg4lkFMrhgZhUe5z3L3MiLDuvyJ/CaPa2A8A==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/parser": "^7.1.0",
        "@babel/types": "^7.0.0"
      }
    },
    "node_modules/@types/babel__traverse": {
      "version": "7.28.0",
      "resolved": "https://registry.npmjs.org/@types/babel__traverse/-/babel__traverse-7.28.0.tgz",
      "integrity": "sha512-8PvcXf70gTDZBgt9ptxJ8elBeBjcLOAcOtoO/mPJjtji1+CdGbHgm77om1GrsPxsiE+uXIpNSK64UYaIwQXd4Q==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/types": "^7.28.2"
      }
    },
    "node_modules/@types/estree": {
      "version": "1.0.8",
      "resolved": "https://registry.npmjs.org/@types/estree/-/estree-1.0.8.tgz",
      "integrity": "sha512-dWHzHa2WqEXI/O1E9OjrocMTKJl2mSrEolh1Iomrv6U+JuNwaHXsXx9bLu5gG7BUWFIN0skIQJQ/L1rIex4X6w==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/@types/json-schema": {
      "version": "7.0.15",
      "resolved": "https://registry.npmjs.org/@types/json-schema/-/json-schema-7.0.15.tgz",
      "integrity": "sha512-5+fP8P8MFNC+AyZCDxrB2pkZFPGzqQWUzpSeuuVLvm8VMcorNYavBqoFcxK8bQz4Qsbn4oUEEem4wDLfcysGHA==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/@types/node": {
      "version": "24.10.9",
      "resolved": "https://registry.npmjs.org/@types/node/-/node-24.10.9.tgz",
      "integrity": "sha512-ne4A0IpG3+2ETuREInjPNhUGis1SFjv1d5asp8MzEAGtOZeTeHVDOYqOgqfhvseqg/iXty2hjBf1zAOb7RNiNw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "undici-types": "~7.16.0"
      }
    },
    "node_modules/@types/react": {
      "version": "19.2.9",
      "resolved": "https://registry.npmjs.org/@types/react/-/react-19.2.9.tgz",
      "integrity": "sha512-Lpo8kgb/igvMIPeNV2rsYKTgaORYdO1XGVZ4Qz3akwOj0ySGYMPlQWa8BaLn0G63D1aSaAQ5ldR06wCpChQCjA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "csstype": "^3.2.2"
      }
    },
    "node_modules/@types/react-dom": {
      "version": "19.2.3",
      "resolved": "https://registry.npmjs.org/@types/react-dom/-/react-dom-19.2.3.tgz",
      "integrity": "sha512-jp2L/eY6fn+KgVVQAOqYItbF0VY/YApe5Mz2F0aykSO8gx31bYCZyvSeYxCHKvzHG5eZjc+zyaS5BrBWya2+kQ==",
      "dev": true,
      "license": "MIT",
      "peerDependencies": {
        "@types/react": "^19.2.0"
      }
    },
    "node_modules/@typescript-eslint/eslint-plugin": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/@typescript-eslint/eslint-plugin/-/eslint-plugin-8.53.1.tgz",
      "integrity": "sha512-cFYYFZ+oQFi6hUnBTbLRXfTJiaQtYE3t4O692agbBl+2Zy+eqSKWtPjhPXJu1G7j4RLjKgeJPDdq3EqOwmX5Ag==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@eslint-community/regexpp": "^4.12.2",
        "@typescript-eslint/scope-manager": "8.53.1",
        "@typescript-eslint/type-utils": "8.53.1",
        "@typescript-eslint/utils": "8.53.1",
        "@typescript-eslint/visitor-keys": "8.53.1",
        "ignore": "^7.0.5",
        "natural-compare": "^1.4.0",
        "ts-api-utils": "^2.4.0"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      },
      "peerDependencies": {
        "@typescript-eslint/parser": "^8.53.1",
        "eslint": "^8.57.0 || ^9.0.0",
        "typescript": ">=4.8.4 <6.0.0"
      }
    },
    "node_modules/@typescript-eslint/eslint-plugin/node_modules/ignore": {
      "version": "7.0.5",
      "resolved": "https://registry.npmjs.org/ignore/-/ignore-7.0.5.tgz",
      "integrity": "sha512-Hs59xBNfUIunMFgWAbGX5cq6893IbWg4KnrjbYwX3tx0ztorVgTDA6B2sxf8ejHJ4wz8BqGUMYlnzNBer5NvGg==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">= 4"
      }
    },
    "node_modules/@typescript-eslint/parser": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/@typescript-eslint/parser/-/parser-8.53.1.tgz",
      "integrity": "sha512-nm3cvFN9SqZGXjmw5bZ6cGmvJSyJPn0wU9gHAZZHDnZl2wF9PhHv78Xf06E0MaNk4zLVHL8hb2/c32XvyJOLQg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@typescript-eslint/scope-manager": "8.53.1",
        "@typescript-eslint/types": "8.53.1",
        "@typescript-eslint/typescript-estree": "8.53.1",
        "@typescript-eslint/visitor-keys": "8.53.1",
        "debug": "^4.4.3"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      },
      "peerDependencies": {
        "eslint": "^8.57.0 || ^9.0.0",
        "typescript": ">=4.8.4 <6.0.0"
      }
    },
    "node_modules/@typescript-eslint/project-service": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/@typescript-eslint/project-service/-/project-service-8.53.1.tgz",
      "integrity": "sha512-WYC4FB5Ra0xidsmlPb+1SsnaSKPmS3gsjIARwbEkHkoWloQmuzcfypljaJcR78uyLA1h8sHdWWPHSLDI+MtNog==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@typescript-eslint/tsconfig-utils": "^8.53.1",
        "@typescript-eslint/types": "^8.53.1",
        "debug": "^4.4.3"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      },
      "peerDependencies": {
        "typescript": ">=4.8.4 <6.0.0"
      }
    },
    "node_modules/@typescript-eslint/scope-manager": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/@typescript-eslint/scope-manager/-/scope-manager-8.53.1.tgz",
      "integrity": "sha512-Lu23yw1uJMFY8cUeq7JlrizAgeQvWugNQzJp8C3x8Eo5Jw5Q2ykMdiiTB9vBVOOUBysMzmRRmUfwFrZuI2C4SQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@typescript-eslint/types": "8.53.1",
        "@typescript-eslint/visitor-keys": "8.53.1"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      }
    },
    "node_modules/@typescript-eslint/tsconfig-utils": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/@typescript-eslint/tsconfig-utils/-/tsconfig-utils-8.53.1.tgz",
      "integrity": "sha512-qfvLXS6F6b1y43pnf0pPbXJ+YoXIC7HKg0UGZ27uMIemKMKA6XH2DTxsEDdpdN29D+vHV07x/pnlPNVLhdhWiA==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      },
      "peerDependencies": {
        "typescript": ">=4.8.4 <6.0.0"
      }
    },
    "node_modules/@typescript-eslint/type-utils": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/@typescript-eslint/type-utils/-/type-utils-8.53.1.tgz",
      "integrity": "sha512-MOrdtNvyhy0rHyv0ENzub1d4wQYKb2NmIqG7qEqPWFW7Mpy2jzFC3pQ2yKDvirZB7jypm5uGjF2Qqs6OIqu47w==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@typescript-eslint/types": "8.53.1",
        "@typescript-eslint/typescript-estree": "8.53.1",
        "@typescript-eslint/utils": "8.53.1",
        "debug": "^4.4.3",
        "ts-api-utils": "^2.4.0"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      },
      "peerDependencies": {
        "eslint": "^8.57.0 || ^9.0.0",
        "typescript": ">=4.8.4 <6.0.0"
      }
    },
    "node_modules/@typescript-eslint/types": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/@typescript-eslint/types/-/types-8.53.1.tgz",
      "integrity": "sha512-jr/swrr2aRmUAUjW5/zQHbMaui//vQlsZcJKijZf3M26bnmLj8LyZUpj8/Rd6uzaek06OWsqdofN/Thenm5O8A==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      }
    },
    "node_modules/@typescript-eslint/typescript-estree": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/@typescript-eslint/typescript-estree/-/typescript-estree-8.53.1.tgz",
      "integrity": "sha512-RGlVipGhQAG4GxV1s34O91cxQ/vWiHJTDHbXRr0li2q/BGg3RR/7NM8QDWgkEgrwQYCvmJV9ichIwyoKCQ+DTg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@typescript-eslint/project-service": "8.53.1",
        "@typescript-eslint/tsconfig-utils": "8.53.1",
        "@typescript-eslint/types": "8.53.1",
        "@typescript-eslint/visitor-keys": "8.53.1",
        "debug": "^4.4.3",
        "minimatch": "^9.0.5",
        "semver": "^7.7.3",
        "tinyglobby": "^0.2.15",
        "ts-api-utils": "^2.4.0"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      },
      "peerDependencies": {
        "typescript": ">=4.8.4 <6.0.0"
      }
    },
    "node_modules/@typescript-eslint/typescript-estree/node_modules/brace-expansion": {
      "version": "2.0.2",
      "resolved": "https://registry.npmjs.org/brace-expansion/-/brace-expansion-2.0.2.tgz",
      "integrity": "sha512-Jt0vHyM+jmUBqojB7E1NIYadt0vI0Qxjxd2TErW94wDz+E2LAm5vKMXXwg6ZZBTHPuUlDgQHKXvjGBdfcF1ZDQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "balanced-match": "^1.0.0"
      }
    },
    "node_modules/@typescript-eslint/typescript-estree/node_modules/minimatch": {
      "version": "9.0.5",
      "resolved": "https://registry.npmjs.org/minimatch/-/minimatch-9.0.5.tgz",
      "integrity": "sha512-G6T0ZX48xgozx7587koeX9Ys2NYy6Gmv//P89sEte9V9whIapMNF4idKxnW2QtCcLiTWlb/wfCabAtAFWhhBow==",
      "dev": true,
      "license": "ISC",
      "dependencies": {
        "brace-expansion": "^2.0.1"
      },
      "engines": {
        "node": ">=16 || 14 >=14.17"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/@typescript-eslint/typescript-estree/node_modules/semver": {
      "version": "7.7.3",
      "resolved": "https://registry.npmjs.org/semver/-/semver-7.7.3.tgz",
      "integrity": "sha512-SdsKMrI9TdgjdweUSR9MweHA4EJ8YxHn8DFaDisvhVlUOe4BF1tLD7GAj0lIqWVl+dPb/rExr0Btby5loQm20Q==",
      "dev": true,
      "license": "ISC",
      "bin": {
        "semver": "bin/semver.js"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/@typescript-eslint/utils": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/@typescript-eslint/utils/-/utils-8.53.1.tgz",
      "integrity": "sha512-c4bMvGVWW4hv6JmDUEG7fSYlWOl3II2I4ylt0NM+seinYQlZMQIaKaXIIVJWt9Ofh6whrpM+EdDQXKXjNovvrg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@eslint-community/eslint-utils": "^4.9.1",
        "@typescript-eslint/scope-manager": "8.53.1",
        "@typescript-eslint/types": "8.53.1",
        "@typescript-eslint/typescript-estree": "8.53.1"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      },
      "peerDependencies": {
        "eslint": "^8.57.0 || ^9.0.0",
        "typescript": ">=4.8.4 <6.0.0"
      }
    },
    "node_modules/@typescript-eslint/visitor-keys": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/@typescript-eslint/visitor-keys/-/visitor-keys-8.53.1.tgz",
      "integrity": "sha512-oy+wV7xDKFPRyNggmXuZQSBzvoLnpmJs+GhzRhPjrxl2b/jIlyjVokzm47CZCDUdXKr2zd7ZLodPfOBpOPyPlg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@typescript-eslint/types": "8.53.1",
        "eslint-visitor-keys": "^4.2.1"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      }
    },
    "node_modules/@vitejs/plugin-react": {
      "version": "5.1.2",
      "resolved": "https://registry.npmjs.org/@vitejs/plugin-react/-/plugin-react-5.1.2.tgz",
      "integrity": "sha512-EcA07pHJouywpzsoTUqNh5NwGayl2PPVEJKUSinGGSxFGYn+shYbqMGBg6FXDqgXum9Ou/ecb+411ssw8HImJQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/core": "^7.28.5",
        "@babel/plugin-transform-react-jsx-self": "^7.27.1",
        "@babel/plugin-transform-react-jsx-source": "^7.27.1",
        "@rolldown/pluginutils": "1.0.0-beta.53",
        "@types/babel__core": "^7.20.5",
        "react-refresh": "^0.18.0"
      },
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      },
      "peerDependencies": {
        "vite": "^4.2.0 || ^5.0.0 || ^6.0.0 || ^7.0.0"
      }
    },
    "node_modules/acorn": {
      "version": "8.15.0",
      "resolved": "https://registry.npmjs.org/acorn/-/acorn-8.15.0.tgz",
      "integrity": "sha512-NZyJarBfL7nWwIq+FDL6Zp/yHEhePMNnnJ0y3qfieCrmNvYct8uvtiV41UvlSe6apAfk0fY1FbWx+NwfmpvtTg==",
      "dev": true,
      "license": "MIT",
      "bin": {
        "acorn": "bin/acorn"
      },
      "engines": {
        "node": ">=0.4.0"
      }
    },
    "node_modules/acorn-jsx": {
      "version": "5.3.2",
      "resolved": "https://registry.npmjs.org/acorn-jsx/-/acorn-jsx-5.3.2.tgz",
      "integrity": "sha512-rq9s+JNhf0IChjtDXxllJ7g41oZk5SlXtp0LHwyA5cejwn7vKmKp4pPri6YEePv2PU65sAsegbXtIinmDFDXgQ==",
      "dev": true,
      "license": "MIT",
      "peerDependencies": {
        "acorn": "^6.0.0 || ^7.0.0 || ^8.0.0"
      }
    },
    "node_modules/ajv": {
      "version": "6.12.6",
      "resolved": "https://registry.npmjs.org/ajv/-/ajv-6.12.6.tgz",
      "integrity": "sha512-j3fVLgvTo527anyYyJOGTYJbG+vnnQYvE0m5mmkc1TK+nxAppkCLMIL0aZ4dblVCNoGShhm+kzE4ZUykBoMg4g==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "fast-deep-equal": "^3.1.1",
        "fast-json-stable-stringify": "^2.0.0",
        "json-schema-traverse": "^0.4.1",
        "uri-js": "^4.2.2"
      },
      "funding": {
        "type": "github",
        "url": "https://github.com/sponsors/epoberezkin"
      }
    },
    "node_modules/ansi-styles": {
      "version": "4.3.0",
      "resolved": "https://registry.npmjs.org/ansi-styles/-/ansi-styles-4.3.0.tgz",
      "integrity": "sha512-zbB9rCJAT1rbjiVDb2hqKFHNYLxgtk8NURxZ3IZwD3F6NtxbXZQCnnSi1Lkx+IDohdPlFp222wVALIheZJQSEg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "color-convert": "^2.0.1"
      },
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/chalk/ansi-styles?sponsor=1"
      }
    },
    "node_modules/argparse": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/argparse/-/argparse-2.0.1.tgz",
      "integrity": "sha512-8+9WqebbFzpX9OR+Wa6O29asIogeRMzcGtAINdpMHHyAg10f05aSFVBbcEqGf/PXw1EjAZ+q2/bEBg3DvurK3Q==",
      "dev": true,
      "license": "Python-2.0"
    },
    "node_modules/balanced-match": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/balanced-match/-/balanced-match-1.0.2.tgz",
      "integrity": "sha512-3oSeUO0TMV67hN1AmbXsK4yaqU7tjiHlbxRDZOpH0KW9+CeX4bRAaX0Anxt0tx2MrpRpWwQaPwIlISEJhYU5Pw==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/baseline-browser-mapping": {
      "version": "2.9.17",
      "resolved": "https://registry.npmjs.org/baseline-browser-mapping/-/baseline-browser-mapping-2.9.17.tgz",
      "integrity": "sha512-agD0MgJFUP/4nvjqzIB29zRPUuCF7Ge6mEv9s8dHrtYD7QWXRcx75rOADE/d5ah1NI+0vkDl0yorDd5U852IQQ==",
      "dev": true,
      "license": "Apache-2.0",
      "bin": {
        "baseline-browser-mapping": "dist/cli.js"
      }
    },
    "node_modules/brace-expansion": {
      "version": "1.1.12",
      "resolved": "https://registry.npmjs.org/brace-expansion/-/brace-expansion-1.1.12.tgz",
      "integrity": "sha512-9T9UjW3r0UW5c1Q7GTwllptXwhvYmEzFhzMfZ9H7FQWt+uZePjZPjBP/W1ZEyZ1twGWom5/56TF4lPcqjnDHcg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "balanced-match": "^1.0.0",
        "concat-map": "0.0.1"
      }
    },
    "node_modules/browserslist": {
      "version": "4.28.1",
      "resolved": "https://registry.npmjs.org/browserslist/-/browserslist-4.28.1.tgz",
      "integrity": "sha512-ZC5Bd0LgJXgwGqUknZY/vkUQ04r8NXnJZ3yYi4vDmSiZmC/pdSN0NbNRPxZpbtO4uAfDUAFffO8IZoM3Gj8IkA==",
      "dev": true,
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/browserslist"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/browserslist"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "license": "MIT",
      "dependencies": {
        "baseline-browser-mapping": "^2.9.0",
        "caniuse-lite": "^1.0.30001759",
        "electron-to-chromium": "^1.5.263",
        "node-releases": "^2.0.27",
        "update-browserslist-db": "^1.2.0"
      },
      "bin": {
        "browserslist": "cli.js"
      },
      "engines": {
        "node": "^6 || ^7 || ^8 || ^9 || ^10 || ^11 || ^12 || >=13.7"
      }
    },
    "node_modules/callsites": {
      "version": "3.1.0",
      "resolved": "https://registry.npmjs.org/callsites/-/callsites-3.1.0.tgz",
      "integrity": "sha512-P8BjAsXvZS+VIDUI11hHCQEv74YT67YUi5JJFNWIqL235sBmjX4+qx9Muvls5ivyNENctx46xQLQ3aTuE7ssaQ==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/caniuse-lite": {
      "version": "1.0.30001766",
      "resolved": "https://registry.npmjs.org/caniuse-lite/-/caniuse-lite-1.0.30001766.tgz",
      "integrity": "sha512-4C0lfJ0/YPjJQHagaE9x2Elb69CIqEPZeG0anQt9SIvIoOH4a4uaRl73IavyO+0qZh6MDLH//DrXThEYKHkmYA==",
      "dev": true,
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/browserslist"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/caniuse-lite"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "license": "CC-BY-4.0"
    },
    "node_modules/chalk": {
      "version": "4.1.2",
      "resolved": "https://registry.npmjs.org/chalk/-/chalk-4.1.2.tgz",
      "integrity": "sha512-oKnbhFyRIXpUuez8iBMmyEa4nbj4IOQyuhc/wy9kY7/WVPcwIO9VA668Pu8RkO7+0G76SLROeyw9CpQ061i4mA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "ansi-styles": "^4.1.0",
        "supports-color": "^7.1.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/chalk/chalk?sponsor=1"
      }
    },
    "node_modules/color-convert": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/color-convert/-/color-convert-2.0.1.tgz",
      "integrity": "sha512-RRECPsj7iu/xb5oKYcsFHSppFNnsj/52OVTRKb4zP5onXwVF3zVmmToNcOfGC+CRDpfK/U584fMg38ZHCaElKQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "color-name": "~1.1.4"
      },
      "engines": {
        "node": ">=7.0.0"
      }
    },
    "node_modules/color-name": {
      "version": "1.1.4",
      "resolved": "https://registry.npmjs.org/color-name/-/color-name-1.1.4.tgz",
      "integrity": "sha512-dOy+3AuW3a2wNbZHIuMZpTcgjGuLU/uBL/ubcZF9OXbDo8ff4O8yVp5Bf0efS8uEoYo5q4Fx7dY9OgQGXgAsQA==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/concat-map": {
      "version": "0.0.1",
      "resolved": "https://registry.npmjs.org/concat-map/-/concat-map-0.0.1.tgz",
      "integrity": "sha512-/Srv4dswyQNBfohGpz9o6Yb3Gz3SrUDqBH5rTuhGR7ahtlbYKnVxw2bCFMRljaA7EXHaXZ8wsHdodFvbkhKmqg==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/convert-source-map": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/convert-source-map/-/convert-source-map-2.0.0.tgz",
      "integrity": "sha512-Kvp459HrV2FEJ1CAsi1Ku+MY3kasH19TFykTz2xWmMeq6bk2NU3XXvfJ+Q61m0xktWwt+1HSYf3JZsTms3aRJg==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/cross-spawn": {
      "version": "7.0.6",
      "resolved": "https://registry.npmjs.org/cross-spawn/-/cross-spawn-7.0.6.tgz",
      "integrity": "sha512-uV2QOWP2nWzsy2aMp8aRibhi9dlzF5Hgh5SHaB9OiTGEyDTiJJyx0uy51QXdyWbtAHNua4XJzUKca3OzKUd3vA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "path-key": "^3.1.0",
        "shebang-command": "^2.0.0",
        "which": "^2.0.1"
      },
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/csstype": {
      "version": "3.2.3",
      "resolved": "https://registry.npmjs.org/csstype/-/csstype-3.2.3.tgz",
      "integrity": "sha512-z1HGKcYy2xA8AGQfwrn0PAy+PB7X/GSj3UVJW9qKyn43xWa+gl5nXmU4qqLMRzWVLFC8KusUX8T/0kCiOYpAIQ==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/debug": {
      "version": "4.4.3",
      "resolved": "https://registry.npmjs.org/debug/-/debug-4.4.3.tgz",
      "integrity": "sha512-RGwwWnwQvkVfavKVt22FGLw+xYSdzARwm0ru6DhTVA3umU5hZc28V3kO4stgYryrTlLpuvgI9GiijltAjNbcqA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "ms": "^2.1.3"
      },
      "engines": {
        "node": ">=6.0"
      },
      "peerDependenciesMeta": {
        "supports-color": {
          "optional": true
        }
      }
    },
    "node_modules/deep-is": {
      "version": "0.1.4",
      "resolved": "https://registry.npmjs.org/deep-is/-/deep-is-0.1.4.tgz",
      "integrity": "sha512-oIPzksmTg4/MriiaYGO+okXDT7ztn/w3Eptv/+gSIdMdKsJo0u4CfYNFJPy+4SKMuCqGw2wxnA+URMg3t8a/bQ==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/electron-to-chromium": {
      "version": "1.5.278",
      "resolved": "https://registry.npmjs.org/electron-to-chromium/-/electron-to-chromium-1.5.278.tgz",
      "integrity": "sha512-dQ0tM1svDRQOwxnXxm+twlGTjr9Upvt8UFWAgmLsxEzFQxhbti4VwxmMjsDxVC51Zo84swW7FVCXEV+VAkhuPw==",
      "dev": true,
      "license": "ISC"
    },
    "node_modules/esbuild": {
      "version": "0.27.2",
      "resolved": "https://registry.npmjs.org/esbuild/-/esbuild-0.27.2.tgz",
      "integrity": "sha512-HyNQImnsOC7X9PMNaCIeAm4ISCQXs5a5YasTXVliKv4uuBo1dKrG0A+uQS8M5eXjVMnLg3WgXaKvprHlFJQffw==",
      "dev": true,
      "hasInstallScript": true,
      "license": "MIT",
      "bin": {
        "esbuild": "bin/esbuild"
      },
      "engines": {
        "node": ">=18"
      },
      "optionalDependencies": {
        "@esbuild/aix-ppc64": "0.27.2",
        "@esbuild/android-arm": "0.27.2",
        "@esbuild/android-arm64": "0.27.2",
        "@esbuild/android-x64": "0.27.2",
        "@esbuild/darwin-arm64": "0.27.2",
        "@esbuild/darwin-x64": "0.27.2",
        "@esbuild/freebsd-arm64": "0.27.2",
        "@esbuild/freebsd-x64": "0.27.2",
        "@esbuild/linux-arm": "0.27.2",
        "@esbuild/linux-arm64": "0.27.2",
        "@esbuild/linux-ia32": "0.27.2",
        "@esbuild/linux-loong64": "0.27.2",
        "@esbuild/linux-mips64el": "0.27.2",
        "@esbuild/linux-ppc64": "0.27.2",
        "@esbuild/linux-riscv64": "0.27.2",
        "@esbuild/linux-s390x": "0.27.2",
        "@esbuild/linux-x64": "0.27.2",
        "@esbuild/netbsd-arm64": "0.27.2",
        "@esbuild/netbsd-x64": "0.27.2",
        "@esbuild/openbsd-arm64": "0.27.2",
        "@esbuild/openbsd-x64": "0.27.2",
        "@esbuild/openharmony-arm64": "0.27.2",
        "@esbuild/sunos-x64": "0.27.2",
        "@esbuild/win32-arm64": "0.27.2",
        "@esbuild/win32-ia32": "0.27.2",
        "@esbuild/win32-x64": "0.27.2"
      }
    },
    "node_modules/escalade": {
      "version": "3.2.0",
      "resolved": "https://registry.npmjs.org/escalade/-/escalade-3.2.0.tgz",
      "integrity": "sha512-WUj2qlxaQtO4g6Pq5c29GTcWGDyd8itL8zTlipgECz3JesAiiOKotd8JU6otB3PACgG6xkJUyVhboMS+bje/jA==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/escape-string-regexp": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/escape-string-regexp/-/escape-string-regexp-4.0.0.tgz",
      "integrity": "sha512-TtpcNJ3XAzx3Gq8sWRzJaVajRs0uVxA2YAkdb1jm2YkPz4G6egUFAyA3n5vtEIZefPk5Wa4UXbKuS5fKkJWdgA==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/eslint": {
      "version": "9.39.2",
      "resolved": "https://registry.npmjs.org/eslint/-/eslint-9.39.2.tgz",
      "integrity": "sha512-LEyamqS7W5HB3ujJyvi0HQK/dtVINZvd5mAAp9eT5S/ujByGjiZLCzPcHVzuXbpJDJF/cxwHlfceVUDZ2lnSTw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@eslint-community/eslint-utils": "^4.8.0",
        "@eslint-community/regexpp": "^4.12.1",
        "@eslint/config-array": "^0.21.1",
        "@eslint/config-helpers": "^0.4.2",
        "@eslint/core": "^0.17.0",
        "@eslint/eslintrc": "^3.3.1",
        "@eslint/js": "9.39.2",
        "@eslint/plugin-kit": "^0.4.1",
        "@humanfs/node": "^0.16.6",
        "@humanwhocodes/module-importer": "^1.0.1",
        "@humanwhocodes/retry": "^0.4.2",
        "@types/estree": "^1.0.6",
        "ajv": "^6.12.4",
        "chalk": "^4.0.0",
        "cross-spawn": "^7.0.6",
        "debug": "^4.3.2",
        "escape-string-regexp": "^4.0.0",
        "eslint-scope": "^8.4.0",
        "eslint-visitor-keys": "^4.2.1",
        "espree": "^10.4.0",
        "esquery": "^1.5.0",
        "esutils": "^2.0.2",
        "fast-deep-equal": "^3.1.3",
        "file-entry-cache": "^8.0.0",
        "find-up": "^5.0.0",
        "glob-parent": "^6.0.2",
        "ignore": "^5.2.0",
        "imurmurhash": "^0.1.4",
        "is-glob": "^4.0.0",
        "json-stable-stringify-without-jsonify": "^1.0.1",
        "lodash.merge": "^4.6.2",
        "minimatch": "^3.1.2",
        "natural-compare": "^1.4.0",
        "optionator": "^0.9.3"
      },
      "bin": {
        "eslint": "bin/eslint.js"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "url": "https://eslint.org/donate"
      },
      "peerDependencies": {
        "jiti": "*"
      },
      "peerDependenciesMeta": {
        "jiti": {
          "optional": true
        }
      }
    },
    "node_modules/eslint-plugin-react-hooks": {
      "version": "7.0.1",
      "resolved": "https://registry.npmjs.org/eslint-plugin-react-hooks/-/eslint-plugin-react-hooks-7.0.1.tgz",
      "integrity": "sha512-O0d0m04evaNzEPoSW+59Mezf8Qt0InfgGIBJnpC0h3NH/WjUAR7BIKUfysC6todmtiZ/A0oUVS8Gce0WhBrHsA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@babel/core": "^7.24.4",
        "@babel/parser": "^7.24.4",
        "hermes-parser": "^0.25.1",
        "zod": "^3.25.0 || ^4.0.0",
        "zod-validation-error": "^3.5.0 || ^4.0.0"
      },
      "engines": {
        "node": ">=18"
      },
      "peerDependencies": {
        "eslint": "^3.0.0 || ^4.0.0 || ^5.0.0 || ^6.0.0 || ^7.0.0 || ^8.0.0-0 || ^9.0.0"
      }
    },
    "node_modules/eslint-plugin-react-refresh": {
      "version": "0.4.26",
      "resolved": "https://registry.npmjs.org/eslint-plugin-react-refresh/-/eslint-plugin-react-refresh-0.4.26.tgz",
      "integrity": "sha512-1RETEylht2O6FM/MvgnyvT+8K21wLqDNg4qD51Zj3guhjt433XbnnkVttHMyaVyAFD03QSV4LPS5iE3VQmO7XQ==",
      "dev": true,
      "license": "MIT",
      "peerDependencies": {
        "eslint": ">=8.40"
      }
    },
    "node_modules/eslint-scope": {
      "version": "8.4.0",
      "resolved": "https://registry.npmjs.org/eslint-scope/-/eslint-scope-8.4.0.tgz",
      "integrity": "sha512-sNXOfKCn74rt8RICKMvJS7XKV/Xk9kA7DyJr8mJik3S7Cwgy3qlkkmyS2uQB3jiJg6VNdZd/pDBJu0nvG2NlTg==",
      "dev": true,
      "license": "BSD-2-Clause",
      "dependencies": {
        "esrecurse": "^4.3.0",
        "estraverse": "^5.2.0"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "url": "https://opencollective.com/eslint"
      }
    },
    "node_modules/eslint-visitor-keys": {
      "version": "4.2.1",
      "resolved": "https://registry.npmjs.org/eslint-visitor-keys/-/eslint-visitor-keys-4.2.1.tgz",
      "integrity": "sha512-Uhdk5sfqcee/9H/rCOJikYz67o0a2Tw2hGRPOG2Y1R2dg7brRe1uG0yaNQDHu+TO/uQPF/5eCapvYSmHUjt7JQ==",
      "dev": true,
      "license": "Apache-2.0",
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "url": "https://opencollective.com/eslint"
      }
    },
    "node_modules/espree": {
      "version": "10.4.0",
      "resolved": "https://registry.npmjs.org/espree/-/espree-10.4.0.tgz",
      "integrity": "sha512-j6PAQ2uUr79PZhBjP5C5fhl8e39FmRnOjsD5lGnWrFU8i2G776tBK7+nP8KuQUTTyAZUwfQqXAgrVH5MbH9CYQ==",
      "dev": true,
      "license": "BSD-2-Clause",
      "dependencies": {
        "acorn": "^8.15.0",
        "acorn-jsx": "^5.3.2",
        "eslint-visitor-keys": "^4.2.1"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "url": "https://opencollective.com/eslint"
      }
    },
    "node_modules/esquery": {
      "version": "1.7.0",
      "resolved": "https://registry.npmjs.org/esquery/-/esquery-1.7.0.tgz",
      "integrity": "sha512-Ap6G0WQwcU/LHsvLwON1fAQX9Zp0A2Y6Y/cJBl9r/JbW90Zyg4/zbG6zzKa2OTALELarYHmKu0GhpM5EO+7T0g==",
      "dev": true,
      "license": "BSD-3-Clause",
      "dependencies": {
        "estraverse": "^5.1.0"
      },
      "engines": {
        "node": ">=0.10"
      }
    },
    "node_modules/esrecurse": {
      "version": "4.3.0",
      "resolved": "https://registry.npmjs.org/esrecurse/-/esrecurse-4.3.0.tgz",
      "integrity": "sha512-KmfKL3b6G+RXvP8N1vr3Tq1kL/oCFgn2NYXEtqP8/L3pKapUA4G8cFVaoF3SU323CD4XypR/ffioHmkti6/Tag==",
      "dev": true,
      "license": "BSD-2-Clause",
      "dependencies": {
        "estraverse": "^5.2.0"
      },
      "engines": {
        "node": ">=4.0"
      }
    },
    "node_modules/estraverse": {
      "version": "5.3.0",
      "resolved": "https://registry.npmjs.org/estraverse/-/estraverse-5.3.0.tgz",
      "integrity": "sha512-MMdARuVEQziNTeJD8DgMqmhwR11BRQ/cBP+pLtYdSTnf3MIO8fFeiINEbX36ZdNlfU/7A9f3gUw49B3oQsvwBA==",
      "dev": true,
      "license": "BSD-2-Clause",
      "engines": {
        "node": ">=4.0"
      }
    },
    "node_modules/esutils": {
      "version": "2.0.3",
      "resolved": "https://registry.npmjs.org/esutils/-/esutils-2.0.3.tgz",
      "integrity": "sha512-kVscqXk4OCp68SZ0dkgEKVi6/8ij300KBWTJq32P/dYeWTSwK41WyTxalN1eRmA5Z9UU/LX9D7FWSmV9SAYx6g==",
      "dev": true,
      "license": "BSD-2-Clause",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/fast-deep-equal": {
      "version": "3.1.3",
      "resolved": "https://registry.npmjs.org/fast-deep-equal/-/fast-deep-equal-3.1.3.tgz",
      "integrity": "sha512-f3qQ9oQy9j2AhBe/H9VC91wLmKBCCU/gDOnKNAYG5hswO7BLKj09Hc5HYNz9cGI++xlpDCIgDaitVs03ATR84Q==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/fast-json-stable-stringify": {
      "version": "2.1.0",
      "resolved": "https://registry.npmjs.org/fast-json-stable-stringify/-/fast-json-stable-stringify-2.1.0.tgz",
      "integrity": "sha512-lhd/wF+Lk98HZoTCtlVraHtfh5XYijIjalXck7saUtuanSDyLMxnHhSXEDJqHxD7msR8D0uCmqlkwjCV8xvwHw==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/fast-levenshtein": {
      "version": "2.0.6",
      "resolved": "https://registry.npmjs.org/fast-levenshtein/-/fast-levenshtein-2.0.6.tgz",
      "integrity": "sha512-DCXu6Ifhqcks7TZKY3Hxp3y6qphY5SJZmrWMDrKcERSOXWQdMhU9Ig/PYrzyw/ul9jOIyh0N4M0tbC5hodg8dw==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/fdir": {
      "version": "6.5.0",
      "resolved": "https://registry.npmjs.org/fdir/-/fdir-6.5.0.tgz",
      "integrity": "sha512-tIbYtZbucOs0BRGqPJkshJUYdL+SDH7dVM8gjy+ERp3WAUjLEFJE+02kanyHtwjWOnwrKYBiwAmM0p4kLJAnXg==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=12.0.0"
      },
      "peerDependencies": {
        "picomatch": "^3 || ^4"
      },
      "peerDependenciesMeta": {
        "picomatch": {
          "optional": true
        }
      }
    },
    "node_modules/file-entry-cache": {
      "version": "8.0.0",
      "resolved": "https://registry.npmjs.org/file-entry-cache/-/file-entry-cache-8.0.0.tgz",
      "integrity": "sha512-XXTUwCvisa5oacNGRP9SfNtYBNAMi+RPwBFmblZEF7N7swHYQS6/Zfk7SRwx4D5j3CH211YNRco1DEMNVfZCnQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "flat-cache": "^4.0.0"
      },
      "engines": {
        "node": ">=16.0.0"
      }
    },
    "node_modules/find-up": {
      "version": "5.0.0",
      "resolved": "https://registry.npmjs.org/find-up/-/find-up-5.0.0.tgz",
      "integrity": "sha512-78/PXT1wlLLDgTzDs7sjq9hzz0vXD+zn+7wypEe4fXQxCmdmqfGsEPQxmiCSQI3ajFV91bVSsvNtrJRiW6nGng==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "locate-path": "^6.0.0",
        "path-exists": "^4.0.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/flat-cache": {
      "version": "4.0.1",
      "resolved": "https://registry.npmjs.org/flat-cache/-/flat-cache-4.0.1.tgz",
      "integrity": "sha512-f7ccFPK3SXFHpx15UIGyRJ/FJQctuKZ0zVuN3frBo4HnK3cay9VEW0R6yPYFHC0AgqhukPzKjq22t5DmAyqGyw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "flatted": "^3.2.9",
        "keyv": "^4.5.4"
      },
      "engines": {
        "node": ">=16"
      }
    },
    "node_modules/flatted": {
      "version": "3.3.3",
      "resolved": "https://registry.npmjs.org/flatted/-/flatted-3.3.3.tgz",
      "integrity": "sha512-GX+ysw4PBCz0PzosHDepZGANEuFCMLrnRTiEy9McGjmkCQYwRq4A/X786G/fjM/+OjsWSU1ZrY5qyARZmO/uwg==",
      "dev": true,
      "license": "ISC"
    },
    "node_modules/fsevents": {
      "version": "2.3.3",
      "resolved": "https://registry.npmjs.org/fsevents/-/fsevents-2.3.3.tgz",
      "integrity": "sha512-5xoDfX+fL7faATnagmWPpbFtwh/R77WmMMqqHGS65C3vvB0YHrgF+B1YmZ3441tMj5n63k0212XNoJwzlhffQw==",
      "dev": true,
      "hasInstallScript": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": "^8.16.0 || ^10.6.0 || >=11.0.0"
      }
    },
    "node_modules/gensync": {
      "version": "1.0.0-beta.2",
      "resolved": "https://registry.npmjs.org/gensync/-/gensync-1.0.0-beta.2.tgz",
      "integrity": "sha512-3hN7NaskYvMDLQY55gnW3NQ+mesEAepTqlg+VEbj7zzqEMBVNhzcGYYeqFo/TlYz6eQiFcp1HcsCZO+nGgS8zg==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/glob-parent": {
      "version": "6.0.2",
      "resolved": "https://registry.npmjs.org/glob-parent/-/glob-parent-6.0.2.tgz",
      "integrity": "sha512-XxwI8EOhVQgWp6iDL+3b0r86f4d6AX6zSU55HfB4ydCEuXLXc5FcYeOu+nnGftS4TEju/11rt4KJPTMgbfmv4A==",
      "dev": true,
      "license": "ISC",
      "dependencies": {
        "is-glob": "^4.0.3"
      },
      "engines": {
        "node": ">=10.13.0"
      }
    },
    "node_modules/globals": {
      "version": "16.5.0",
      "resolved": "https://registry.npmjs.org/globals/-/globals-16.5.0.tgz",
      "integrity": "sha512-c/c15i26VrJ4IRt5Z89DnIzCGDn9EcebibhAOjw5ibqEHsE1wLUgkPn9RDmNcUKyU87GeaL633nyJ+pplFR2ZQ==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=18"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/has-flag": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/has-flag/-/has-flag-4.0.0.tgz",
      "integrity": "sha512-EykJT/Q1KjTWctppgIAgfSO0tKVuZUjhgMr17kqTumMl6Afv3EISleU7qZUzoXDFTAHTDC4NOoG/ZxU3EvlMPQ==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/hermes-estree": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/hermes-estree/-/hermes-estree-0.25.1.tgz",
      "integrity": "sha512-0wUoCcLp+5Ev5pDW2OriHC2MJCbwLwuRx+gAqMTOkGKJJiBCLjtrvy4PWUGn6MIVefecRpzoOZ/UV6iGdOr+Cw==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/hermes-parser": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/hermes-parser/-/hermes-parser-0.25.1.tgz",
      "integrity": "sha512-6pEjquH3rqaI6cYAXYPcz9MS4rY6R4ngRgrgfDshRptUZIc3lw0MCIJIGDj9++mfySOuPTHB4nrSW99BCvOPIA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "hermes-estree": "0.25.1"
      }
    },
    "node_modules/ignore": {
      "version": "5.3.2",
      "resolved": "https://registry.npmjs.org/ignore/-/ignore-5.3.2.tgz",
      "integrity": "sha512-hsBTNUqQTDwkWtcdYI2i06Y/nUBEsNEDJKjWdigLvegy8kDuJAS8uRlpkkcQpyEXL0Z/pjDy5HBmMjRCJ2gq+g==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">= 4"
      }
    },
    "node_modules/import-fresh": {
      "version": "3.3.1",
      "resolved": "https://registry.npmjs.org/import-fresh/-/import-fresh-3.3.1.tgz",
      "integrity": "sha512-TR3KfrTZTYLPB6jUjfx6MF9WcWrHL9su5TObK4ZkYgBdWKPOFoSoQIdEuTuR82pmtxH2spWG9h6etwfr1pLBqQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "parent-module": "^1.0.0",
        "resolve-from": "^4.0.0"
      },
      "engines": {
        "node": ">=6"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/imurmurhash": {
      "version": "0.1.4",
      "resolved": "https://registry.npmjs.org/imurmurhash/-/imurmurhash-0.1.4.tgz",
      "integrity": "sha512-JmXMZ6wuvDmLiHEml9ykzqO6lwFbof0GG4IkcGaENdCRDDmMVnny7s5HsIgHCbaq0w2MyPhDqkhTUgS2LU2PHA==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=0.8.19"
      }
    },
    "node_modules/is-extglob": {
      "version": "2.1.1",
      "resolved": "https://registry.npmjs.org/is-extglob/-/is-extglob-2.1.1.tgz",
      "integrity": "sha512-SbKbANkN603Vi4jEZv49LeVJMn4yGwsbzZworEoyEiutsN3nJYdbO36zfhGJ6QEDpOZIFkDtnq5JRxmvl3jsoQ==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/is-glob": {
      "version": "4.0.3",
      "resolved": "https://registry.npmjs.org/is-glob/-/is-glob-4.0.3.tgz",
      "integrity": "sha512-xelSayHH36ZgE7ZWhli7pW34hNbNl8Ojv5KVmkJD4hBdD3th8Tfk9vYasLM+mXWOZhFkgZfxhLSnrwRr4elSSg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "is-extglob": "^2.1.1"
      },
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/isexe": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/isexe/-/isexe-2.0.0.tgz",
      "integrity": "sha512-RHxMLp9lnKHGHRng9QFhRCMbYAcVpn69smSGcq3f36xjgVVWThj4qqLbTLlq7Ssj8B+fIQ1EuCEGI2lKsyQeIw==",
      "dev": true,
      "license": "ISC"
    },
    "node_modules/js-tokens": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/js-tokens/-/js-tokens-4.0.0.tgz",
      "integrity": "sha512-RdJUflcE3cUzKiMqQgsCu06FPu9UdIJO0beYbPhHN4k6apgJtifcoCtT9bcxOpYBtpD2kCM6Sbzg4CausW/PKQ==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/js-yaml": {
      "version": "4.1.1",
      "resolved": "https://registry.npmjs.org/js-yaml/-/js-yaml-4.1.1.tgz",
      "integrity": "sha512-qQKT4zQxXl8lLwBtHMWwaTcGfFOZviOJet3Oy/xmGk2gZH677CJM9EvtfdSkgWcATZhj/55JZ0rmy3myCT5lsA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "argparse": "^2.0.1"
      },
      "bin": {
        "js-yaml": "bin/js-yaml.js"
      }
    },
    "node_modules/jsesc": {
      "version": "3.1.0",
      "resolved": "https://registry.npmjs.org/jsesc/-/jsesc-3.1.0.tgz",
      "integrity": "sha512-/sM3dO2FOzXjKQhJuo0Q173wf2KOo8t4I8vHy6lF9poUp7bKT0/NHE8fPX23PwfhnykfqnC2xRxOnVw5XuGIaA==",
      "dev": true,
      "license": "MIT",
      "bin": {
        "jsesc": "bin/jsesc"
      },
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/json-buffer": {
      "version": "3.0.1",
      "resolved": "https://registry.npmjs.org/json-buffer/-/json-buffer-3.0.1.tgz",
      "integrity": "sha512-4bV5BfR2mqfQTJm+V5tPPdf+ZpuhiIvTuAB5g8kcrXOZpTT/QwwVRWBywX1ozr6lEuPdbHxwaJlm9G6mI2sfSQ==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/json-schema-traverse": {
      "version": "0.4.1",
      "resolved": "https://registry.npmjs.org/json-schema-traverse/-/json-schema-traverse-0.4.1.tgz",
      "integrity": "sha512-xbbCH5dCYU5T8LcEhhuh7HJ88HXuW3qsI3Y0zOZFKfZEHcpWiHU/Jxzk629Brsab/mMiHQti9wMP+845RPe3Vg==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/json-stable-stringify-without-jsonify": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/json-stable-stringify-without-jsonify/-/json-stable-stringify-without-jsonify-1.0.1.tgz",
      "integrity": "sha512-Bdboy+l7tA3OGW6FjyFHWkP5LuByj1Tk33Ljyq0axyzdk9//JSi2u3fP1QSmd1KNwq6VOKYGlAu87CisVir6Pw==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/json5": {
      "version": "2.2.3",
      "resolved": "https://registry.npmjs.org/json5/-/json5-2.2.3.tgz",
      "integrity": "sha512-XmOWe7eyHYH14cLdVPoyg+GOH3rYX++KpzrylJwSW98t3Nk+U8XOl8FWKOgwtzdb8lXGf6zYwDUzeHMWfxasyg==",
      "dev": true,
      "license": "MIT",
      "bin": {
        "json5": "lib/cli.js"
      },
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/keyv": {
      "version": "4.5.4",
      "resolved": "https://registry.npmjs.org/keyv/-/keyv-4.5.4.tgz",
      "integrity": "sha512-oxVHkHR/EJf2CNXnWxRLW6mg7JyCCUcG0DtEGmL2ctUo1PNTin1PUil+r/+4r5MpVgC/fn1kjsx7mjSujKqIpw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "json-buffer": "3.0.1"
      }
    },
    "node_modules/levn": {
      "version": "0.4.1",
      "resolved": "https://registry.npmjs.org/levn/-/levn-0.4.1.tgz",
      "integrity": "sha512-+bT2uH4E5LGE7h/n3evcS/sQlJXCpIp6ym8OWJ5eV6+67Dsql/LaaT7qJBAt2rzfoa/5QBGBhxDix1dMt2kQKQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "prelude-ls": "^1.2.1",
        "type-check": "~0.4.0"
      },
      "engines": {
        "node": ">= 0.8.0"
      }
    },
    "node_modules/locate-path": {
      "version": "6.0.0",
      "resolved": "https://registry.npmjs.org/locate-path/-/locate-path-6.0.0.tgz",
      "integrity": "sha512-iPZK6eYjbxRu3uB4/WZ3EsEIMJFMqAoopl3R+zuq0UjcAm/MO6KCweDgPfP3elTztoKP3KtnVHxTn2NHBSDVUw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "p-locate": "^5.0.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/lodash.merge": {
      "version": "4.6.2",
      "resolved": "https://registry.npmjs.org/lodash.merge/-/lodash.merge-4.6.2.tgz",
      "integrity": "sha512-0KpjqXRVvrYyCsX1swR/XTK0va6VQkQM6MNo7PqW77ByjAhoARA8EfrP1N4+KlKj8YS0ZUCtRT/YUuhyYDujIQ==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/lru-cache": {
      "version": "5.1.1",
      "resolved": "https://registry.npmjs.org/lru-cache/-/lru-cache-5.1.1.tgz",
      "integrity": "sha512-KpNARQA3Iwv+jTA0utUVVbrh+Jlrr1Fv0e56GGzAFOXN7dk/FviaDW8LHmK52DlcH4WP2n6gI8vN1aesBFgo9w==",
      "dev": true,
      "license": "ISC",
      "dependencies": {
        "yallist": "^3.0.2"
      }
    },
    "node_modules/minimatch": {
      "version": "3.1.2",
      "resolved": "https://registry.npmjs.org/minimatch/-/minimatch-3.1.2.tgz",
      "integrity": "sha512-J7p63hRiAjw1NDEww1W7i37+ByIrOWO5XQQAzZ3VOcL0PNybwpfmV/N05zFAzwQ9USyEcX6t3UO+K5aqBQOIHw==",
      "dev": true,
      "license": "ISC",
      "dependencies": {
        "brace-expansion": "^1.1.7"
      },
      "engines": {
        "node": "*"
      }
    },
    "node_modules/ms": {
      "version": "2.1.3",
      "resolved": "https://registry.npmjs.org/ms/-/ms-2.1.3.tgz",
      "integrity": "sha512-6FlzubTLZG3J2a/NVCAleEhjzq5oxgHyaCU9yYXvcLsvoVaHJq/s5xXI6/XXP6tz7R9xAOtHnSO/tXtF3WRTlA==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/nanoid": {
      "version": "3.3.11",
      "resolved": "https://registry.npmjs.org/nanoid/-/nanoid-3.3.11.tgz",
      "integrity": "sha512-N8SpfPUnUp1bK+PMYW8qSWdl9U+wwNWI4QKxOYDy9JAro3WMX7p2OeVRF9v+347pnakNevPmiHhNmZ2HbFA76w==",
      "dev": true,
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "license": "MIT",
      "bin": {
        "nanoid": "bin/nanoid.cjs"
      },
      "engines": {
        "node": "^10 || ^12 || ^13.7 || ^14 || >=15.0.1"
      }
    },
    "node_modules/natural-compare": {
      "version": "1.4.0",
      "resolved": "https://registry.npmjs.org/natural-compare/-/natural-compare-1.4.0.tgz",
      "integrity": "sha512-OWND8ei3VtNC9h7V60qff3SVobHr996CTwgxubgyQYEpg290h9J0buyECNNJexkFm5sOajh5G116RYA1c8ZMSw==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/node-releases": {
      "version": "2.0.27",
      "resolved": "https://registry.npmjs.org/node-releases/-/node-releases-2.0.27.tgz",
      "integrity": "sha512-nmh3lCkYZ3grZvqcCH+fjmQ7X+H0OeZgP40OierEaAptX4XofMh5kwNbWh7lBduUzCcV/8kZ+NDLCwm2iorIlA==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/optionator": {
      "version": "0.9.4",
      "resolved": "https://registry.npmjs.org/optionator/-/optionator-0.9.4.tgz",
      "integrity": "sha512-6IpQ7mKUxRcZNLIObR0hz7lxsapSSIYNZJwXPGeF0mTVqGKFIXj1DQcMoT22S3ROcLyY/rz0PWaWZ9ayWmad9g==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "deep-is": "^0.1.3",
        "fast-levenshtein": "^2.0.6",
        "levn": "^0.4.1",
        "prelude-ls": "^1.2.1",
        "type-check": "^0.4.0",
        "word-wrap": "^1.2.5"
      },
      "engines": {
        "node": ">= 0.8.0"
      }
    },
    "node_modules/p-limit": {
      "version": "3.1.0",
      "resolved": "https://registry.npmjs.org/p-limit/-/p-limit-3.1.0.tgz",
      "integrity": "sha512-TYOanM3wGwNGsZN2cVTYPArw454xnXj5qmWF1bEoAc4+cU/ol7GVh7odevjp1FNHduHc3KZMcFduxU5Xc6uJRQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "yocto-queue": "^0.1.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/p-locate": {
      "version": "5.0.0",
      "resolved": "https://registry.npmjs.org/p-locate/-/p-locate-5.0.0.tgz",
      "integrity": "sha512-LaNjtRWUBY++zB5nE/NwcaoMylSPk+S+ZHNB1TzdbMJMny6dynpAGt7X/tl/QYq3TIeE6nxHppbo2LGymrG5Pw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "p-limit": "^3.0.2"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/parent-module": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/parent-module/-/parent-module-1.0.1.tgz",
      "integrity": "sha512-GQ2EWRpQV8/o+Aw8YqtfZZPfNRWZYkbidE9k5rpl/hC3vtHHBfGm2Ifi6qWV+coDGkrUKZAxE3Lot5kcsRlh+g==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "callsites": "^3.0.0"
      },
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/path-exists": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/path-exists/-/path-exists-4.0.0.tgz",
      "integrity": "sha512-ak9Qy5Q7jYb2Wwcey5Fpvg2KoAc/ZIhLSLOSBmRmygPsGwkVVt0fZa0qrtMz+m6tJTAHfZQ8FnmB4MG4LWy7/w==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/path-key": {
      "version": "3.1.1",
      "resolved": "https://registry.npmjs.org/path-key/-/path-key-3.1.1.tgz",
      "integrity": "sha512-ojmeN0qd+y0jszEtoY48r0Peq5dwMEkIlCOu6Q5f41lfkswXuKtYrhgoTpLnyIcHm24Uhqx+5Tqm2InSwLhE6Q==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/picocolors": {
      "version": "1.1.1",
      "resolved": "https://registry.npmjs.org/picocolors/-/picocolors-1.1.1.tgz",
      "integrity": "sha512-xceH2snhtb5M9liqDsmEw56le376mTZkEX/jEb/RxNFyegNul7eNslCXP9FDj/Lcu0X8KEyMceP2ntpaHrDEVA==",
      "dev": true,
      "license": "ISC"
    },
    "node_modules/picomatch": {
      "version": "4.0.3",
      "resolved": "https://registry.npmjs.org/picomatch/-/picomatch-4.0.3.tgz",
      "integrity": "sha512-5gTmgEY/sqK6gFXLIsQNH19lWb4ebPDLA4SdLP7dsWkIXHWlG66oPuVvXSGFPppYZz8ZDZq0dYYrbHfBCVUb1Q==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/sponsors/jonschlinkert"
      }
    },
    "node_modules/postcss": {
      "version": "8.5.6",
      "resolved": "https://registry.npmjs.org/postcss/-/postcss-8.5.6.tgz",
      "integrity": "sha512-3Ybi1tAuwAP9s0r1UQ2J4n5Y0G05bJkpUIO0/bI9MhwmD70S5aTWbXGBwxHrelT+XM1k6dM0pk+SwNkpTRN7Pg==",
      "dev": true,
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/postcss/"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/postcss"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "license": "MIT",
      "dependencies": {
        "nanoid": "^3.3.11",
        "picocolors": "^1.1.1",
        "source-map-js": "^1.2.1"
      },
      "engines": {
        "node": "^10 || ^12 || >=14"
      }
    },
    "node_modules/prelude-ls": {
      "version": "1.2.1",
      "resolved": "https://registry.npmjs.org/prelude-ls/-/prelude-ls-1.2.1.tgz",
      "integrity": "sha512-vkcDPrRZo1QZLbn5RLGPpg/WmIQ65qoWWhcGKf/b5eplkkarX0m9z8ppCat4mlOqUsWpyNuYgO3VRyrYHSzX5g==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">= 0.8.0"
      }
    },
    "node_modules/punycode": {
      "version": "2.3.1",
      "resolved": "https://registry.npmjs.org/punycode/-/punycode-2.3.1.tgz",
      "integrity": "sha512-vYt7UD1U9Wg6138shLtLOvdAu+8DsC/ilFtEVHcH+wydcSpNE20AfSOduf6MkRFahL5FY7X1oU7nKVZFtfq8Fg==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/react": {
      "version": "19.2.3",
      "resolved": "https://registry.npmjs.org/react/-/react-19.2.3.tgz",
      "integrity": "sha512-Ku/hhYbVjOQnXDZFv2+RibmLFGwFdeeKHFcOTlrt7xplBnya5OGn/hIRDsqDiSUcfORsDC7MPxwork8jBwsIWA==",
      "license": "MIT",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/react-dom": {
      "version": "19.2.3",
      "resolved": "https://registry.npmjs.org/react-dom/-/react-dom-19.2.3.tgz",
      "integrity": "sha512-yELu4WmLPw5Mr/lmeEpox5rw3RETacE++JgHqQzd2dg+YbJuat3jH4ingc+WPZhxaoFzdv9y33G+F7Nl5O0GBg==",
      "license": "MIT",
      "dependencies": {
        "scheduler": "^0.27.0"
      },
      "peerDependencies": {
        "react": "^19.2.3"
      }
    },
    "node_modules/react-refresh": {
      "version": "0.18.0",
      "resolved": "https://registry.npmjs.org/react-refresh/-/react-refresh-0.18.0.tgz",
      "integrity": "sha512-QgT5//D3jfjJb6Gsjxv0Slpj23ip+HtOpnNgnb2S5zU3CB26G/IDPGoy4RJB42wzFE46DRsstbW6tKHoKbhAxw==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/resolve-from": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/resolve-from/-/resolve-from-4.0.0.tgz",
      "integrity": "sha512-pb/MYmXstAkysRFx8piNI1tGFNQIFA3vkE3Gq4EuA1dF6gHp/+vgZqsCGJapvy8N3Q+4o7FwvquPJcnZ7RYy4g==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/rollup": {
      "version": "4.56.0",
      "resolved": "https://registry.npmjs.org/rollup/-/rollup-4.56.0.tgz",
      "integrity": "sha512-9FwVqlgUHzbXtDg9RCMgodF3Ua4Na6Gau+Sdt9vyCN4RhHfVKX2DCHy3BjMLTDd47ITDhYAnTwGulWTblJSDLg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@types/estree": "1.0.8"
      },
      "bin": {
        "rollup": "dist/bin/rollup"
      },
      "engines": {
        "node": ">=18.0.0",
        "npm": ">=8.0.0"
      },
      "optionalDependencies": {
        "@rollup/rollup-android-arm-eabi": "4.56.0",
        "@rollup/rollup-android-arm64": "4.56.0",
        "@rollup/rollup-darwin-arm64": "4.56.0",
        "@rollup/rollup-darwin-x64": "4.56.0",
        "@rollup/rollup-freebsd-arm64": "4.56.0",
        "@rollup/rollup-freebsd-x64": "4.56.0",
        "@rollup/rollup-linux-arm-gnueabihf": "4.56.0",
        "@rollup/rollup-linux-arm-musleabihf": "4.56.0",
        "@rollup/rollup-linux-arm64-gnu": "4.56.0",
        "@rollup/rollup-linux-arm64-musl": "4.56.0",
        "@rollup/rollup-linux-loong64-gnu": "4.56.0",
        "@rollup/rollup-linux-loong64-musl": "4.56.0",
        "@rollup/rollup-linux-ppc64-gnu": "4.56.0",
        "@rollup/rollup-linux-ppc64-musl": "4.56.0",
        "@rollup/rollup-linux-riscv64-gnu": "4.56.0",
        "@rollup/rollup-linux-riscv64-musl": "4.56.0",
        "@rollup/rollup-linux-s390x-gnu": "4.56.0",
        "@rollup/rollup-linux-x64-gnu": "4.56.0",
        "@rollup/rollup-linux-x64-musl": "4.56.0",
        "@rollup/rollup-openbsd-x64": "4.56.0",
        "@rollup/rollup-openharmony-arm64": "4.56.0",
        "@rollup/rollup-win32-arm64-msvc": "4.56.0",
        "@rollup/rollup-win32-ia32-msvc": "4.56.0",
        "@rollup/rollup-win32-x64-gnu": "4.56.0",
        "@rollup/rollup-win32-x64-msvc": "4.56.0",
        "fsevents": "~2.3.2"
      }
    },
    "node_modules/scheduler": {
      "version": "0.27.0",
      "resolved": "https://registry.npmjs.org/scheduler/-/scheduler-0.27.0.tgz",
      "integrity": "sha512-eNv+WrVbKu1f3vbYJT/xtiF5syA5HPIMtf9IgY/nKg0sWqzAUEvqY/xm7OcZc/qafLx/iO9FgOmeSAp4v5ti/Q==",
      "license": "MIT"
    },
    "node_modules/semver": {
      "version": "6.3.1",
      "resolved": "https://registry.npmjs.org/semver/-/semver-6.3.1.tgz",
      "integrity": "sha512-BR7VvDCVHO+q2xBEWskxS6DJE1qRnb7DxzUrogb71CWoSficBxYsiAGd+Kl0mmq/MprG9yArRkyrQxTO6XjMzA==",
      "dev": true,
      "license": "ISC",
      "bin": {
        "semver": "bin/semver.js"
      }
    },
    "node_modules/shebang-command": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/shebang-command/-/shebang-command-2.0.0.tgz",
      "integrity": "sha512-kHxr2zZpYtdmrN1qDjrrX/Z1rR1kG8Dx+gkpK1G4eXmvXswmcE1hTWBWYUzlraYw1/yZp6YuDY77YtvbN0dmDA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "shebang-regex": "^3.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/shebang-regex": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/shebang-regex/-/shebang-regex-3.0.0.tgz",
      "integrity": "sha512-7++dFhtcx3353uBaq8DDR4NuxBetBzC7ZQOhmTQInHEd6bSrXdiEyzCvG07Z44UYdLShWUyXt5M/yhz8ekcb1A==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/source-map-js": {
      "version": "1.2.1",
      "resolved": "https://registry.npmjs.org/source-map-js/-/source-map-js-1.2.1.tgz",
      "integrity": "sha512-UXWMKhLOwVKb728IUtQPXxfYU+usdybtUrK/8uGE8CQMvrhOpwvzDBwj0QhSL7MQc7vIsISBG8VQ8+IDQxpfQA==",
      "dev": true,
      "license": "BSD-3-Clause",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/strip-json-comments": {
      "version": "3.1.1",
      "resolved": "https://registry.npmjs.org/strip-json-comments/-/strip-json-comments-3.1.1.tgz",
      "integrity": "sha512-6fPc+R4ihwqP6N/aIv2f1gMH8lOVtWQHoqC4yK6oSDVVocumAsfCqjkXnqiYMhmMwS/mEHLp7Vehlt3ql6lEig==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/supports-color": {
      "version": "7.2.0",
      "resolved": "https://registry.npmjs.org/supports-color/-/supports-color-7.2.0.tgz",
      "integrity": "sha512-qpCAvRl9stuOHveKsn7HncJRvv501qIacKzQlO/+Lwxc9+0q2wLyv4Dfvt80/DPn2pqOBsJdDiogXGR9+OvwRw==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "has-flag": "^4.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/tinyglobby": {
      "version": "0.2.15",
      "resolved": "https://registry.npmjs.org/tinyglobby/-/tinyglobby-0.2.15.tgz",
      "integrity": "sha512-j2Zq4NyQYG5XMST4cbs02Ak8iJUdxRM0XI5QyxXuZOzKOINmWurp3smXu3y5wDcJrptwpSjgXHzIQxR0omXljQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "fdir": "^6.5.0",
        "picomatch": "^4.0.3"
      },
      "engines": {
        "node": ">=12.0.0"
      },
      "funding": {
        "url": "https://github.com/sponsors/SuperchupuDev"
      }
    },
    "node_modules/ts-api-utils": {
      "version": "2.4.0",
      "resolved": "https://registry.npmjs.org/ts-api-utils/-/ts-api-utils-2.4.0.tgz",
      "integrity": "sha512-3TaVTaAv2gTiMB35i3FiGJaRfwb3Pyn/j3m/bfAvGe8FB7CF6u+LMYqYlDh7reQf7UNvoTvdfAqHGmPGOSsPmA==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=18.12"
      },
      "peerDependencies": {
        "typescript": ">=4.8.4"
      }
    },
    "node_modules/type-check": {
      "version": "0.4.0",
      "resolved": "https://registry.npmjs.org/type-check/-/type-check-0.4.0.tgz",
      "integrity": "sha512-XleUoc9uwGXqjWwXaUTZAmzMcFZ5858QA2vvx1Ur5xIcixXIP+8LnFDgRplU30us6teqdlskFfu+ae4K79Ooew==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "prelude-ls": "^1.2.1"
      },
      "engines": {
        "node": ">= 0.8.0"
      }
    },
    "node_modules/typescript": {
      "version": "5.9.3",
      "resolved": "https://registry.npmjs.org/typescript/-/typescript-5.9.3.tgz",
      "integrity": "sha512-jl1vZzPDinLr9eUt3J/t7V6FgNEw9QjvBPdysz9KfQDD41fQrC2Y4vKQdiaUpFT4bXlb1RHhLpp8wtm6M5TgSw==",
      "dev": true,
      "license": "Apache-2.0",
      "bin": {
        "tsc": "bin/tsc",
        "tsserver": "bin/tsserver"
      },
      "engines": {
        "node": ">=14.17"
      }
    },
    "node_modules/typescript-eslint": {
      "version": "8.53.1",
      "resolved": "https://registry.npmjs.org/typescript-eslint/-/typescript-eslint-8.53.1.tgz",
      "integrity": "sha512-gB+EVQfP5RDElh9ittfXlhZJdjSU4jUSTyE2+ia8CYyNvet4ElfaLlAIqDvQV9JPknKx0jQH1racTYe/4LaLSg==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@typescript-eslint/eslint-plugin": "8.53.1",
        "@typescript-eslint/parser": "8.53.1",
        "@typescript-eslint/typescript-estree": "8.53.1",
        "@typescript-eslint/utils": "8.53.1"
      },
      "engines": {
        "node": "^18.18.0 || ^20.9.0 || >=21.1.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/typescript-eslint"
      },
      "peerDependencies": {
        "eslint": "^8.57.0 || ^9.0.0",
        "typescript": ">=4.8.4 <6.0.0"
      }
    },
    "node_modules/undici-types": {
      "version": "7.16.0",
      "resolved": "https://registry.npmjs.org/undici-types/-/undici-types-7.16.0.tgz",
      "integrity": "sha512-Zz+aZWSj8LE6zoxD+xrjh4VfkIG8Ya6LvYkZqtUQGJPZjYl53ypCaUwWqo7eI0x66KBGeRo+mlBEkMSeSZ38Nw==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/update-browserslist-db": {
      "version": "1.2.3",
      "resolved": "https://registry.npmjs.org/update-browserslist-db/-/update-browserslist-db-1.2.3.tgz",
      "integrity": "sha512-Js0m9cx+qOgDxo0eMiFGEueWztz+d4+M3rGlmKPT+T4IS/jP4ylw3Nwpu6cpTTP8R1MAC1kF4VbdLt3ARf209w==",
      "dev": true,
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/browserslist"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/browserslist"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "license": "MIT",
      "dependencies": {
        "escalade": "^3.2.0",
        "picocolors": "^1.1.1"
      },
      "bin": {
        "update-browserslist-db": "cli.js"
      },
      "peerDependencies": {
        "browserslist": ">= 4.21.0"
      }
    },
    "node_modules/uri-js": {
      "version": "4.4.1",
      "resolved": "https://registry.npmjs.org/uri-js/-/uri-js-4.4.1.tgz",
      "integrity": "sha512-7rKUyy33Q1yc98pQ1DAmLtwX109F7TIfWlW1Ydo8Wl1ii1SeHieeh0HHfPeL2fMXK6z0s8ecKs9frCuLJvndBg==",
      "dev": true,
      "license": "BSD-2-Clause",
      "dependencies": {
        "punycode": "^2.1.0"
      }
    },
    "node_modules/vite": {
      "version": "7.3.1",
      "resolved": "https://registry.npmjs.org/vite/-/vite-7.3.1.tgz",
      "integrity": "sha512-w+N7Hifpc3gRjZ63vYBXA56dvvRlNWRczTdmCBBa+CotUzAPf5b7YMdMR/8CQoeYE5LX3W4wj6RYTgonm1b9DA==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "esbuild": "^0.27.0",
        "fdir": "^6.5.0",
        "picomatch": "^4.0.3",
        "postcss": "^8.5.6",
        "rollup": "^4.43.0",
        "tinyglobby": "^0.2.15"
      },
      "bin": {
        "vite": "bin/vite.js"
      },
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      },
      "funding": {
        "url": "https://github.com/vitejs/vite?sponsor=1"
      },
      "optionalDependencies": {
        "fsevents": "~2.3.3"
      },
      "peerDependencies": {
        "@types/node": "^20.19.0 || >=22.12.0",
        "jiti": ">=1.21.0",
        "less": "^4.0.0",
        "lightningcss": "^1.21.0",
        "sass": "^1.70.0",
        "sass-embedded": "^1.70.0",
        "stylus": ">=0.54.8",
        "sugarss": "^5.0.0",
        "terser": "^5.16.0",
        "tsx": "^4.8.1",
        "yaml": "^2.4.2"
      },
      "peerDependenciesMeta": {
        "@types/node": {
          "optional": true
        },
        "jiti": {
          "optional": true
        },
        "less": {
          "optional": true
        },
        "lightningcss": {
          "optional": true
        },
        "sass": {
          "optional": true
        },
        "sass-embedded": {
          "optional": true
        },
        "stylus": {
          "optional": true
        },
        "sugarss": {
          "optional": true
        },
        "terser": {
          "optional": true
        },
        "tsx": {
          "optional": true
        },
        "yaml": {
          "optional": true
        }
      }
    },
    "node_modules/which": {
      "version": "2.0.2",
      "resolved": "https://registry.npmjs.org/which/-/which-2.0.2.tgz",
      "integrity": "sha512-BLI3Tl1TW3Pvl70l3yq3Y64i+awpwXqsGBYWkkqMtnbXgrMD+yj7rhW0kuEDxzJaYXGjEW5ogapKNMEKNMjibA==",
      "dev": true,
      "license": "ISC",
      "dependencies": {
        "isexe": "^2.0.0"
      },
      "bin": {
        "node-which": "bin/node-which"
      },
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/word-wrap": {
      "version": "1.2.5",
      "resolved": "https://registry.npmjs.org/word-wrap/-/word-wrap-1.2.5.tgz",
      "integrity": "sha512-BN22B5eaMMI9UMtjrGd5g5eCYPpCPDUy0FJXbYsaT5zYxjFOckS53SQDE3pWkVoWpHXVb3BrYcEN4Twa55B5cA==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/yallist": {
      "version": "3.1.1",
      "resolved": "https://registry.npmjs.org/yallist/-/yallist-3.1.1.tgz",
      "integrity": "sha512-a4UGQaWPH59mOXUYnAG2ewncQS4i4F43Tv3JoAM+s2VDAmS9NsK8GpDMLrCHPksFT7h3K6TOoUNn2pb7RoXx4g==",
      "dev": true,
      "license": "ISC"
    },
    "node_modules/yocto-queue": {
      "version": "0.1.0",
      "resolved": "https://registry.npmjs.org/yocto-queue/-/yocto-queue-0.1.0.tgz",
      "integrity": "sha512-rVksvsnNCdJ/ohGc6xgPwyN8eheCxsiLM8mxuE/t/mOVqJewPuO1miLpTHQiRgTKCLexL4MeAFVagts7HmNZ2Q==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/zod": {
      "version": "4.3.6",
      "resolved": "https://registry.npmjs.org/zod/-/zod-4.3.6.tgz",
      "integrity": "sha512-rftlrkhHZOcjDwkGlnUtZZkvaPHCsDATp4pGpuOOMDaTdDDXF91wuVDJoWoPsKX/3YPQ5fHuF3STjcYyKr+Qhg==",
      "dev": true,
      "license": "MIT",
      "funding": {
        "url": "https://github.com/sponsors/colinhacks"
      }
    },
    "node_modules/zod-validation-error": {
      "version": "4.0.2",
      "resolved": "https://registry.npmjs.org/zod-validation-error/-/zod-validation-error-4.0.2.tgz",
      "integrity": "sha512-Q6/nZLe6jxuU80qb/4uJ4t5v2VEZ44lzQjPDhYJNztRQ4wyWc6VF3D3Kb/fAuPetZQnhS3hnajCf9CsWesghLQ==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": ">=18.0.0"
      },
      "peerDependencies": {
        "zod": "^3.25.0 || ^4.0.0"
      }
    }
  }
}
```

## frontend\package.json

```json
{
  "name": "frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^19.2.0",
    "react-dom": "^19.2.0"
  },
  "devDependencies": {
    "@eslint/js": "^9.39.1",
    "@types/node": "^24.10.1",
    "@types/react": "^19.2.5",
    "@types/react-dom": "^19.2.3",
    "@vitejs/plugin-react": "^5.1.1",
    "eslint": "^9.39.1",
    "eslint-plugin-react-hooks": "^7.0.1",
    "eslint-plugin-react-refresh": "^0.4.24",
    "globals": "^16.5.0",
    "typescript": "~5.9.3",
    "typescript-eslint": "^8.46.4",
    "vite": "^7.2.4"
  }
}
```

## frontend\public\vite.svg

```svg
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="31.88" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 257"><defs><linearGradient id="IconifyId1813088fe1fbc01fb466" x1="-.828%" x2="57.636%" y1="7.652%" y2="78.411%"><stop offset="0%" stop-color="#41D1FF"></stop><stop offset="100%" stop-color="#BD34FE"></stop></linearGradient><linearGradient id="IconifyId1813088fe1fbc01fb467" x1="43.376%" x2="50.316%" y1="2.242%" y2="89.03%"><stop offset="0%" stop-color="#FFEA83"></stop><stop offset="8.333%" stop-color="#FFDD35"></stop><stop offset="100%" stop-color="#FFA800"></stop></linearGradient></defs><path fill="url(#IconifyId1813088fe1fbc01fb466)" d="M255.153 37.938L134.897 252.976c-2.483 4.44-8.862 4.466-11.382.048L.875 37.958c-2.746-4.814 1.371-10.646 6.827-9.67l120.385 21.517a6.537 6.537 0 0 0 2.322-.004l117.867-21.483c5.438-.991 9.574 4.796 6.877 9.62Z"></path><path fill="url(#IconifyId1813088fe1fbc01fb467)" d="M185.432.063L96.44 17.501a3.268 3.268 0 0 0-2.634 3.014l-5.474 92.456a3.268 3.268 0 0 0 3.997 3.378l24.777-5.718c2.318-.535 4.413 1.507 3.936 3.838l-7.361 36.047c-.495 2.426 1.782 4.5 4.151 3.78l15.304-4.649c2.372-.72 4.652 1.36 4.15 3.788l-11.698 56.621c-.732 3.542 3.979 5.473 5.943 2.437l1.313-2.028l72.516-144.72c1.215-2.423-.88-5.186-3.54-4.672l-25.505 4.922c-2.396.462-4.435-1.77-3.759-4.114l16.646-57.705c.677-2.35-1.37-4.583-3.769-4.113Z"></path></svg>
```

