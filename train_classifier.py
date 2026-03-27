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

print("✅ Trained model saved at:", MODEL_PATH)

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

print("✅ Model metrics updated and saved at:", METRICS_PATH)
