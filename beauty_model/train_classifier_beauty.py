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

print("✅ Beauty model saved at:", MODEL_PATH)

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

print("✅ Beauty model metrics saved at:", METRICS_PATH)
