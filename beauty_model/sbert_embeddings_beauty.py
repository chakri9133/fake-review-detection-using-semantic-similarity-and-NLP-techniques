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
print("✅ BEAUTY SBERT EMBEDDINGS GENERATED & SAVED")
print("📁 Embeddings:", EMB_PATH)
print("📁 Labels:", LABEL_PATH)
print("==============================================")
