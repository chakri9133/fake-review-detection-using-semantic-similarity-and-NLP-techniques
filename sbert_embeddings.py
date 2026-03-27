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
print("✅ SBERT EMBEDDINGS GENERATED & SAVED")
print("📁 Embeddings:", EMB_PATH)
print("📁 Labels:", LABEL_PATH)
print("==============================================")
