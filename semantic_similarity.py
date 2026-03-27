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

print(f"Average Fake–Fake similarity     : {fake_fake_sim.mean():.4f}")
print(f"Average Genuine–Genuine similarity: {genuine_genuine_sim.mean():.4f}")
print(f"Average Fake–Genuine similarity   : {fake_genuine_sim.mean():.4f}")
