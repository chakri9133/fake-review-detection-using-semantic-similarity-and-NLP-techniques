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
# 1 → unreliable-like
# 0 → reliable-like
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
print("✅ BEAUTY TRAIN PREPROCESSING COMPLETED")
print("📁 Output saved at:")
print(OUTPUT_PATH)
print("==============================================")
