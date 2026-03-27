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
    print("✅ Amazon Kindle CSV found:", csv_file)
    df = pd.read_csv(csv_file)

elif json_file:
    print("⚠️ CSV not found, loading Amazon Kindle JSON:", json_file)
    data = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

else:
    raise FileNotFoundError(
        "❌ No Amazon Kindle review dataset found inside Datasets"
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
print("✅ PREPROCESSING COMPLETED SUCCESSFULLY")
print("📁 Output saved at:")
print(OUTPUT_PATH)
print("==============================================")
