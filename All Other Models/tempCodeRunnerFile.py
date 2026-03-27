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