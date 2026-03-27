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
