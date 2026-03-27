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
