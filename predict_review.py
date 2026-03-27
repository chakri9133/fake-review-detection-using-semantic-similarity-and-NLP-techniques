import re
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
from config import MODEL_PATH, SBERT_MODEL_NAME, THRESHOLD_FAKE, THRESHOLD_GENUINE

# =====================================================
# 2. TEXT CLEANING (SAME AS TRAINING)
# =====================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"&quot;|&#34;", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# 3. LOAD MODELS
# =====================================================
print("Loading SBERT model...")
embedder = SentenceTransformer(SBERT_MODEL_NAME)

print("Loading classifier...")
clf = joblib.load(MODEL_PATH)

# =====================================================
# 4. DECISION POLICY
# =====================================================
def decision_policy(p_fake):
    if p_fake < THRESHOLD_GENUINE:
        return "LIKELY GENUINE"
    elif p_fake > THRESHOLD_FAKE:
        return "LIKELY FAKE"
    else:
        return "UNCERTAIN"

# =====================================================
# 5. MAIN PREDICTION FUNCTION
# =====================================================
def predict_review(review_text):
    """
    Input  : Raw review text (string)
    Output : Dictionary with label + probability
    """

    # Clean text
    cleaned = clean_text(review_text)

    # Embed
    embedding = embedder.encode(
        cleaned,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).reshape(1, -1)

    # Predict probability
    p_fake = clf.predict_proba(embedding)[0, 1]

    # Decision
    label = decision_policy(p_fake)

    return {
    "review": review_text,
    "prediction": label,
    "fake_probability": round(float(p_fake), 4),
    "confidence_percent": round(float(p_fake) * 100, 2)
}


# =====================================================
# 6. CLI INTERFACE (FOR TESTING)
# =====================================================
if __name__ == "__main__":
    print("\nEnter a review (type 'exit' to quit):\n")

    while True:
        user_review = input("> ")

        if user_review.lower() == "exit":
            break

        result = predict_review(user_review)

        print("\n--- RESULT ---")
        print("Prediction      :", result["prediction"])
        print("Fake Probability:", result["fake_probability"])
        print("----------------\n")
