"""
Flask Backend API for Fake Review Detection
Connects React Frontend to SBERT Model + Logistic Regression Classifier
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_review import predict_review
import json
from config import (
    BACKEND_DEBUG,
    BACKEND_HOST,
    BACKEND_PORT,
    CORS_ORIGINS,
    MODEL_METRICS_PATH,
    TEXT_MAX_LENGTH,
    TEXT_MIN_LENGTH,
)

app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, resources={
    r"/api/*": {
        "origins": CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load model metrics
if MODEL_METRICS_PATH.exists():
    with MODEL_METRICS_PATH.open("r", encoding="utf-8") as f:
        model_metrics = json.load(f)
else:
    model_metrics = {"status": "Model metrics not found"}


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API Endpoint: Predict if a review is fake or genuine
    
    Request:
    {
        "review": "Review text here"
    }
    
    Response:
    {
        "review": "Review text here",
        "prediction": "LIKELY GENUINE|LIKELY FAKE|UNCERTAIN",
        "fake_probability": 0.XX,
        "confidence_percent": XX.XX
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        review_text = data.get("review", "").strip()
        
        if not review_text:
            return jsonify({"error": "Review text is required"}), 400
        
        if len(review_text) < TEXT_MIN_LENGTH:
            return jsonify({"error": f"Review must be at least {TEXT_MIN_LENGTH} characters long"}), 400

        if len(review_text) > TEXT_MAX_LENGTH:
            return jsonify({"error": f"Review must be at most {TEXT_MAX_LENGTH} characters long"}), 400
        
        # Get prediction from SBERT model
        result = predict_review(review_text)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """API Endpoint: Get model metrics and statistics"""
    try:
        return jsonify(model_metrics), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "backend": "Flask",
        "model": "SBERT + Logistic Regression",
        "api_version": "1.0"
    }), 200


@app.route("/", methods=["GET"])
def index():
    """Root endpoint"""
    return jsonify({
        "message": "Fake Review Detection API",
        "endpoints": {
            "POST /api/predict": "Analyze a review",
            "GET /api/metrics": "Get model metrics",
            "GET /health": "Health check"
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Fake Review Detection - Backend API")
    print("=" * 60)
    print("SBERT Model: all-MiniLM-L6-v2")
    print("Classifier: Logistic Regression")
    print("Frontend origins:", ", ".join(CORS_ORIGINS))
    print(f"Backend: Flask ({BACKEND_HOST}:{BACKEND_PORT})")
    print("=" * 60)
    print("Starting server...")
    print("=" * 60)
    
    app.run(
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        debug=BACKEND_DEBUG,
        use_reloader=True
    )
