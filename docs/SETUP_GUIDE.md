# Fake Review Detection - Setup Guide

This project connects a React/TypeScript frontend with a Python Flask backend using SBERT embeddings for fake review detection.

## Project Structure

- **Backend**: Flask application with SBERT model and Logistic Regression classifier
- **Frontend**: React/TypeScript app built with Vite
- **Models**: Pre-trained SBERT embeddings + Logistic Regression classifier

## Setup Instructions

### 1. Backend Setup

#### Install Python Dependencies
```bash
cd "D:\Fake Review Detection"
pip install -r requirements.txt
```

#### Run Flask Backend
```bash
python app.py
```

The backend will start on `http://localhost:5000`

**Key Endpoints:**
- `POST /api/predict` - Send review text for prediction (JSON)
  - Request: `{"review": "review text here"}`
  - Response: `{"review": "...", "prediction": "...", "fake_probability": 0.X, "confidence_percent": X}`
- `GET /api/metrics` - Get model metrics

### 2. Frontend Setup

#### Install Node Dependencies
```bash
cd "D:\Fake Review Detection\frontend"
npm install
```

#### Run Development Server
```bash
npm run dev
```

The frontend will start on `http://localhost:5173` (or another port if 5173 is busy)

#### Build for Production
```bash
npm run build
```

## How It Works

1. **User Interface**: The Demo section in the frontend allows users to input a review
2. **API Call**: Frontend sends the review to `http://localhost:5000/api/predict`
3. **Processing**: 
   - Text is cleaned and preprocessed
   - SBERT model generates embeddings
   - Logistic Regression classifier predicts fake probability
4. **Response**: Result shows prediction (LIKELY GENUINE, LIKELY FAKE, or UNCERTAIN) with confidence score
5. **Display**: Frontend displays the prediction with visual styling

## API Response Format

```json
{
  "review": "Original review text",
  "prediction": "LIKELY GENUINE",
  "fake_probability": 0.2345,
  "confidence_percent": 23.45
}
```

## Prediction Labels

- **LIKELY GENUINE**: Probability < 30%
- **UNCERTAIN**: Probability between 30% - 75%
- **LIKELY FAKE**: Probability > 75%

## Development

### Running Both Simultaneously

**Terminal 1 - Backend:**
```bash
cd "D:\Fake Review Detection"
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd "D:\Fake Review Detection\frontend"
npm run dev
```

Then open your browser to the frontend URL (usually http://localhost:5173)

## Troubleshooting

### CORS Errors
- Ensure Flask backend is running on port 5000
- Check that CORS is enabled in app.py
- Frontend must make requests to `http://localhost:5000/api/predict`

### Backend Not Responding
- Verify Flask is running: `python app.py` should show "Running on http://127.0.0.1:5000"
- Check for errors in the Flask terminal

### Models Not Found
- Verify paths in `predict_review.py` are correct
- Ensure `Models/logistic_model.joblib` and `Models/model_metrics.json` exist

### SBERT Model Download
- On first run, SBERT model will be downloaded (~500MB)
- Requires internet connection
- May take several minutes

## Technologies Used

- **Backend**: Python, Flask, Flask-CORS, Sentence-BERT
- **Frontend**: React, TypeScript, Vite
- **ML**: scikit-learn, NumPy
- **API**: JSON over HTTP

## File Changes Made

1. **app.py** - Added `/api/predict` and `/api/metrics` JSON endpoints with CORS support
2. **predict_review.py** - No changes needed (already compatible)
3. **Demo.tsx** - Updated to make actual API calls to the backend
4. **Demo.css** - Enhanced styling for result display
5. **requirements.txt** - Created with all Python dependencies

## Notes

- Backend runs on port 5000
- Frontend runs on port 5173 (configurable in vite.config.ts)
- SBERT embeddings are cached after first use
- Model predictions are deterministic (same input = same output)
