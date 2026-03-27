import React, { useState } from 'react';
import './Demo.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000';

interface PredictionResult {
  review: string;
  prediction: string;
  fake_probability: number;
  confidence_percent: number;
}

const Demo: React.FC = () => {
  const [review, setReview] = useState('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to predict');
      }
      
      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="demo" id="demo">
      <div className="container">
        <h2>Try the Demo</h2>
        <p>Enter a review text below to check if it's likely to be fake or genuine.</p>
        <form onSubmit={handleSubmit} className="demo-form">
          <textarea
            value={review}
            onChange={(e) => setReview(e.target.value)}
            placeholder="Enter your review text here..."
            rows={5}
            required
            disabled={loading}
          />
          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze Review'}
          </button>
        </form>
        {loading && <div className="loading">Analyzing review...</div>}
        {error && <div className="error">Error: {error}</div>}
        {result && (
          <div className="result">
            <h3>Prediction Result</h3>
            <div className="prediction-card">
              <div className={`prediction-status ${result.prediction.toLowerCase().replace(' ', '-')}`}>
                <span className="status-emoji">
                  {result.prediction.toLowerCase() === 'likely genuine' && '😊'}
                  {result.prediction.toLowerCase() === 'likely fake' && '😠'}
                  {result.prediction.toLowerCase() === 'uncertain' && '🤔'}
                </span>
                <strong>{result.prediction}</strong>
              </div>
              <div className="prediction-details">
                <p><strong>Fake Probability:</strong> {result.fake_probability.toFixed(4)} ({result.confidence_percent.toFixed(2)}%)</p>
                <p><strong>Confidence:</strong> {result.confidence_percent > 50 ? 'High' : result.confidence_percent > 30 ? 'Medium' : 'Low'}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default Demo;
