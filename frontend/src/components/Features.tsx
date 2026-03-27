import React from 'react';
import './Features.css';

const Features: React.FC = () => {
  return (
    <section className="features">
      <div className="container">
        <h2>Key Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <h3>Machine Learning Model</h3>
            <p>Trained classifier using SBERT embeddings for accurate fake review detection.</p>
          </div>
          <div className="feature-card">
            <h3>Semantic Analysis</h3>
            <p>Advanced semantic similarity analysis to identify review patterns and anomalies.</p>
          </div>
          <div className="feature-card">
            <h3>Data Preprocessing</h3>
            <p>Comprehensive preprocessing pipeline for cleaning and preparing review data.</p>
          </div>
          <div className="feature-card">
            <h3>High Accuracy</h3>
            <p>Achieves high precision and recall in detecting deceptive reviews.</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;