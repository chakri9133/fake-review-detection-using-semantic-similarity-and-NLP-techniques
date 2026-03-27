import React, { useState } from 'react';
import './Methodology.css';

const Methodology: React.FC = () => {
  const [hoveredStep, setHoveredStep] = useState<string | null>(null);

  const pipelineSteps = [
    { id: 'dataset', label: 'Review\nDataset', icon: '📊', tooltip: 'Collect reviews from e-commerce platforms' },
    { id: 'preprocessing', label: 'Data\nPreprocessing', icon: '🧹', tooltip: 'Clean, normalize, tokenize text' },
    { id: 'features', label: 'Feature\nExtraction', icon: '🔍', tooltip: 'BoW, TF-IDF, SBERT Embeddings' },
    { id: 'training', label: 'Model\nTraining', icon: '⚙️', tooltip: '4 models trained independently' },
    { id: 'similarity', label: 'Semantic\nSimilarity', icon: '🔗', tooltip: 'Cosine similarity analysis' },
    { id: 'evaluation', label: 'Model\nEvaluation', icon: '📈', tooltip: 'Accuracy, Precision, Recall, F1' },
    { id: 'prediction', label: 'Fake/Genuine\nPrediction', icon: '✅', tooltip: 'Web app delivers prediction' }
  ];

  const features = [
    {
      icon: '🟦',
      title: 'Bag of Words',
      description: 'Word frequency-based representation'
    },
    {
      icon: '🟨',
      title: 'TF-IDF',
      description: 'Weighted word importance'
    },
    {
      icon: '🟩',
      title: 'SBERT Embeddings',
      description: 'Context-aware semantic vectors'
    }
  ];

  const models = [
    {
      name: 'BoW + NB',
      strength: 'Simple baseline',
      icon: '📦',
      color: '#FF6B6B'
    },
    {
      name: 'TF-IDF + NB',
      strength: 'Better word weighting',
      icon: '⚖️',
      color: '#4ECDC4'
    },
    {
      name: 'GRU',
      strength: 'Sequential learning',
      icon: '🧠',
      color: '#95E1D3'
    },
    {
      name: 'SBERT + LR',
      strength: 'Best semantic understanding',
      icon: '⭐',
      color: '#FFD93D',
      isFinal: true
    }
  ];

  const metrics = [
    { icon: '📊', label: 'Accuracy', key: 'accuracy' },
    { icon: '🎯', label: 'Precision', key: 'precision' },
    { icon: '🔁', label: 'Recall', key: 'recall' },
    { icon: '⚖️', label: 'F1-Score', key: 'f1' },
    { icon: '🧩', label: 'Confusion Matrix', key: 'matrix' }
  ];

  return (
    <section className="methodology-visual" id="methodology">
      <div className="container">
        <h2>Methodology</h2>
        <p className="subtitle">End-to-End Fake Review Detection Pipeline</p>

        {/* Main Pipeline */}
        <div className="pipeline-section">
          <h3>System Pipeline</h3>
          <div className="pipeline">
            {pipelineSteps.map((step, index) => (
              <React.Fragment key={step.id}>
                <div
                  className={`pipeline-step ${hoveredStep === step.id ? 'active' : ''}`}
                  onMouseEnter={() => setHoveredStep(step.id)}
                  onMouseLeave={() => setHoveredStep(null)}
                  title={step.tooltip}
                >
                  <div className="step-icon">{step.icon}</div>
                  <div className="step-label">{step.label}</div>
                  {hoveredStep === step.id && (
                    <div className="tooltip">{step.tooltip}</div>
                  )}
                </div>
                {index < pipelineSteps.length - 1 && (
                  <div className="pipeline-arrow">→</div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Feature Extraction */}
        <div className="features-section">
          <h3>Feature Extraction Techniques</h3>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="feature-icon">{feature.icon}</div>
                <h4>{feature.title}</h4>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Model Comparison */}
        <div className="models-section">
          <h3>Model Comparison</h3>
          <div className="models-grid">
            {models.map((model, index) => (
              <div key={index} className={`model-card ${model.isFinal ? 'final-model' : ''}`}
                   style={{ borderTopColor: model.color }}>
                <div className="model-icon">{model.icon}</div>
                <h4>{model.name}</h4>
                <p className="model-strength">{model.strength}</p>
                {model.isFinal && <div className="final-badge">FINAL MODEL</div>}
              </div>
            ))}
          </div>
        </div>

        {/* Semantic Similarity */}
        <div className="similarity-section">
          <h3>Semantic Similarity Analysis</h3>
          <div className="similarity-diagram">
            <div className="review-box">
              <p>Review A</p>
            </div>
            <div className="similarity-connection">
              <div className="similarity-line"></div>
              <div className="similarity-label">High Cosine Similarity</div>
            </div>
            <div className="review-box">
              <p>Review B</p>
            </div>
            <div className="similarity-insight">→ Possible Fake Review Detected</div>
          </div>
          <p className="similarity-caption">High semantic similarity indicates duplicated or manipulated reviews</p>
        </div>

        {/* Evaluation Metrics */}
        <div className="metrics-section">
          <h3>Model Evaluation Metrics</h3>
          <div className="metrics-grid">
            {metrics.map((metric, index) => (
              <div key={index} className="metric-badge">
                <div className="metric-icon">{metric.icon}</div>
                <p>{metric.label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Deployment */}
        <div className="deployment-section">
          <h3>Web Application Deployment</h3>
          <div className="deployment-flow">
            <div className="deployment-box">
              <p>User Input</p>
              <p className="small">Review Text</p>
            </div>
            <div className="deployment-arrow">↓</div>
            <div className="deployment-box highlight">
              <p>SBERT + Logistic Regression</p>
              <p className="small">Final Model</p>
            </div>
            <div className="deployment-arrow">↓</div>
            <div className="deployment-box result">
              <p>🎯 Prediction Output</p>
              <p className="small">Fake / Genuine</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Methodology;
