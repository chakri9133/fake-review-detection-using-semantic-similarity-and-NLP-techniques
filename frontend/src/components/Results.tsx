import React from 'react';
import './Results.css';

const Results: React.FC = () => {
  return (
    <section className="results-section" id="results">
      <div className="container">
        <div className="results">
          <h2>Results & Evaluation</h2>
          <table className="results-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>BoW + Naive Bayes</td>
                <td>0.86</td>
                <td>0.85</td>
                <td>0.89</td>
                <td>0.87</td>
              </tr>
              <tr>
                <td>TF-IDF + Naive Bayes</td>
                <td>0.88</td>
                <td>0.96</td>
                <td>0.28</td>
                <td>0.44</td>
              </tr>
              <tr>
                <td>GRU (from-scratch embeddings)</td>
                <td>0.85</td>
                <td>0.8218</td>
                <td>0.8029</td>
                <td>0.8323</td>
              </tr>
              <tr className="highlight">
                <td><strong>SBERT + Logistic Regression</strong></td>
                <td><strong>0.89</strong></td>
                <td><strong>0.88</strong></td>
                <td><strong>0.90</strong></td>
                <td><strong>0.89</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};

export default Results;
