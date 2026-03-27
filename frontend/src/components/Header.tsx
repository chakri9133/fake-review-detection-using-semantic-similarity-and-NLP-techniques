import React from 'react';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="container">
        <h1 className="title">Detection of Fake Reviews Using Semantic Similarity and Natural Language Processing Techniques</h1>
        <div className="problem-statement" id="problem-statement">
          <h2>Problem Statement</h2>
          <p>
            Online reviews play a crucial role in shaping consumer decisions across ecommerce, travel, food delivery, and service platforms. However, the growing presence of fake reviews that mislead potential customers poses a serious challenge to the credibility of these platforms. Traditional review moderation systems often rely on manual checks or simple keyword filtering, which are ineffective in identifying fake reviews.
          </p>
          <p>
            There is a need for an intelligent system that can automatically detect fake reviews by analyzing their semantic content and writing patterns. This project aims to develop a solution that uses Natural Language Processing (NLP) techniques such as semantic similarity analysis to evaluate the authenticity of reviews. By comparing reviews across products and users and identifying unnatural repetition, emotional exaggeration, or linguistic anomalies, the system can detect potentially fake reviews. This will enhance the trustworthiness and reliability of online review platforms.
          </p>
        </div>
        <div className="objectives" id="objectives">
          <h2>Objectives</h2>
          <ul>
            <li>To design and develop an intelligent system that automatically detects fake reviews using Natural Language Processing techniques.</li>
            <li>To apply semantic similarity analysis for identifying repetitive, exaggerated, or semantically inconsistent reviews across products and users.</li>
            <li>To extract linguistic and sentiment features (e.g., emotional tone, exaggeration, unnatural patterns) that help distinguish between genuine and deceptive reviews.</li>
            <li>Using only the S-BERT and Logistic regression (for classification).</li>
            <li>To improve detection accuracy and reduce false positives by integrating semantic similarity with advanced NLP-based feature extraction.</li>
          </ul>
        </div>
      </div>
    </header>
  );
};

export default Header;