import React from 'react';
import './About.css';

const About: React.FC = () => {
  return (
    <section className="about">
      <div className="container">
        <h2>About the Project</h2>
        <p>
          This project aims to develop a robust system for detecting fake reviews in e-commerce platforms,
          particularly focusing on Amazon Kindle Store reviews. By leveraging machine learning techniques
          and natural language processing, we analyze review patterns to distinguish between genuine and
          deceptive feedback.
        </p>
        <p>
          The system uses SBERT embeddings for semantic similarity analysis and a trained classifier
          to predict the authenticity of reviews with high accuracy.
        </p>
      </div>
    </section>
  );
};

export default About;