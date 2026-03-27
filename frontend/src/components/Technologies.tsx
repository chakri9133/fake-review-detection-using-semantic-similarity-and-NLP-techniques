import React from 'react';
import './Technologies.css';

const Technologies: React.FC = () => {
  const technologies = [
    {
      category: 'Machine Learning & NLP',
      items: [
        'Python 3.x',
        'Sentence Transformers (SBERT)',
        'Scikit-learn',
        'NumPy',
        'Pandas',
        'Joblib (Model Serialization)'
      ]
    },
    {
      category: 'Backend',
      items: [
        'Flask',
        'Flask-CORS',
        'JSON API',
        'Python Virtual Environment'
      ]
    },
    {
      category: 'Frontend',
      items: [
        'React 19.2.0',
        'TypeScript 5.9.3',
        'Vite 7.2.4',
        'CSS3',
        'Responsive Design'
      ]
    },
    {
      category: 'Data & Datasets',
      items: [
        'Amazon Kindle Reviews',
        'CSV Processing',
        'NumPy Arrays (.npy)'
      ]
    }
  ];

  return (
    <section className="technologies" id="technologies">
      <div className="container">
        <h2>Technologies Used</h2>
        <div className="tech-list">
          {technologies.map((tech, index) => (
            <div className="tech-category" key={index}>
              <h3>{tech.category}</h3>
              <ul>
                {tech.items.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Technologies;
