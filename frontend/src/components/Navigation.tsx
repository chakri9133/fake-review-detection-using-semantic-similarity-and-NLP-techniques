import React, { useState } from 'react';
import './Navigation.css';

const Navigation: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setIsOpen(false);
    }
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <span className="brand-text">Fake Review Detection</span>
        </div>

        <button
          className={`hamburger ${isOpen ? 'active' : ''}`}
          onClick={() => setIsOpen(!isOpen)}
        >
          <span></span>
          <span></span>
          <span></span>
        </button>

        <ul className={`nav-menu ${isOpen ? 'active' : ''}`}>
          <li className="nav-item">
            <button onClick={() => scrollToSection('problem-statement')} className="nav-link">
              Problem Statement
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('objectives')} className="nav-link">
              Objectives
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('methodology')} className="nav-link">
              Methodology
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('features')} className="nav-link">
              Key Features
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('technologies')} className="nav-link">
              Technologies Used
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('results')} className="nav-link">
              Results & Evaluation
            </button>
          </li>
          <li className="nav-item">
            <button onClick={() => scrollToSection('demo')} className="nav-link">
              Demo
            </button>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navigation;
