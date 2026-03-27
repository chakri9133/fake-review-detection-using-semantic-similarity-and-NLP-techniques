import React from 'react';
import './Team.css';

const Team: React.FC = () => {
  const teamMembers = [
    { name: 'Chakri K', role: 'Team Leader' },
    { name: 'Delan Pinto', role: 'Project Team Member' },
    { name: 'Hruday Kumar', role: 'Project Team Member' },
    { name: 'Mohammed Mukram', role: 'Project Team Member' }
  ];

  return (
    <section className="team">
      <div className="container">
        <h2>Project Team</h2>
        
        <div className="guide-section">
          <h3>Under the Guidance of</h3>
          <div className="guide-card">
            <p className="guide-name">Dr Sangeetha R</p>
            <p className="guide-title">Faculty Advisor</p>
          </div>
        </div>

        <div className="team-members">
          {teamMembers.map((member, index) => (
            <div className="member" key={index}>
              <h3>{member.name}</h3>
              <p>{member.role}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Team;