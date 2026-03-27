import Navigation from './components/Navigation';
import Header from './components/Header';
import Methodology from './components/Methodology';
import Features from './components/Features';
import Technologies from './components/Technologies';
import Results from './components/Results';
import Demo from './components/Demo';
import Team from './components/Team';
import Footer from './components/Footer';
import './App.css';

function App() {
  return (
    <div className="App">
      <Navigation />
      <div id="header">
        <Header />
      </div>
      <div id="methodology">
        <Methodology />
        <Features />
      </div>
      <div id="technologies">
        <Technologies />
      </div>
      <div id="results">
        <Results />
      </div>
      <div id="demo">
        <Demo />
      </div>
      <Team />
      <Footer />
    </div>
  );
}

export default App;
