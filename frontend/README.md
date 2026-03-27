# Fake Review Detection - Frontend

This is the frontend application for the Fake Review Detection project, built with React and TypeScript using Vite.

## Project Overview

This college engineering project focuses on detecting fake reviews in e-commerce platforms using machine learning techniques. The frontend provides an interface to showcase the project details, features, and includes a demo section for review analysis.

## Features

- **Project Showcase**: Displays comprehensive information about the fake review detection system
- **Interactive Demo**: Allows users to input review text for analysis (frontend-only simulation)
- **Responsive Design**: Clean, professional design suitable for academic presentations
- **Technology Stack Display**: Showcases the technologies used in the project

## Technologies Used

- **Frontend Framework**: React with TypeScript
- **Build Tool**: Vite
- **Styling**: CSS3 with modern design principles
- **Development**: ESLint for code quality

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Project Structure

```
src/
├── components/
│   ├── Header.tsx/css
│   ├── About.tsx/css
│   ├── Features.tsx/css
│   ├── Technologies.tsx/css
│   ├── Demo.tsx/css
│   ├── Team.tsx/css
│   └── Footer.tsx/css
├── App.tsx
├── App.css
├── index.css
└── main.tsx
```

## Academic Context

This frontend is part of a comprehensive computer science engineering project that includes:

- Machine learning model development (Python)
- Data preprocessing and analysis
- Model training and evaluation
- Web application development

## Future Enhancements

- Backend API integration for real-time review analysis
- User authentication and review history
- Advanced visualization of model predictions
- Mobile application development

## License

This project is developed as part of an academic assignment.
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
