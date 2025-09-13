import React, { useState, useEffect } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import axios from "axios";

// Import components
import ImageClassifier from "./components/ImageClassifier";
import DatasetManager from "./components/DatasetManager";
import ModelTrainer from "./components/ModelTrainer";
import ModelManager from "./components/ModelManager";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Navigation Component
const Navigation = () => {
  return (
    <nav className="bg-green-600 text-white p-4 shadow-lg">
      <div className="container mx-auto flex justify-between items-center">
        <Link to="/" className="text-2xl font-bold flex items-center">
          🍉 Watermelon Classifier
        </Link>
        <div className="flex space-x-6">
          <Link to="/" className="hover:text-green-200 transition-colors">
            Classify
          </Link>
          <Link to="/datasets" className="hover:text-green-200 transition-colors">
            Datasets
          </Link>
          <Link to="/train" className="hover:text-green-200 transition-colors">
            Train Model
          </Link>
          <Link to="/models" className="hover:text-green-200 transition-colors">
            Models
          </Link>
        </div>
      </div>
    </nav>
  );
};

// Home Component - Image Classification
const Home = () => {
  const [apiStatus, setApiStatus] = useState(null);

  const checkApiStatus = async () => {
    try {
      const response = await axios.get(`${API}/health`);
      setApiStatus(response.data);
    } catch (e) {
      console.error("API health check failed:", e);
      setApiStatus({ status: "unhealthy", error: e.message });
    }
  };

  useEffect(() => {
    checkApiStatus();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-green-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-green-800 mb-4">
            AI-Powered Watermelon Classifier
          </h1>
          <p className="text-lg text-green-600 mb-6">
            Detect Crimsonsweet F1 variety and ripeness with advanced machine learning
          </p>
          
          {/* API Status */}
          {apiStatus && (
            <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${
              apiStatus.status === 'healthy' 
                ? 'bg-green-100 text-green-800' 
                : 'bg-red-100 text-red-800'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                apiStatus.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              {apiStatus.status === 'healthy' ? 'API Ready' : 'API Error'}
              {apiStatus.model_loaded && (
                <span className="ml-2">• Model Loaded</span>
              )}
            </div>
          )}
        </div>

        {/* Main Classification Interface */}
        <ImageClassifier />
        
        {/* Features Section */}
        <div className="mt-16">
          <h2 className="text-2xl font-bold text-green-800 text-center mb-8">
            Features
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white rounded-lg p-6 shadow-md">
              <div className="text-3xl mb-4">🎯</div>
              <h3 className="text-xl font-semibold mb-2">Variety Detection</h3>
              <p className="text-gray-600">
                Specifically trained to identify Crimsonsweet F1 variety watermelons
              </p>
            </div>
            
            <div className="bg-white rounded-lg p-6 shadow-md">
              <div className="text-3xl mb-4">📊</div>
              <h3 className="text-xl font-semibold mb-2">Ripeness Analysis</h3>
              <p className="text-gray-600">
                Accurately determine if your watermelon is ripe or unripe
              </p>
            </div>
            
            <div className="bg-white rounded-lg p-6 shadow-md">
              <div className="text-3xl mb-4">🔬</div>
              <h3 className="text-xl font-semibold mb-2">Confidence Scoring</h3>
              <p className="text-gray-600">
                Get detailed confidence scores for reliable classification
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Navigation />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/datasets" element={<DatasetManager />} />
          <Route path="/train" element={<ModelTrainer />} />
          <Route path="/models" element={<ModelManager />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;