import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ModelTrainer = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [modelName, setModelName] = useState('');
  const [epochs, setEpochs] = useState(15);
  const [loading, setLoading] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Load datasets and training status on component mount
  useEffect(() => {
    loadDatasets();
    checkTrainingStatus();
    
    // Poll training status every 2 seconds when training is active
    const interval = setInterval(() => {
      if (trainingStatus?.is_training) {
        checkTrainingStatus();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [trainingStatus?.is_training]);

  const loadDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
    } catch (err) {
      console.error('Error loading datasets:', err);
      setError('Failed to load datasets');
    }
  };

  const checkTrainingStatus = async () => {
    try {
      const response = await axios.get(`${API}/train/status`);
      setTrainingStatus(response.data);
    } catch (err) {
      console.error('Error checking training status:', err);
    }
  };

  const startTraining = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset');
      return;
    }

    if (!modelName.trim()) {
      setError('Please enter a model name');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await axios.post(`${API}/train`, {
        dataset_name: selectedDataset,
        epochs: epochs,
        model_name: modelName.trim()
      });

      setSuccess('Training started successfully!');
      checkTrainingStatus();
    } catch (err) {
      console.error('Training start error:', err);
      setError(err.response?.data?.detail || 'Failed to start training');
    } finally {
      setLoading(false);
    }
  };

  const getSelectedDatasetInfo = () => {
    return datasets.find(d => d.name === selectedDataset);
  };

  const getProgressColor = (progress) => {
    if (progress < 30) return 'bg-red-500';
    if (progress < 70) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const generateModelName = () => {
    if (selectedDataset) {
      const timestamp = new Date().toISOString().slice(0, 16).replace(/[:-]/g, '');
      setModelName(`${selectedDataset}_model_${timestamp}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-8">Model Training</h1>

          {/* Training Status Card */}
          {trainingStatus && (
            <div className={`mb-8 p-6 rounded-lg shadow-md ${
              trainingStatus.is_training ? 'bg-blue-50 border border-blue-200' : 
              trainingStatus.message.includes('completed') ? 'bg-green-50 border border-green-200' :
              trainingStatus.message.includes('failed') ? 'bg-red-50 border border-red-200' :
              'bg-gray-50 border border-gray-200'
            }`}>
              <h2 className="text-xl font-semibold mb-4">
                {trainingStatus.is_training ? '🔄 Training in Progress' : '📊 Training Status'}
              </h2>
              
              {trainingStatus.is_training && (
                <div className="mb-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Progress</span>
                    <span className="text-sm">{trainingStatus.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all duration-300 ${getProgressColor(trainingStatus.progress)}`}
                      style={{ width: `${trainingStatus.progress}%` }}
                    ></div>
                  </div>
                </div>
              )}

              <p className={`font-medium ${
                trainingStatus.is_training ? 'text-blue-800' :
                trainingStatus.message.includes('completed') ? 'text-green-800' :
                trainingStatus.message.includes('failed') ? 'text-red-800' :
                'text-gray-800'
              }`}>
                {trainingStatus.message}
              </p>

              {trainingStatus.is_training && (
                <div className="mt-4 flex items-center text-sm text-blue-600">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                  Training is running in the background. You can leave this page and come back later.
                </div>
              )}
            </div>
          )}

          {/* Training Configuration */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-6">Configure Training</h2>

            {/* Dataset Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Dataset
              </label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                disabled={trainingStatus?.is_training}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
              >
                <option value="">Choose a dataset...</option>
                {datasets.map((dataset) => (
                  <option key={dataset.name} value={dataset.name}>
                    {dataset.name} ({dataset.train_samples + dataset.val_samples + dataset.test_samples} images)
                  </option>
                ))}
              </select>
            </div>

            {/* Dataset Info */}
            {selectedDataset && getSelectedDatasetInfo() && (
              <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                <h3 className="font-semibold text-blue-800 mb-2">Dataset Information</h3>
                <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-700">
                  <div>
                    <span className="font-medium">Classes:</span> {getSelectedDatasetInfo().classes.join(', ')}
                  </div>
                  <div>
                    <span className="font-medium">Training samples:</span> {getSelectedDatasetInfo().train_samples}
                  </div>
                  <div>
                    <span className="font-medium">Validation samples:</span> {getSelectedDatasetInfo().val_samples}
                  </div>
                  <div>
                    <span className="font-medium">Test samples:</span> {getSelectedDatasetInfo().test_samples}
                  </div>
                </div>
              </div>
            )}

            {/* Model Name */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Name
              </label>
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="Enter model name..."
                  disabled={trainingStatus?.is_training}
                  className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
                />
                <button
                  onClick={generateModelName}
                  disabled={!selectedDataset || trainingStatus?.is_training}
                  className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white px-4 py-3 rounded-lg transition-colors"
                >
                  Generate
                </button>
              </div>
            </div>

            {/* Epochs */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Training Epochs
              </label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(Math.max(1, Math.min(100, parseInt(e.target.value) || 15)))}
                min="1"
                max="100"
                disabled={trainingStatus?.is_training}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
              />
              <p className="text-sm text-gray-500 mt-1">
                More epochs = better training but takes longer (recommended: 10-30)
              </p>
            </div>

            {/* Start Training Button */}
            <button
              onClick={startTraining}
              disabled={loading || trainingStatus?.is_training || !selectedDataset || !modelName.trim()}
              className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white py-3 px-4 rounded-lg transition-colors font-medium"
            >
              {loading ? 'Starting Training...' : 
               trainingStatus?.is_training ? 'Training in Progress...' : 
               'Start Training'}
            </button>
          </div>

          {/* Status Messages */}
          {error && (
            <div className="mb-6 p-4 bg-red-100 border border-red-300 rounded-lg">
              <p className="text-red-700">{error}</p>
            </div>
          )}

          {success && (
            <div className="mb-6 p-4 bg-green-100 border border-green-300 rounded-lg">
              <p className="text-green-700">{success}</p>
            </div>
          )}

          {/* Training Information */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Training Information</h2>
            
            <div className="space-y-4 text-sm text-gray-600">
              <div>
                <h3 className="font-semibold text-gray-800 mb-2">What happens during training:</h3>
                <ul className="space-y-1 ml-4">
                  <li>• The AI model learns to recognize watermelon features</li>
                  <li>• It's trained to classify both variety (Crimsonsweet F1 vs others) and ripeness</li>
                  <li>• The model uses transfer learning from MobileNetV2 for better accuracy</li>
                  <li>• Training data is automatically balanced to handle class imbalances</li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold text-gray-800 mb-2">Training time estimates:</h3>
                <ul className="space-y-1 ml-4">
                  <li>• Small dataset (&lt; 500 images): 5-15 minutes</li>
                  <li>• Medium dataset (500-2000 images): 15-45 minutes</li>
                  <li>• Large dataset (&gt; 2000 images): 45+ minutes</li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold text-gray-800 mb-2">After training:</h3>
                <ul className="space-y-1 ml-4">
                  <li>• Your model will be automatically saved</li>
                  <li>• You can load and use it for classification</li>
                  <li>• Export to TensorFlow Lite for mobile deployment</li>
                  <li>• Evaluate performance on test data</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelTrainer;