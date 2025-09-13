import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ModelManager = () => {
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [evaluatingModel, setEvaluatingModel] = useState(null);

  useEffect(() => {
    loadModels();
    loadDatasets();
  }, []);

  const loadModels = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/models`);
      setModels(response.data);
      setError(null);
    } catch (err) {
      console.error('Error loading models:', err);
      setError('Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
    } catch (err) {
      console.error('Error loading datasets:', err);
    }
  };

  const loadModel = async (modelName) => {
    try {
      setError(null);
      const response = await axios.post(`${API}/models/${modelName}/load`);
      setSuccess(`Model "${modelName}" loaded successfully and ready for use!`);
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      console.error('Error loading model:', err);
      setError(err.response?.data?.detail || 'Failed to load model');
    }
  };

  const evaluateModel = async (modelName, datasetName) => {
    setEvaluatingModel(modelName);
    setError(null);
    setEvaluationResults(null);

    try {
      const response = await axios.post(`${API}/models/${modelName}/evaluate?dataset_name=${datasetName}`);
      setEvaluationResults({
        modelName,
        datasetName,
        ...response.data
      });
    } catch (err) {
      console.error('Error evaluating model:', err);
      setError(err.response?.data?.detail || 'Failed to evaluate model');
    } finally {
      setEvaluatingModel(null);
    }
  };

  const exportToTFLite = async (modelName) => {
    try {
      setError(null);
      const response = await axios.post(`${API}/models/${modelName}/export/tflite`);
      setSuccess(`Model "${modelName}" exported to TensorFlow Lite successfully!`);
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      console.error('Error exporting model:', err);
      setError(err.response?.data?.detail || 'Failed to export model');
    }
  };

  const formatSize = (sizeInMB) => {
    if (sizeInMB < 1) {
      return `${(sizeInMB * 1024).toFixed(1)} KB`;
    } else if (sizeInMB > 1024) {
      return `${(sizeInMB / 1024).toFixed(1)} GB`;
    } else {
      return `${sizeInMB.toFixed(1)} MB`;
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 0.9) return 'text-green-600';
    if (accuracy >= 0.8) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-8">Model Management</h1>

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

          {/* Models List */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-8">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-semibold text-gray-800">Trained Models</h2>
              <button
                onClick={loadModels}
                disabled={loading}
                className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg transition-colors"
              >
                {loading ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            {loading ? (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <p className="mt-2 text-gray-600">Loading models...</p>
              </div>
            ) : models.length === 0 ? (
              <div className="text-center py-8">
                <div className="text-4xl mb-4">🤖</div>
                <p className="text-gray-600">No trained models found. Train your first model to get started!</p>
                <p className="text-sm text-gray-500 mt-2">
                  Go to the "Train Model" section to create your watermelon classifier.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {models.map((model, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                    <div className="flex flex-wrap items-start justify-between mb-4">
                      <div className="flex-1 min-w-0 mr-4">
                        <h3 className="text-lg font-semibold text-gray-800 truncate">
                          {model.name}
                        </h3>
                        <div className="flex flex-wrap items-center gap-4 mt-2 text-sm text-gray-600">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            model.format === 'SavedModel' ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800'
                          }`}>
                            {model.format}
                          </span>
                          <span>Size: {formatSize(model.size_mb)}</span>
                          {model.class_names && (
                            <span>Classes: {model.class_names.length}</span>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex flex-wrap gap-2">
                        <button
                          onClick={() => loadModel(model.name)}
                          className="bg-green-600 hover:bg-green-700 text-white px-3 py-1.5 rounded text-sm transition-colors"
                        >
                          Load for Use
                        </button>
                        <button
                          onClick={() => exportToTFLite(model.name)}
                          className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded text-sm transition-colors"
                        >
                          Export TFLite
                        </button>
                      </div>
                    </div>

                    {/* Model Details */}
                    {model.class_names && (
                      <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                        <h4 className="font-medium text-gray-700 mb-2">Model Details</h4>
                        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-600">
                          <div>
                            <span className="font-medium">Classes:</span>
                            <p className="mt-1">{model.class_names.join(', ')}</p>
                          </div>
                          {model.confidence_threshold && (
                            <div>
                              <span className="font-medium">Confidence Threshold:</span> {model.confidence_threshold}
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Evaluation Section */}
                    <div className="border-t border-gray-100 pt-4">
                      <h4 className="font-medium text-gray-700 mb-3">Model Evaluation</h4>
                      <div className="flex flex-wrap items-center gap-3">
                        <select
                          className="border border-gray-300 rounded px-3 py-1.5 text-sm"
                          onChange={(e) => {
                            if (e.target.value) {
                              evaluateModel(model.name, e.target.value);
                            }
                          }}
                          disabled={evaluatingModel === model.name}
                        >
                          <option value="">Select dataset to evaluate...</option>
                          {datasets.map((dataset) => (
                            <option key={dataset.name} value={dataset.name}>
                              {dataset.name} ({dataset.test_samples} test samples)
                            </option>
                          ))}
                        </select>
                        
                        {evaluatingModel === model.name && (
                          <div className="flex items-center text-sm text-blue-600">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                            Evaluating...
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Evaluation Results */}
          {evaluationResults && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-6">
                Evaluation Results: {evaluationResults.modelName}
              </h2>
              
              <div className="mb-6">
                <p className="text-sm text-gray-600 mb-4">
                  Evaluated on dataset: <span className="font-medium">{evaluationResults.datasetName}</span>
                  <span className="ml-4">Test samples: {evaluationResults.num_test_samples}</span>
                </p>
                
                <div className="flex items-center">
                  <span className="text-lg font-medium text-gray-700 mr-3">Overall Accuracy:</span>
                  <span className={`text-2xl font-bold ${getAccuracyColor(evaluationResults.accuracy)}`}>
                    {(evaluationResults.accuracy * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Classification Report */}
              {evaluationResults.classification_report && (
                <div className="mb-6">
                  <h3 className="font-semibold text-gray-800 mb-3">Per-Class Performance</h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full border border-gray-200 rounded-lg">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Class</th>
                          <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Precision</th>
                          <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Recall</th>
                          <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">F1-Score</th>
                          <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Support</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(evaluationResults.classification_report)
                          .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
                          .map(([className, metrics]) => (
                            <tr key={className} className="border-t border-gray-200">
                              <td className="px-4 py-2 text-sm font-medium text-gray-800 capitalize">
                                {className.replace('_', ' ')}
                              </td>
                              <td className="px-4 py-2 text-sm text-gray-600">
                                {(metrics.precision * 100).toFixed(1)}%
                              </td>
                              <td className="px-4 py-2 text-sm text-gray-600">
                                {(metrics.recall * 100).toFixed(1)}%
                              </td>
                              <td className="px-4 py-2 text-sm text-gray-600">
                                {(metrics['f1-score'] * 100).toFixed(1)}%
                              </td>
                              <td className="px-4 py-2 text-sm text-gray-600">
                                {metrics.support}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Confusion Matrix */}
              {evaluationResults.confusion_matrix && (
                <div>
                  <h3 className="font-semibold text-gray-800 mb-3">Confusion Matrix</h3>
                  <div className="text-sm text-gray-600 mb-2">
                    Rows: Actual classes, Columns: Predicted classes
                  </div>
                  <div className="overflow-x-auto">
                    <table className="border border-gray-200 rounded-lg">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-3 py-2 text-sm font-medium text-gray-700">Actual \ Predicted</th>
                          {evaluationResults.confusion_matrix[0].map((_, colIndex) => (
                            <th key={colIndex} className="px-3 py-2 text-sm font-medium text-gray-700 text-center">
                              Class {colIndex}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {evaluationResults.confusion_matrix.map((row, rowIndex) => (
                          <tr key={rowIndex} className="border-t border-gray-200">
                            <td className="px-3 py-2 text-sm font-medium text-gray-700 bg-gray-50">
                              Class {rowIndex}
                            </td>
                            {row.map((value, colIndex) => (
                              <td
                                key={colIndex}
                                className={`px-3 py-2 text-sm text-center ${
                                  rowIndex === colIndex 
                                    ? 'bg-green-100 text-green-800 font-medium' 
                                    : value > 0 
                                    ? 'bg-red-50 text-red-700' 
                                    : 'text-gray-600'
                                }`}
                              >
                                {value}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              
              <button
                onClick={() => setEvaluationResults(null)}
                className="mt-6 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
              >
                Close Results
              </button>
            </div>
          )}

          {/* Model Usage Instructions */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">How to Use Your Models</h2>
            
            <div className="space-y-4 text-sm text-gray-600">
              <div>
                <h3 className="font-semibold text-gray-800 mb-2">Loading a Model:</h3>
                <p>Click "Load for Use" to make a model active for classification. Only one model can be active at a time.</p>
              </div>

              <div>
                <h3 className="font-semibold text-gray-800 mb-2">Model Evaluation:</h3>
                <p>Select a test dataset to evaluate model performance. This shows accuracy, precision, recall, and confusion matrix.</p>
              </div>

              <div>
                <h3 className="font-semibold text-gray-800 mb-2">TensorFlow Lite Export:</h3>
                <p>Export models to TFLite format for mobile deployment. The .tflite file will be optimized for mobile devices.</p>
              </div>

              <div>
                <h3 className="font-semibold text-gray-800 mb-2">Performance Tips:</h3>
                <ul className="space-y-1 ml-4">
                  <li>• Accuracy above 90% is excellent</li>
                  <li>• Accuracy 80-90% is good for most applications</li>
                  <li>• Below 80% may need more training data or longer training</li>
                  <li>• Check per-class performance for class-specific issues</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelManager;