import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DatasetManager = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  // Load datasets on component mount
  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
      setError(null);
    } catch (err) {
      console.error('Error loading datasets:', err);
      setError('Failed to load datasets');
    } finally {
      setLoading(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      uploadDataset(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      uploadDataset(file);
    }
  };

  const uploadDataset = async (file) => {
    if (!file.name.endsWith('.zip')) {
      setError('Please select a ZIP file containing your dataset');
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('dataset_name', file.name.replace('.zip', ''));

      const response = await axios.post(`${API}/dataset/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setSuccess(`Dataset "${file.name}" uploaded successfully!`);
      loadDatasets(); // Refresh dataset list
      
      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.detail || 'Failed to upload dataset');
    } finally {
      setUploading(false);
    }
  };

  const formatClassList = (classes) => {
    if (!classes || classes.length === 0) return 'No classes';
    return classes.join(', ');
  };

  const getTotalSamples = (dataset) => {
    return dataset.train_samples + dataset.val_samples + dataset.test_samples;
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-8">Dataset Management</h1>

          {/* Upload Section */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Upload New Dataset</h2>
            
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive 
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-gray-300 hover:border-blue-400 bg-gray-50'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="text-4xl mb-4">📁</div>
              <p className="text-lg text-gray-600 mb-4">
                Drag and drop your dataset ZIP file here
              </p>
              <p className="text-sm text-gray-500 mb-4">
                Dataset should contain images organized in folders by class
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg transition-colors"
              >
                {uploading ? 'Uploading...' : 'Choose File'}
              </button>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".zip"
              onChange={handleFileSelect}
              className="hidden"
            />

            {/* Dataset Structure Info */}
            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold text-blue-800 mb-2">Expected Dataset Structure:</h3>
              <pre className="text-sm text-blue-700">
{`dataset.zip
├── crimsonsweet_ripe/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── crimsonsweet_unripe/
│   ├── image1.jpg
│   └── ...
├── other_variety/
│   └── ...
└── not_valid/
    └── ...`}
              </pre>
              <p className="text-sm text-blue-600 mt-2">
                The system will automatically split your data into train/validation/test sets (70/20/10)
              </p>
            </div>
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

          {/* Datasets List */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-semibold text-gray-800">Available Datasets</h2>
              <button
                onClick={loadDatasets}
                disabled={loading}
                className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg transition-colors"
              >
                {loading ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            {loading ? (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <p className="mt-2 text-gray-600">Loading datasets...</p>
              </div>
            ) : datasets.length === 0 ? (
              <div className="text-center py-8">
                <div className="text-4xl mb-4">📂</div>
                <p className="text-gray-600">No datasets found. Upload your first dataset to get started!</p>
              </div>
            ) : (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {datasets.map((dataset, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between mb-3">
                      <h3 className="font-semibold text-gray-800 truncate flex-1 mr-2">
                        {dataset.name}
                      </h3>
                      <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                        Dataset
                      </span>
                    </div>

                    <div className="space-y-2 text-sm text-gray-600">
                      <div>
                        <span className="font-medium">Classes:</span>
                        <p className="text-xs mt-1 break-words">{formatClassList(dataset.classes)}</p>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="font-medium">Train:</span> {dataset.train_samples}
                        </div>
                        <div>
                          <span className="font-medium">Val:</span> {dataset.val_samples}
                        </div>
                        <div>
                          <span className="font-medium">Test:</span> {dataset.test_samples}
                        </div>
                        <div>
                          <span className="font-medium">Total:</span> {getTotalSamples(dataset)}
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-gray-100">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-500">
                          {dataset.classes?.length || 0} classes
                        </span>
                        <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                          View Details
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Instructions */}
          <div className="mt-8 bg-yellow-50 border border-yellow-200 rounded-lg p-6">
            <h3 className="font-semibold text-yellow-800 mb-2">💡 Tips for Better Results:</h3>
            <ul className="text-sm text-yellow-700 space-y-1">
              <li>• Ensure images are clear and well-lit</li>
              <li>• Include diverse angles and backgrounds</li>
              <li>• Have at least 50+ images per class for good training</li>
              <li>• Use consistent image quality across all classes</li>
              <li>• Include both ripe and unripe examples for Crimsonsweet F1</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetManager;