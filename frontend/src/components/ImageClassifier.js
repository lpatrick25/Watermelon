import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ImageClassifier = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);

  // Handle file selection
  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(file);
      setError(null);
      setPrediction(null);
    } else {
      setError('Please select a valid image file (JPG, PNG, BMP)');
    }
  };

  // Handle drag and drop
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFileSelect(files[0]);
    }
  }, []);

  // Handle file input change
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
        setError(null);
      }
    } catch (err) {
      setError('Camera access denied or not available');
      console.error('Camera error:', err);
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
    }
  };

  // Capture photo from camera
  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        handleFileSelect(file);
        stopCamera();
      }, 'image/jpeg', 0.8);
    }
  };

  // Make prediction
  const classifyImage = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post(`${API}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Prediction response:', response.data); // Debug log
      setPrediction(response.data);
    } catch (err) {
      console.error('Classification error:', err);
      setError(err.response?.data?.detail || 'Classification failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Clear all
  const clearAll = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
    stopCamera();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Helper to format feature names for display
  const formatFeatureName = (name) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
        Watermelon Image Classifier
      </h2>

      {/* Image Upload Area */}
      <div className="mb-6">
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive 
              ? 'border-green-500 bg-green-50' 
              : 'border-gray-300 hover:border-green-400 bg-gray-50'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {!preview ? (
            <div>
              <div className="text-4xl mb-4">🍉</div>
              <p className="text-lg text-gray-600 mb-4">
                Drag and drop your watermelon image here
              </p>
              <p className="text-sm text-gray-500 mb-4">or</p>
              <div className="flex justify-center space-x-4">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg transition-colors"
                >
                  Choose File
                </button>
                <button
                  onClick={cameraActive ? stopCamera : startCamera}
                  className={`px-6 py-2 rounded-lg transition-colors ${
                    cameraActive 
                      ? 'bg-red-600 hover:bg-red-700 text-white' 
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
                >
                  {cameraActive ? 'Stop Camera' : 'Use Camera'}
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <img
                src={preview}
                alt="Preview"
                className="max-w-full max-h-64 mx-auto rounded-lg shadow-md"
              />
              <div className="flex justify-center space-x-4">
                <button
                  onClick={classifyImage}
                  disabled={loading}
                  className="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg transition-colors"
                >
                  {loading ? 'Analyzing...' : 'Classify Watermelon'}
                </button>
                <button
                  onClick={clearAll}
                  className="bg-gray-600 hover:bg-gray-700 text-white px-6 py-2 rounded-lg transition-colors"
                >
                  Clear
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Camera View */}
        {cameraActive && (
          <div className="mt-4 text-center">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="max-w-full rounded-lg shadow-md"
            />
            <div className="mt-4">
              <button
                onClick={capturePhoto}
                className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg mr-4"
              >
                📸 Capture Photo
              </button>
            </div>
          </div>
        )}

        <canvas ref={canvasRef} style={{ display: 'none' }} />
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 border border-red-300 rounded-lg">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* Prediction Results */}
      {prediction && (
        <div className="bg-gray-50 rounded-lg p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Classification Results</h3>
          
          {/* Main Result */}
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white rounded-lg p-4 shadow">
              <h4 className="font-semibold text-gray-700 mb-2">Variety</h4>
              <div className="flex items-center">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  prediction.is_crimsonsweet 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-yellow-100 text-yellow-800'
                }`}>
                  {prediction.variety === 'crimsonsweet' ? 'Crimsonsweet F1' : prediction.variety === 'other' ? 'Other Variety' : 'Unknown'}
                </span>
              </div>
            </div>
            
            <div className="bg-white rounded-lg p-4 shadow">
              <h4 className="font-semibold text-gray-700 mb-2">Ripeness</h4>
              <div className="flex items-center">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  prediction.ripeness === 'ripe' 
                    ? 'bg-green-100 text-green-800' 
                    : prediction.ripeness === 'unripe'
                    ? 'bg-orange-100 text-orange-800'
                    : 'bg-gray-100 text-gray-800'
                }`}>
                  {prediction.ripeness === 'ripe' ? '🍉 Ripe' : 
                   prediction.ripeness === 'unripe' ? '🟡 Unripe' : '❓ Unknown'}
                </span>
              </div>
            </div>
          </div>

          {/* Confidence */}
          <div className="mb-6">
            <h4 className="font-semibold text-gray-700 mb-2">Overall Confidence</h4>
            <div className="flex items-center">
              <div className="flex-1 bg-gray-200 rounded-full h-3 mr-3">
                <div
                  className={`h-3 rounded-full ${
                    prediction.confidence >= 0.8 ? 'bg-green-500' :
                    prediction.confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${prediction.confidence * 100}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Validation Status */}
          <div className="mb-6">
            <div className={`p-3 rounded-lg ${
              prediction.is_valid 
                ? 'bg-green-100 border border-green-300' 
                : 'bg-red-100 border border-red-300'
            }`}>
              <div className="flex items-center">
                <span className="text-lg mr-2">
                  {prediction.is_valid ? '✅' : '❌'}
                </span>
                <span className={`font-medium ${
                  prediction.is_valid ? 'text-green-800' : 'text-red-800'
                }`}>
                  {prediction.is_valid 
                    ? 'Valid watermelon detected' 
                    : 'Invalid or unclear image - please try another photo'
                  }
                </span>
              </div>
            </div>
          </div>

          {/* Confidence Breakdown */}
          <div className="mb-6">
            <h4 className="font-semibold text-gray-700 mb-3">Confidence Breakdown</h4>
            <div className="space-y-2">
              {Object.entries(prediction.confidence_breakdown).map(([className, confidence]) => (
                <div key={className} className="flex items-center">
                  <div className="w-32 text-sm text-gray-600 capitalize">
                    {className.replace('_', ' ')}
                  </div>
                  <div className="flex-1 bg-gray-200 rounded-full h-2 mx-3">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${confidence * 100}%` }}
                    ></div>
                  </div>
                  <div className="w-12 text-sm text-gray-700">
                    {(confidence * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Feature Analysis */}
          {prediction && (prediction.visual_analysis || prediction.shape_analysis || prediction.surface_analysis) && (
            <div className="mb-6">
              <h4 className="font-semibold text-gray-700 mb-3">Feature Analysis</h4>
              <div className="grid md:grid-cols-3 gap-4">
                {/* Visual Analysis */}
                {prediction.visual_analysis && (
                  <div className="bg-white rounded-lg p-4 shadow">
                    <h5 className="font-semibold text-gray-700 mb-2">Visual Features</h5>
                    <div className="space-y-2">
                      {Object.entries(prediction.visual_analysis).map(([key, value]) => (
                        <div key={key} className="flex items-center">
                          <div className="w-32 text-sm text-gray-600 capitalize">
                            {formatFeatureName(key)}
                          </div>
                          <div className="flex-1 bg-gray-200 rounded-full h-2 mx-3">
                            <div
                              className="bg-green-500 h-2 rounded-full"
                              style={{ width: `${value * 100}%` }}
                            ></div>
                          </div>
                          <div className="w-12 text-sm text-gray-700">
                            {(value * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {/* Shape Analysis */}
                {prediction.shape_analysis && (
                  <div className="bg-white rounded-lg p-4 shadow">
                    <h5 className="font-semibold text-gray-700 mb-2">Shape Features</h5>
                    <div className="space-y-2">
                      {Object.entries(prediction.shape_analysis).map(([key, value]) => (
                        <div key={key} className="flex items-center">
                          <div className="w-32 text-sm text-gray-600 capitalize">
                            {formatFeatureName(key)}
                          </div>
                          <div className="flex-1 bg-gray-200 rounded-full h-2 mx-3">
                            <div
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ width: `${value * 100}%` }}
                            ></div>
                          </div>
                          <div className="w-12 text-sm text-gray-700">
                            {(value * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {/* Surface Analysis */}
                {prediction.surface_analysis && (
                  <div className="bg-white rounded-lg p-4 shadow">
                    <h5 className="font-semibold text-gray-700 mb-2">Surface Features</h5>
                    <div className="space-y-2">
                      {Object.entries(prediction.surface_analysis).map(([key, value]) => (
                        <div key={key} className="flex items-center">
                          <div className="w-32 text-sm text-gray-600 capitalize">
                            {formatFeatureName(key)}
                          </div>
                          <div className="flex-1 bg-gray-200 rounded-full h-2 mx-3">
                            <div
                              className="bg-purple-500 h-2 rounded-full"
                              style={{ width: `${value * 100}%` }}
                            ></div>
                          </div>
                          <div className="w-12 text-sm text-gray-700">
                            {(value * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ImageClassifier;