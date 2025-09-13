# 🍉 Watermelon Classifier - AI-Powered Variety & Ripeness Detection

A comprehensive full-stack machine learning application that uses TensorFlow to classify watermelons by variety (specifically Crimsonsweet F1) and ripeness level (ripe/unripe). Built with React frontend, FastAPI backend, and MongoDB database.

## ✨ Features

### 🎯 Core Functionality
- **Variety Detection**: Specifically identifies Crimsonsweet F1 watermelons vs other varieties
- **Ripeness Classification**: Determines if watermelons are ripe or unripe
- **Confidence Scoring**: Provides detailed confidence scores for reliable classification
- **Image Input Options**: Supports file upload and camera capture
- **Batch Processing**: Can handle multiple images and dataset management

### 🔧 Advanced Features
- **Dataset Management**: Upload and organize training datasets via web interface
- **Model Training**: Train custom models with your own datasets
- **Model Evaluation**: Comprehensive performance metrics and confusion matrices
- **TensorFlow Lite Export**: Export models for mobile deployment
- **Background Training**: Non-blocking model training with progress tracking
- **Transfer Learning**: Uses MobileNetV2 for efficient and accurate classification

## 🛠️ Technology Stack

### Frontend
- **React 19** with modern hooks and components
- **Tailwind CSS** for responsive UI design
- **Axios** for API communication
- **React Router** for navigation
- **Camera API** integration for mobile photo capture

### Backend
- **FastAPI** for high-performance REST API
- **TensorFlow** for machine learning model training and inference
- **MongoDB** with Motor for async database operations
- **OpenCV & PIL** for image processing
- **Scikit-learn** for model evaluation metrics

### Infrastructure
- **Docker** containerization
- **Supervisor** for process management
- **CORS** enabled for cross-origin requests

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB
- Docker (optional)

### Installation

1. **Clone and setup backend:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Setup frontend:**
```bash
cd frontend
yarn install
```

3. **Environment configuration:**
   - Backend: Configure MongoDB URL in `backend/.env`
   - Frontend: Set backend URL in `frontend/.env`

4. **Start services:**
```bash
# Start all services with supervisor
sudo supervisorctl restart all

# Or manually:
# Backend: uvicorn server:app --host 0.0.0.0 --port 8001
# Frontend: yarn start
```

### First Steps

1. **Access the application**: Open http://localhost:3000
2. **Upload a dataset**: Go to Datasets tab and upload a ZIP file with watermelon images
3. **Train a model**: Use the Train Model tab to create your classifier
4. **Start classifying**: Upload watermelon images on the home page

## 📊 Dataset Format

### Expected Structure
```
dataset.zip
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
    └── ...
```

### Dataset Guidelines
- **Minimum 50+ images per class** for good training results
- **High-quality images** with good lighting and clear watermelon visibility
- **Diverse angles and backgrounds** for better generalization
- **Consistent image quality** across all classes
- **Supported formats**: JPG, JPEG, PNG, BMP

## 🔧 API Reference

### Core Endpoints

#### Health & Info
- `GET /api/health` - Check API health and model status
- `GET /api/info` - Get API information and features

#### Image Classification
- `POST /api/predict` - Classify watermelon image (requires image file)
- `GET /api/predictions` - Get recent classification results

#### Dataset Management
- `GET /api/datasets` - List all uploaded datasets
- `POST /api/dataset/upload` - Upload new dataset (ZIP file)

#### Model Training
- `POST /api/train` - Start model training
- `GET /api/train/status` - Check training progress

#### Model Management
- `GET /api/models` - List all trained models
- `POST /api/models/{name}/load` - Load model for inference
- `POST /api/models/{name}/evaluate` - Evaluate model performance
- `POST /api/models/{name}/export/tflite` - Export to TensorFlow Lite

### Response Format

#### Classification Result
```json
{
  "variety": "crimsonsweet",
  "ripeness": "ripe",
  "predicted_class": "crimsonsweet_ripe",
  "confidence": 0.94,
  "confidence_breakdown": {
    "crimsonsweet_ripe": 0.94,
    "crimsonsweet_unripe": 0.04,
    "other_variety": 0.01,
    "not_valid": 0.01
  },
  "is_valid": true,
  "is_crimsonsweet": true
}
```

## 🎯 Model Architecture

### Base Model
- **MobileNetV2** pre-trained on ImageNet for feature extraction
- **Transfer Learning** approach for efficient training
- **Image Size**: 224x224 pixels
- **Input Preprocessing**: Automatic resizing and normalization

### Classification Head
- **Global Average Pooling** layer
- **Dropout (0.3)** for regularization
- **Dense Layer** with softmax activation for multi-class classification
- **Class Balancing** with automatic weight calculation

### Training Configuration
- **Optimizer**: Adam with learning rate 1e-4
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Data Split**: 70% train, 20% validation, 10% test
- **Epochs**: Configurable (default 15)

## 📱 Mobile Integration

### TensorFlow Lite Export
Models can be exported to TensorFlow Lite format for mobile deployment:

```python
# Export model to TFLite
POST /api/models/{model_name}/export/tflite
```

### Camera Integration
The web interface supports camera capture for real-time classification:
- Automatically switches to back camera on mobile devices
- Real-time photo capture and classification
- Works on both desktop and mobile browsers

## 🔍 Performance Metrics

### Model Evaluation
- **Overall Accuracy**: Target >90% for production use
- **Per-Class Metrics**: Precision, Recall, F1-Score
- **Confusion Matrix**: Visual representation of classification performance
- **Confidence Thresholding**: Configurable confidence threshold (default 0.7)

### Expected Performance
- **Excellent**: >90% accuracy
- **Good**: 80-90% accuracy
- **Needs Improvement**: <80% accuracy

## 🛠️ Development

### Project Structure
```
├── backend/
│   ├── server.py           # Main FastAPI application
│   ├── ml_models.py        # ML model classes and utilities
│   ├── file_utils.py       # Dataset and file management
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js         # Main React component
│   │   ├── components/    # React components
│   │   └── ...
│   ├── package.json       # Node.js dependencies
│   └── .env              # Environment variables
└── README.md             # This file
```

### Key Components

#### Backend Classes
- `WatermelonClassifier`: Main ML model class
- `DatasetManager`: Handles dataset upload and organization
- Background training with progress tracking

#### Frontend Components
- `ImageClassifier`: Main classification interface
- `DatasetManager`: Dataset upload and management
- `ModelTrainer`: Model training configuration
- `ModelManager`: Model loading and evaluation

## 🐛 Troubleshooting

### Common Issues

#### "No model loaded" error
- Upload a dataset and train a model first
- Or load an existing trained model from the Models page

#### Dataset upload fails
- Ensure the file is a valid ZIP archive
- Check that images are organized in class folders
- Verify image formats are supported (JPG, PNG, BMP)

#### Training takes too long
- Reduce the number of epochs
- Use a smaller dataset for testing
- Training runs in background - you can close the browser

#### Low classification accuracy
- Increase training epochs (20-30)
- Add more diverse training images
- Ensure good image quality and proper labeling

### Log Files
- Backend logs: `/var/log/supervisor/backend.*.log`
- Frontend logs: `/var/log/supervisor/frontend.*.log`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the excellent ML framework
- **MobileNetV2** architecture for efficient image classification
- **FastAPI** for the high-performance backend framework
- **React** team for the modern frontend library

## 📞 Support

For questions, issues, or feature requests:
1. Check the troubleshooting section above
2. Review the API documentation
3. Create an issue in the repository
4. Check the application logs for detailed error information

---

**Built with ❤️ using modern ML and web technologies**
