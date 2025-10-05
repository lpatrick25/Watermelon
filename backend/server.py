from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import asyncio
import json

# Import ML components
from ml_models import watermelon_classifier, WatermelonClassifier
from file_utils import dataset_manager, DatasetManager

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Watermelon Classifier API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables for training status
training_status = {"is_training": False, "progress": 0, "message": ""}

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class PredictionResult(BaseModel):
    variety: str
    ripeness: str
    predicted_class: str
    confidence: float
    confidence_breakdown: Dict[str, float]
    visual_analysis: Dict[str, float]
    shape_analysis: Dict[str, float]
    surface_analysis: Dict[str, float]
    is_valid: bool
    is_crimsonsweet: bool

class TrainingRequest(BaseModel):
    dataset_name: str
    epochs: int = 15
    model_name: str = "watermelon_model"

class DatasetInfo(BaseModel):
    name: str
    path: str
    classes: List[str]
    train_samples: int
    val_samples: int
    test_samples: int

class ModelInfo(BaseModel):
    name: str
    path: str
    format: str
    size_mb: float
    class_names: Optional[List[str]] = None
    confidence_threshold: Optional[float] = None

# Basic routes
@api_router.get("/")
async def root():
    return {"message": "Watermelon Classifier API - Ready to classify your watermelons!"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# ML Prediction Endpoints
@api_router.post("/predict", response_model=PredictionResult)
async def predict_watermelon(file: UploadFile = File(...)):
    """Predict watermelon variety and ripeness from uploaded image"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image (JPG, PNG, BMP)")
        
        # Check if model is loaded
        if watermelon_classifier.model is None:
            raise HTTPException(status_code=400, detail="No model loaded. Please load or train a model first.")
        
        # Read image data
        image_data = await file.read()
        
        # Make prediction
        result = watermelon_classifier.predict(image_data)
        
        # Store prediction in database
        prediction_record = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "prediction": result,
            "timestamp": datetime.utcnow()
        }
        await db.predictions.insert_one(prediction_record)
        
        return PredictionResult(**result)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@api_router.get("/predictions")
async def get_predictions(limit: int = 50):
    """Get recent predictions"""
    try:
        predictions = await db.predictions.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return predictions
    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching predictions")

# Dataset Management Endpoints
@api_router.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...), dataset_name: str = ""):
    """Upload and extract dataset zip file"""
    try:
        # Validate file type
        if not file.filename or not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="File must be a ZIP archive")
        
        # Use filename as dataset name if not provided
        if not dataset_name:
            dataset_name = Path(file.filename).stem
        
        # Save uploaded file
        zip_path = await dataset_manager.save_uploaded_file(file, "datasets")
        
        # Extract and organize dataset
        dataset_info = dataset_manager.extract_dataset(zip_path, dataset_name)
        
        # Clean up uploaded zip file
        os.remove(zip_path)
        
        return {
            "message": "Dataset uploaded and extracted successfully",
            "dataset_info": dataset_info
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Dataset upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error uploading dataset")

@api_router.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List all available datasets"""
    try:
        datasets = dataset_manager.list_datasets()
        return [DatasetInfo(**dataset) for dataset in datasets]
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing datasets")

# Model Training Endpoints
async def train_model_background(dataset_name: str, epochs: int, model_name: str):
    """Background task for model training"""
    global training_status
    
    try:
        training_status = {"is_training": True, "progress": 0, "message": "Starting training..."}
        
        # Get dataset paths
        datasets = dataset_manager.list_datasets()
        dataset = next((d for d in datasets if d["name"] == dataset_name), None)
        
        if not dataset:
            training_status = {"is_training": False, "progress": 0, "message": f"Dataset '{dataset_name}' not found"}
            return
        
        dataset_path = Path(dataset["path"])
        train_path = str(dataset_path / "train")
        val_path = str(dataset_path / "val")
        
        training_status["message"] = "Initializing model..."
        training_status["progress"] = 10
        
        # Train model
        training_status["message"] = "Training model..."
        training_status["progress"] = 20
        
        training_report = watermelon_classifier.train_model(train_path, val_path, epochs)
        
        training_status["message"] = "Saving model..."
        training_status["progress"] = 90

        # Save model
        model_path = dataset_manager.models_path / model_name
        watermelon_classifier.save_model(str(model_path))

        # Save training report
        report_path = dataset_manager.models_path / f"{model_name}_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(training_report, f, indent=2)
        
        training_status = {
            "is_training": False, 
            "progress": 100, 
            "message": f"Training completed successfully! Final accuracy: {training_report['final_val_accuracy']:.3f}"
        }
        
    except Exception as e:
        training_status = {"is_training": False, "progress": 0, "message": f"Training failed: {str(e)}"}
        logger.error(f"Training error: {str(e)}")

@api_router.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training in background"""
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Add training task to background
    background_tasks.add_task(
        train_model_background, 
        request.dataset_name, 
        request.epochs, 
        request.model_name
    )
    
    return {"message": "Training started", "status": training_status}

@api_router.get("/train/status")
async def get_training_status():
    """Get current training status"""
    return training_status

# Model Management Endpoints
@api_router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all saved models"""
    try:
        models = dataset_manager.list_models()
        return [ModelInfo(**model) for model in models]
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing models")

@api_router.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a specific model for inference"""
    try:
        # Try directory path first
        model_path = dataset_manager.models_path / model_name
        
        # If directory doesn't exist, try .keras or .h5 file
        if not model_path.exists():
            model_path_keras = dataset_manager.models_path / f"{model_name}.keras"
            model_path_h5 = dataset_manager.models_path / f"{model_name}.h5"
            
            if model_path_keras.exists():
                model_path = model_path_keras
            elif model_path_h5.exists():
                model_path = model_path_h5
            else:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Load the model
        watermelon_classifier.load_model(str(model_path))
        
        return {"message": f"Model '{model_name}' loaded successfully"}
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading model")

@api_router.post("/models/{model_name}/evaluate")
async def evaluate_model(model_name: str, dataset_name: str):
    """Evaluate model performance on test dataset"""
    try:
        # Try directory path first
        model_path = dataset_manager.models_path / model_name

        # If not found, try .keras or .h5 file
        if not model_path.exists():
            model_path_keras = dataset_manager.models_path / f"{model_name}.keras"
            model_path_h5 = dataset_manager.models_path / f"{model_name}.h5"
            
            if model_path_keras.exists():
                model_path = model_path_keras
            elif model_path_h5.exists():
                model_path = model_path_h5
            else:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Load model
        temp_classifier = WatermelonClassifier()
        temp_classifier.load_model(str(model_path))
        
        # Get dataset test path
        datasets = dataset_manager.list_datasets()
        dataset = next((d for d in datasets if d["name"] == dataset_name), None)
        
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        test_path = str(Path(dataset["path"]) / "test")
        
        # Evaluate model
        evaluation_results = temp_classifier.evaluate_model(test_path)
        
        return evaluation_results
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error evaluating model")

@api_router.post("/models/{model_name}/export/tflite")
async def export_model_tflite(model_name: str):
    """Export model to TensorFlow Lite format"""
    try:
        # Try directory path first
        model_path = dataset_manager.models_path / model_name
        
        # If directory doesn't exist, try .keras or .h5 file
        if not model_path.exists():
            model_path_keras = dataset_manager.models_path / f"{model_name}.keras"
            model_path_h5 = dataset_manager.models_path / f"{model_name}.h5"
            
            if model_path_keras.exists():
                model_path = model_path_keras
            elif model_path_h5.exists():
                model_path = model_path_h5
            else:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Define TFLite output path
        model_name_no_ext = Path(model_name).stem
        tflite_path = dataset_manager.models_path / f"{model_name_no_ext}.tflite"
        
        # Export to TFLite
        temp_classifier = WatermelonClassifier()
        temp_classifier.export_tflite(str(model_path), str(tflite_path))
        
        return {
            "message": f"Model exported to TFLite successfully",
            "tflite_path": str(tflite_path)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error exporting model")

# Health check and info endpoints
@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": watermelon_classifier.model is not None,
        "available_classes": watermelon_classifier.class_names,
        "confidence_threshold": watermelon_classifier.confidence_threshold
    }

@api_router.get("/info")
async def get_api_info():
    """Get API information"""
    return {
        "title": "Watermelon Classifier API",
        "version": "1.0.0",
        "description": "AI-powered watermelon variety and ripeness classification",
        "features": [
            "Crimsonsweet F1 variety detection",
            "Ripeness classification (ripe/unripe)",
            "Confidence-based validation",
            "Dataset management",
            "Model training and evaluation",
            "TensorFlow Lite export"
        ],
        "supported_formats": ["JPG", "JPEG", "PNG", "BMP"],
        "max_file_size": "10MB"
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Watermelon Classifier API...")
    
    # Clean up old temp files
    dataset_manager.cleanup_temp_files()
    
    # Try to load a default model if available
    models = dataset_manager.list_models()
    if models:
        try:
            default_model = models[0]
            watermelon_classifier.load_model(default_model["path"])
            logger.info(f"Loaded default model: {default_model['name']}")
        except Exception as e:
            logger.warning(f"Could not load default model: {str(e)}")
    
    logger.info("Watermelon Classifier API started successfully!")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    logger.info("Application shutdown complete")
