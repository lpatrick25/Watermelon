import os
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import logging
import aiofiles
from fastapi import UploadFile
import uuid

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manage dataset uploads and organization"""
    
    def __init__(self, base_path: str = None):
        # Use project-relative paths instead of hardcoded /app/...
        project_root = Path(__file__).resolve().parent.parent  # goes up to "backend/.."
        backend_path = project_root / "backend"

        if base_path is None:
            base_path = backend_path / "datasets"

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create standard directories under backend/
        self.models_path = backend_path / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.temp_path = backend_path / "temp"
        self.temp_path.mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_file(self, file: UploadFile, subfolder: str = "") -> str:
        """Save uploaded file and return the path"""
        try:
            # Create unique filename to avoid conflicts
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix if file.filename else ""
            filename = f"{file_id}{file_extension}"
            
            # Create subfolder if specified
            save_dir = self.temp_path / subfolder if subfolder else self.temp_path
            save_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = save_dir / filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            logger.info(f"File saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise ValueError(f"Failed to save file: {str(e)}")
    
    def extract_dataset(self, zip_path: str, dataset_name: str) -> Dict[str, str]:
        """Extract dataset zip file and organize into train/val/test structure"""
        try:
            dataset_dir = self.base_path / dataset_name
            
            # Remove existing dataset if it exists
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            
            # Look for standard dataset structure
            train_dir = None
            val_dir = None
            test_dir = None
            
            # Check for existing train/val/test structure
            for root, dirs, files in os.walk(dataset_dir):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    if dir_name.lower() in ['train', 'training']:
                        train_dir = str(dir_path)
                    elif dir_name.lower() in ['val', 'valid', 'validation']:
                        val_dir = str(dir_path)
                    elif dir_name.lower() in ['test', 'testing']:
                        test_dir = str(dir_path)
            
            # If no standard structure found, organize files
            if not train_dir:
                train_dir, val_dir, test_dir = self.organize_dataset(dataset_dir)
            
            # Validate dataset structure
            self.validate_dataset_structure(train_dir, val_dir, test_dir)
            
            return {
                "dataset_path": str(dataset_dir),
                "train_path": train_dir,
                "val_path": val_dir,
                "test_path": test_dir,
                "classes": self.get_dataset_classes(train_dir)
            }
            
        except Exception as e:
            logger.error(f"Error extracting dataset: {str(e)}")
            raise ValueError(f"Failed to extract dataset: {str(e)}")
    
    def organize_dataset(self, dataset_dir: Path) -> tuple:
        """Organize flat dataset into train/val/test structure"""
        try:
            # Create train/val/test directories
            train_dir = dataset_dir / "train"
            val_dir = dataset_dir / "val"
            test_dir = dataset_dir / "test"
            
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)
            test_dir.mkdir(exist_ok=True)
            
            # Find all class directories (containing images)
            class_dirs = []
            for item in dataset_dir.iterdir():
                if item.is_dir() and item.name not in ['train', 'val', 'test']:
                    # Check if directory contains images
                    image_files = [f for f in item.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                    if image_files:
                        class_dirs.append(item)
            
            # If no class directories found, create from flat structure
            if not class_dirs:
                # Look for images directly in dataset_dir and organize by filename patterns
                all_images = [f for f in dataset_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                if all_images:
                    # Create a default class directory
                    default_class = train_dir / "watermelon"
                    default_class.mkdir(exist_ok=True)
                    
                    # Move all images to default class
                    for img in all_images:
                        shutil.move(str(img), str(default_class / img.name))
                    
                    return str(train_dir), str(val_dir), str(test_dir)
            
            # Split each class into train/val/test (70/20/10)
            for class_dir in class_dirs:
                class_name = class_dir.name
                
                # Get all images in class
                images = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                images.sort()  # Ensure consistent ordering
                
                if not images:
                    continue
                
                # Create class directories in train/val/test
                train_class_dir = train_dir / class_name
                val_class_dir = val_dir / class_name
                test_class_dir = test_dir / class_name
                
                train_class_dir.mkdir(exist_ok=True)
                val_class_dir.mkdir(exist_ok=True)
                test_class_dir.mkdir(exist_ok=True)
                
                # Split images
                num_images = len(images)
                train_split = int(0.7 * num_images)
                val_split = int(0.9 * num_images)
                
                # Move images to respective directories
                for i, img in enumerate(images):
                    if i < train_split:
                        dest_dir = train_class_dir
                    elif i < val_split:
                        dest_dir = val_class_dir
                    else:
                        dest_dir = test_class_dir
                    
                    shutil.move(str(img), str(dest_dir / img.name))
                
                # Remove original class directory
                if class_dir.exists() and not any(class_dir.iterdir()):
                    class_dir.rmdir()
            
            return str(train_dir), str(val_dir), str(test_dir)
            
        except Exception as e:
            logger.error(f"Error organizing dataset: {str(e)}")
            raise ValueError(f"Failed to organize dataset: {str(e)}")
    
    def validate_dataset_structure(self, train_dir: str, val_dir: str, test_dir: str):
        """Validate that dataset has proper structure"""
        for dir_path, dir_name in [(train_dir, "train"), (val_dir, "validation"), (test_dir, "test")]:
            if not os.path.exists(dir_path):
                raise ValueError(f"{dir_name} directory not found: {dir_path}")
            
            # Check for class subdirectories
            class_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            if not class_dirs:
                raise ValueError(f"No class directories found in {dir_name} directory")
            
            # Check for images in class directories
            for class_dir in class_dirs:
                class_path = os.path.join(dir_path, class_dir)
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if not images:
                    logger.warning(f"No images found in {dir_name}/{class_dir}")
    
    def get_dataset_classes(self, train_dir: str) -> List[str]:
        """Get list of classes in the dataset"""
        if not os.path.exists(train_dir):
            return []
        
        classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        return sorted(classes)
    
    def list_datasets(self) -> List[Dict]:
        """List all available datasets"""
        datasets = []
        
        for dataset_dir in self.base_path.iterdir():
            if dataset_dir.is_dir():
                dataset_info = {
                    "name": dataset_dir.name,
                    "path": str(dataset_dir),
                    "classes": [],
                    "train_samples": 0,
                    "val_samples": 0,
                    "test_samples": 0
                }
                
                # Check for train directory and count samples
                train_dir = dataset_dir / "train"
                if train_dir.exists():
                    classes = self.get_dataset_classes(str(train_dir))
                    dataset_info["classes"] = classes
                    
                    # Count samples
                    for split in ["train", "val", "test"]:
                        split_dir = dataset_dir / split
                        if split_dir.exists():
                            count = 0
                            for class_dir in classes:
                                class_path = split_dir / class_dir
                                if class_path.exists():
                                    images = [f for f in class_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                                    count += len(images)
                            dataset_info[f"{split}_samples"] = count
                
                datasets.append(dataset_info)
        
        return datasets

    def list_models(self) -> List[Dict]:
        """List all saved models"""
        models = []
        
        for item in self.models_path.iterdir():
            if item.is_dir():
                # Check for saved_model.pb, .h5, or .keras files in directories
                has_saved_model = (item / "saved_model.pb").exists()
                h5_files = list(item.glob("*.h5"))
                keras_files = list(item.glob("*.keras"))
                
                if has_saved_model or h5_files or keras_files:
                    model_info = {
                        "name": item.name,
                        "path": str(item),
                        "format": "SavedModel" if has_saved_model else "H5" if h5_files else "Keras",
                        "size_mb": self.get_directory_size(item) / (1024 * 1024)
                    }
                    
                    # Load metadata if available
                    metadata_path = item / "metadata.json"
                    if metadata_path.exists():
                        import json
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            model_info.update(metadata)
                        except Exception as e:
                            logger.error(f"Error reading metadata for {item.name}: {str(e)}")
                    
                    models.append(model_info)
            elif item.is_file() and item.suffix.lower() in ['.h5', '.keras']:
                # Handle individual .h5 or .keras files
                model_info = {
                    "name": item.stem,
                    "path": str(item),
                    "format": "H5" if item.suffix.lower() == '.h5' else "Keras",
                    "size_mb": item.stat().st_size / (1024 * 1024)
                }
                
                # Check for metadata (e.g., metadata.json or <model_name>_metadata.json)
                metadata_path = item.parent / f"{item.stem}_metadata.json"
                if not metadata_path.exists():
                    metadata_path = item.parent / "metadata.json"
                if metadata_path.exists():
                    import json
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        model_info.update(metadata)
                    except Exception as e:
                        logger.error(f"Error reading metadata for {item.name}: {str(e)}")
                
                models.append(model_info)
        
        logger.info(f"Found {len(models)} models in {self.models_path}")
        return models
    
    def get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files older than specified hours"""
        import time
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        for file_path in self.temp_path.rglob('*'):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        logger.info(f"Cleaned up temp file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error cleaning up {file_path}: {str(e)}")

# Global dataset manager instance
dataset_manager = DatasetManager()