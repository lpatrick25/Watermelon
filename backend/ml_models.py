import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

class WatermelonClassifier:
    """Combined watermelon variety and ripeness classifier"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.class_names = []
        self.img_size = (224, 224)
        self.confidence_threshold = 0.6  # Lowered from 0.7
        
        # Define combined classes: variety_ripeness
        self.default_classes = [
            "crimsonsweet_ripe",
            "crimsonsweet_unripe", 
            "other_variety",
            "not_valid"
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for model input"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.img_size)
            
            # Convert to numpy array and normalize
            img_array = np.array(image) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Invalid image format: {str(e)}")
    
    def extract_visual_features(self, image_data: bytes) -> Dict[str, float]:
        """Extract visual cues: Green Coverage, Color Saturation, Stripe Pattern, Ground Spot"""
        try:
            image = Image.open(io.BytesIO(image_data))
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Green Coverage
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            green_coverage = np.sum(green_mask > 0) / (img_cv.shape[0] * img_cv.shape[1])
            
            # Color Saturation (average saturation)
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            
            # Stripe Pattern (edge detection strength)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            stripe_pattern = np.sum(edges > 0) / (img_cv.shape[0] * img_cv.shape[1])
            
            # Ground Spot (detect light/yellow regions)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            ground_spot = np.sum(yellow_mask > 0) / (img_cv.shape[0] * img_cv.shape[1])
            
            return {
                'green_coverage': float(green_coverage),
                'color_saturation': float(saturation),
                'stripe_pattern': float(stripe_pattern),
                'ground_spot': float(ground_spot)
            }
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {str(e)}")
            return {
                'green_coverage': 0.0,
                'color_saturation': 0.0,
                'stripe_pattern': 0.0,
                'ground_spot': 0.0
            }
    
    def extract_shape_features(self, image_data: bytes) -> Dict[str, float]:
        """Extract shape analysis: Shape Score, Roundness"""
        try:
            image = Image.open(io.BytesIO(image_data))
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Find contours
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'shape_score': 0.0, 'roundness': 0.0}
            
            # Get largest contour (assume it's the watermelon)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter == 0:
                return {'shape_score': 0.0, 'roundness': 0.0}
            
            # Roundness = 4π * area / perimeter²
            roundness = (4 * np.pi * area) / (perimeter * perimeter)
            
            # Shape score based on contour approximation
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            shape_score = 1.0 / max(len(approx), 1)  # More vertices = less regular shape
            
            return {
                'shape_score': float(min(shape_score, 1.0)),
                'roundness': float(min(roundness, 1.0))
            }
            
        except Exception as e:
            logger.error(f"Error extracting shape features: {str(e)}")
            return {'shape_score': 0.0, 'roundness': 0.0}
    
    def extract_surface_features(self, image_data: bytes) -> Dict[str, float]:
        """Extract surface analysis: Matte Appearance, Color Uniformity"""
        try:
            image = Image.open(io.BytesIO(image_data))
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Matte Appearance (inverse of reflection/shininess)
            # Calculate local standard deviation to detect shine
            kernel = np.ones((5, 5), np.float32) / 25
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
            variance = sqr_mean - mean**2
            std_dev = np.sqrt(np.maximum(variance, 0))
            matte_score = 1.0 - (np.mean(std_dev) / 255.0)
            
            # Color Uniformity
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            h_std = np.std(hsv[:, :, 0])
            s_std = np.std(hsv[:, :, 1])
            v_std = np.std(hsv[:, :, 2])
            
            # Normalize and invert (lower std = higher uniformity)
            color_uniformity = 1.0 - ((h_std + s_std/255.0 + v_std/255.0) / 3.0)
            
            return {
                'matte_appearance': float(max(matte_score, 0.0)),
                'color_uniformity': float(max(color_uniformity, 0.0))
            }
            
        except Exception as e:
            logger.error(f"Error extracting surface features: {str(e)}")
            return {'matte_appearance': 0.0, 'color_uniformity': 0.0}
    
    def predict(self, image_data: bytes) -> Dict:
        """Predict watermelon variety and ripeness"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            confidence_scores = predictions[0]
            
            # Extract all features
            visual_features = self.extract_visual_features(image_data)
            shape_features = self.extract_shape_features(image_data)
            surface_features = self.extract_surface_features(image_data)
            
            # Get predicted class
            predicted_class_idx = np.argmax(confidence_scores)
            max_confidence = float(np.max(confidence_scores))
            
            # Apply confidence threshold
            if max_confidence < self.confidence_threshold:
                predicted_class = "not_valid"
                final_confidence = max_confidence
            else:
                predicted_class = self.class_names[predicted_class_idx] if self.class_names else self.default_classes[predicted_class_idx]
                final_confidence = max_confidence
            
            # Parse variety and ripeness from class name
            variety, ripeness = self.parse_class_name(predicted_class)
            
            # Create confidence breakdown
            confidence_breakdown = {}
            for i, class_name in enumerate(self.class_names or self.default_classes):
                confidence_breakdown[class_name] = float(confidence_scores[i])
            
            return {
                "variety": variety,
                "ripeness": ripeness,
                "predicted_class": predicted_class,
                "confidence": final_confidence,
                "confidence_breakdown": confidence_breakdown,
                "visual_analysis": visual_features,
                "shape_analysis": shape_features,
                "surface_analysis": surface_features,
                "is_valid": predicted_class != "not_valid",
                "is_crimsonsweet": variety == "crimsonsweet"
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def parse_class_name(self, class_name: str) -> Tuple[str, str]:
        """Parse variety and ripeness from class name"""
        if class_name == "not_valid":
            return "unknown", "unknown"
        elif class_name == "other_variety":
            return "other", "unknown"
        elif "_" in class_name:
            parts = class_name.split("_")
            variety = parts[0]
            ripeness = parts[1] if len(parts) > 1 else "unknown"
            return variety, ripeness
        else:
            return "unknown", "unknown"
    
    def build_model(self, num_classes: int = 4) -> tf.keras.Model:
        """Build MobileNetV2-based model"""
        # Base model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.img_size + (3,),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False
        
        # Add custom classification head
        global_avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        dropout = tf.keras.layers.Dropout(0.3)(global_avg)
        predictions = tf.keras.layers.Dense(num_classes, activation="softmax")(dropout)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def train_model(self, train_data_path: str, val_data_path: str, epochs: int = 15) -> Dict:
        """Train the watermelon classifier model"""
        try:
            # Create data pipelines
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_data_path,
                image_size=self.img_size,
                batch_size=32,
                label_mode='int'
            )
            
            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_data_path,
                image_size=self.img_size,
                batch_size=32,
                label_mode='int'
            )
            
            # Get class names
            self.class_names = train_ds.class_names
            num_classes = len(self.class_names)
            
            # Build model
            self.model = self.build_model(num_classes)
            
            # Calculate class weights for imbalanced data
            class_weights = self.calculate_class_weights(train_data_path)
            
            # Optimize datasets
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.prefetch(AUTOTUNE)
            val_ds = val_ds.prefetch(AUTOTUNE)
            
            # Train model
            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                class_weight=class_weights,
                verbose=1
            )
            
            # Generate training report
            training_report = {
                "epochs": epochs,
                "class_names": self.class_names,
                "num_classes": num_classes,
                "final_train_accuracy": float(history.history['accuracy'][-1]),
                "final_val_accuracy": float(history.history['val_accuracy'][-1]),
                "final_train_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1]),
                "class_weights": class_weights
            }
            
            return training_report
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise ValueError(f"Training failed: {str(e)}")
    
    def calculate_class_weights(self, data_path: str) -> Dict[int, float]:
        """Calculate class weights for imbalanced dataset"""
        class_counts = {}
        for class_name in os.listdir(data_path):
            class_dir = os.path.join(data_path, class_name)
            if os.path.isdir(class_dir):
                class_counts[class_name] = len(os.listdir(class_dir))
        
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        class_weights = {}
        for i, (class_name, count) in enumerate(class_counts.items()):
            weight = total_samples / (num_classes * count)
            class_weights[i] = weight
        
        return class_weights
    
    def evaluate_model(self, test_data_path: str) -> Dict:
        """Evaluate model performance on test set"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        try:
            # Load test data
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_data_path,
                image_size=self.img_size,
                batch_size=32,
                shuffle=False,
                label_mode='int'
            )
            
            # Make predictions
            y_true = np.concatenate([y for x, y in test_ds], axis=0)
            y_pred_probs = self.model.predict(test_ds, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Apply confidence threshold
            final_preds = []
            for i, probs in enumerate(y_pred_probs):
                max_prob = np.max(probs)
                if max_prob < self.confidence_threshold:
                    # Check if "not_valid" class exists
                    if "not_valid" in self.class_names:
                        final_preds.append(self.class_names.index("not_valid"))
                    else:
                        final_preds.append(y_pred[i])  # Keep original prediction
                else:
                    final_preds.append(y_pred[i])
            
            # Generate classification report
            report = classification_report(
                y_true, 
                final_preds, 
                target_names=self.class_names,
                output_dict=True
            )
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, final_preds)
            
            return {
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "accuracy": report['accuracy'],
                "num_test_samples": len(y_true)
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise ValueError(f"Evaluation failed: {str(e)}")
    
    def save_model(self, model_path: str) -> bool:
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            self.model.save(model_path)
            
            # Save class names and metadata
            metadata = {
                "class_names": self.class_names,
                "img_size": self.img_size,
                "confidence_threshold": self.confidence_threshold
            }
            
            metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise ValueError(f"Failed to save model: {str(e)}")
    
    # def load_model(self, model_path: str) -> bool:
    #     """Load a trained model"""
    #     try:
    #         # Load model
    #         self.model = tf.keras.models.load_model(model_path)
            
    #         # Load metadata
    #         metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
    #         if os.path.exists(metadata_path):
    #             with open(metadata_path, 'r') as f:
    #                 metadata = json.load(f)
                
    #             self.class_names = metadata.get("class_names", self.default_classes)
    #             self.img_size = tuple(metadata.get("img_size", (224, 224)))
    #             self.confidence_threshold = metadata.get("confidence_threshold", 0.7)
            
    #         logger.info(f"Model loaded from {model_path}")
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Error loading model: {str(e)}")
    #         raise ValueError(f"Failed to load model: {str(e)}")
    def load_model(self, model_path: str) -> bool:
        """Load a trained model"""
        try:
            model_path = Path(model_path)
            # Check if model_path is a file (.h5/.keras) or a directory (SavedModel)
            if model_path.is_file() and model_path.suffix.lower() in ['.h5', '.keras']:
                # Load .h5 or .keras file
                self.model = tf.keras.models.load_model(str(model_path))
            else:
                # Assume directory with SavedModel
                self.model = tf.keras.models.load_model(str(model_path))
            
            # Load metadata
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if not metadata_path.exists():
                metadata_path = model_path.parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.class_names = metadata.get("class_names", self.default_classes)
                self.img_size = tuple(metadata.get("img_size", (224, 224)))
                self.confidence_threshold = metadata.get("confidence_threshold", 0.7)
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")
    
    # def export_tflite(self, model_path: str, tflite_path: str) -> bool:
    #     """Export model to TensorFlow Lite format"""
    #     try:
    #         # Load the SavedModel if not already loaded
    #         if self.model is None:
    #             self.load_model(model_path)
            
    #         # Convert to TFLite
    #         converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
    #         tflite_model = converter.convert()
            
    #         # Save TFLite model
    #         with open(tflite_path, "wb") as f:
    #             f.write(tflite_model)
            
    #         logger.info(f"TFLite model exported to {tflite_path}")
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Error exporting to TFLite: {str(e)}")
    #         raise ValueError(f"Failed to export TFLite model: {str(e)}")
    def export_tflite(self, model_path: str, tflite_path: str) -> bool:
        """Export model to TensorFlow Lite format"""
        try:
            # Load the model if not already loaded
            if self.model is None:
                self.load_model(model_path)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            
            # Save TFLite model
            os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            
            logger.info(f"TFLite model exported to {tflite_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to TFLite: {str(e)}")
            raise ValueError(f"Failed to export TFLite model: {str(e)}")

# Global model instance
watermelon_classifier = WatermelonClassifier()