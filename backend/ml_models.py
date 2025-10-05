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
from datetime import datetime  # Added import

logger = logging.getLogger(__name__)

class WatermelonClassifier:
    """Enhanced watermelon variety and ripeness classifier"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.class_names = []
        self.img_size = (224, 224)
        self.confidence_threshold = 0.65  # Slightly increased for better filtering
        
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
        """Enhanced image preprocessing with additional normalization"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image with antialiasing
            image = image.resize(self.img_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Normalize to [-1, 1] range (matches MobileNetV2 preprocessing)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
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
            
            # Enhanced Green Coverage with broader range
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            green_coverage = np.sum(green_mask > 0) / (img_cv.shape[0] * img_cv.shape[1])
            
            # Enhanced Color Saturation with weighted channels
            saturation = np.mean(hsv[:, :, 1]) / 255.0 * 0.6 + np.mean(hsv[:, :, 2]) / 255.0 * 0.4
            
            # Improved Stripe Pattern detection
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            stripe_pattern = np.sum(edges > 0) / (img_cv.shape[0] * img_cv.shape[1])
            
            # Enhanced Ground Spot detection
            lower_yellow = np.array([20, 80, 80])
            upper_yellow = np.array([35, 255, 255])
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
        """Enhanced shape analysis with additional metrics"""
        try:
            image = Image.open(io.BytesIO(image_data))
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Adaptive thresholding for better contour detection
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'shape_score': 0.0, 'roundness': 0.0, 'aspect_ratio': 0.0}
            
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter == 0:
                return {'shape_score': 0.0, 'roundness': 0.0, 'aspect_ratio': 0.0}
            
            # Roundness calculation
            roundness = (4 * np.pi * area) / (perimeter * perimeter)
            
            # Shape score with improved contour approximation
            epsilon = 0.015 * perimeter
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            shape_score = 1.0 / max(len(approx), 1)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 0.0
            
            return {
                'shape_score': float(min(shape_score, 1.0)),
                'roundness': float(min(roundness, 1.0)),
                'aspect_ratio': float(min(max(aspect_ratio, 0.0), 2.0))
            }
            
        except Exception as e:
            logger.error(f"Error extracting shape features: {str(e)}")
            return {'shape_score': 0.0, 'roundness': 0.0, 'aspect_ratio': 0.0}
    
    def extract_surface_features(self, image_data: bytes) -> Dict[str, float]:
        """Enhanced surface analysis with texture metrics"""
        try:
            image = Image.open(io.BytesIO(image_data))
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Improved Matte Appearance calculation
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            mean = cv2.filter2D(blurred.astype(np.float32), -1, np.ones((5, 5), np.float32) / 25)
            sqr_mean = cv2.filter2D((blurred.astype(np.float32))**2, -1, np.ones((5, 5), np.float32) / 25)
            variance = sqr_mean - mean**2
            std_dev = np.sqrt(np.maximum(variance, 0))
            matte_score = 1.0 - (np.mean(std_dev) / 255.0)
            
            # Enhanced Color Uniformity
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            h_std = np.std(hsv[:, :, 0]) / 180.0
            s_std = np.std(hsv[:, :, 1]) / 255.0
            v_std = np.std(hsv[:, :, 2]) / 255.0
            color_uniformity = 1.0 - ((h_std + s_std + v_std) / 3.0)
            
            # Add texture analysis
            glcm = self.calculate_glcm(gray)
            texture_contrast = np.mean(glcm)
            
            return {
                'matte_appearance': float(max(matte_score, 0.0)),
                'color_uniformity': float(max(color_uniformity, 0.0)),
                'texture_contrast': float(max(min(texture_contrast, 1.0), 0.0))
            }
            
        except Exception as e:
            logger.error(f"Error extracting surface features: {str(e)}")
            return {'matte_appearance': 0.0, 'color_uniformity': 0.0, 'texture_contrast': 0.0}
    
    def calculate_glcm(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate Gray-Level Co-occurrence Matrix for texture analysis"""
        try:
            from skimage.feature import graycomatrix, graycoprops
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray_image, distances=distances, angles=angles, 
                              levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')
            return contrast
        except Exception as e:
            logger.error(f"Error calculating GLCM: {str(e)}")
            return np.zeros((1, len([0, np.pi/4, np.pi/2, 3*np.pi/4])))
    
    def predict(self, image_data: bytes) -> Dict:
        """Enhanced prediction with feature fusion"""
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
            
            # Enhanced confidence adjustment based on features
            feature_confidence = (
                visual_features['color_saturation'] * 0.3 +
                visual_features['stripe_pattern'] * 0.3 +
                shape_features['roundness'] * 0.2 +
                surface_features['color_uniformity'] * 0.2
            )
            adjusted_confidence = (max_confidence * 0.7 + feature_confidence * 0.3)
            
            # Apply confidence threshold
            if adjusted_confidence < self.confidence_threshold:
                predicted_class = "not_valid"
                final_confidence = adjusted_confidence
            else:
                predicted_class = self.class_names[predicted_class_idx] if self.class_names else self.default_classes[predicted_class_idx]
                final_confidence = adjusted_confidence
            
            # Parse variety and ripeness
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
        """Enhanced MobileNetV2-based model with batch normalization"""
        # Base model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.img_size + (3,),
            include_top=False,
            weights="imagenet"
        )
        
        # Fine-tune last few layers
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 20
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Enhanced model architecture
        inputs = tf.keras.Input(shape=self.img_size + (3,))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        predictions = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        
        # Compile model with learning rate schedule
        initial_learning_rate = 1e-4
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def train_model(self, train_data_path: str, val_data_path: str, epochs: int = 15) -> Dict:
        """Enhanced training with data augmentation and early stopping"""
        try:
            # Enhanced data augmentation
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomContrast(0.2),
                tf.keras.layers.RandomBrightness(0.2),
            ])
            
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
            
            # Get class names before applying augmentation
            self.class_names = train_ds.class_names
            num_classes = len(self.class_names)
            
            # Apply augmentation to training data
            train_ds = train_ds.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Calculate class weights
            class_weights = self.calculate_class_weights(train_data_path)
            
            # Optimize datasets
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
            val_ds = val_ds.cache().prefetch(AUTOTUNE)
            
            # Build model
            self.model = self.build_model(num_classes)
            
            # Callbacks for training
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            # Generate training report
            training_report = {
                "epochs": len(history.history['loss']),
                "class_names": self.class_names,
                "num_classes": num_classes,
                "final_train_accuracy": float(history.history['accuracy'][-1]),
                "final_val_accuracy": float(history.history['val_accuracy'][-1]),
                "final_train_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1]),
                "class_weights": class_weights,
                "best_val_accuracy": float(max(history.history['val_accuracy']))
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
            class_weights[i] = float(min(weight, 5.0))  # Cap weights to prevent extreme values
        
        return class_weights
    
    def evaluate_model(self, test_data_path: str) -> Dict:
        """Enhanced evaluation with detailed metrics"""
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
                    if "not_valid" in self.class_names:
                        final_preds.append(self.class_names.index("not_valid"))
                    else:
                        final_preds.append(y_pred[i])
                else:
                    final_preds.append(y_pred[i])
            
            # Generate detailed classification report
            report = classification_report(
                y_true, 
                final_preds, 
                target_names=self.class_names,
                output_dict=True
            )
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, final_preds)
            
            # Calculate per-class accuracy
            per_class_accuracy = {}
            for i, class_name in enumerate(self.class_names):
                class_mask = (y_true == i)
                if np.sum(class_mask) > 0:
                    per_class_accuracy[class_name] = np.mean(y_pred[class_mask] == i)
            
            return {
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "accuracy": float(report['accuracy']),
                "num_test_samples": len(y_true),
                "per_class_accuracy": per_class_accuracy
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise ValueError(f"Evaluation failed: {str(e)}")
    
    def save_model(self, model_path: str) -> bool:
        """Save the trained model with enhanced metadata"""
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            
            metadata = {
                "class_names": self.class_names,
                "img_size": self.img_size,
                "confidence_threshold": self.confidence_threshold,
                "model_architecture": str(self.model.to_json()),
                "training_timestamp": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise ValueError(f"Failed to save model: {str(e)}")

    def load_model(self, model_path: str) -> bool:
        """Load a trained model"""
        try:
            model_path = Path(model_path)
            if model_path.is_file() and model_path.suffix.lower() in ['.h5', '.keras']:
                self.model = tf.keras.models.load_model(str(model_path))
            else:
                self.model = tf.keras.models.load_model(str(model_path))
            
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if not metadata_path.exists():
                metadata_path = model_path.parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.class_names = metadata.get("class_names", self.default_classes)
                self.img_size = tuple(metadata.get("img_size", (224, 224)))
                self.confidence_threshold = metadata.get("confidence_threshold", 0.65)
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")

    def export_tflite(self, model_path: str, tflite_path: str) -> bool:
        """Export model to TensorFlow Lite with quantization"""
        try:
            if self.model is None:
                self.load_model(model_path)
            
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
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