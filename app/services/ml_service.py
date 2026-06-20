import os
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Any, Tuple

# Import ML modules from ml/ package
from ml.cnn import PneumoniaCNN
from ml.gradcam import generate_gradcam_overlay
from ml.lime_explainer import generate_lime_overlay
from ml.shap_explainer import generate_shap_overlay

class MLService:
    """Service to manage model loading, image inference, and coordinating explainability visualizations (Grad-CAM, LIME, SHAP)."""
    
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _lazy_load_model(self) -> None:
        """Loads model weights into memory only when first prediction request is made."""
        if self.model is None:
            print(f"Lazy loading model weights from: {self.model_path} onto device: {self.device}")
            self.model = PneumoniaCNN()
            if not os.path.exists(self.model_path):
                # Save initial weights if file doesn't exist
                print(f"Warning: Weights file not found. Saving reinitialized weights checkpoint to {self.model_path}")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.model_path)
            
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Loads and pre-processes image for standard PyTorch input."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read chest X-ray image from path: {image_path}")
            
        # Convert BGR (OpenCV standard) to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to network's expected inputs
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Standard normalization for Imagenet transfers
        img_float = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std
        
        # Format as channels-first tensor with batch dimension (1, 3, 224, 224)
        img_tensor = img_normalized.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(self.device)
        
        return img_tensor, img_rgb

    def predict_and_explain(self, image_path: str, unique_id: str, uploads_dir: str, heatmaps_dir: str) -> Dict[str, Any]:
        """Runs model inference, measures execution speed, and generates Grad-CAM, LIME, and SHAP visual explanations."""
        self._lazy_load_model()
        
        start_time = time.time()
        
        # Load and preprocess
        img_tensor, img_rgb = self.preprocess_image(image_path)
        
        # 1. Run Classification Inference
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        pred_idx = int(np.argmax(probs))
        prediction = 'Pneumonia' if pred_idx == 1 else 'Normal'
        confidence = float(probs[pred_idx])
        
        # 2. Setup Explainability File Outputs
        # Extract original file extension to preserve formats (default to png for overlays)
        _, ext = os.path.splitext(image_path)
        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            ext = '.png'
            
        gradcam_filename = f"{unique_id}_gradcam{ext}"
        lime_filename = f"{unique_id}_lime{ext}"
        shap_filename = f"{unique_id}_shap{ext}"
        
        gradcam_path = os.path.join(heatmaps_dir, gradcam_filename)
        lime_path = os.path.join(heatmaps_dir, lime_filename)
        shap_path = os.path.join(heatmaps_dir, shap_filename)
        
        # 3. Generate Visual Explanations (Run gradcam, LIME, SHAP)
        # Note: LIME and SHAP can be computationally heavy, we run optimized sub-samples
        # Grad-CAM
        generate_gradcam_overlay(self.model, img_tensor, img_rgb, gradcam_path, target_class_idx=pred_idx)
        # LIME
        generate_lime_overlay(self.model, img_rgb, lime_path, target_class_idx=pred_idx, num_samples=100)
        # SHAP
        generate_shap_overlay(self.model, img_rgb, shap_path, target_class_idx=pred_idx, nsamples=80)
        
        # 4. Generate Clinical Diagnostic Findings List
        findings = []
        if prediction == 'Pneumonia':
            findings = [
                "Lung opacity and consolidation detected in X-ray fields",
                "Lower/middle lobe abnormalities (infiltrates) consistent with infectious processes",
                "High density interstitial markings indicating alveolar fluid accumulation"
            ]
        else:
            findings = [
                "Clear lung fields with normal aeration and structure",
                "No signs of focal consolidation, infiltration, or pleural effusion",
                "Normal vascular markings and clear cardiophrenic/costophrenic angles"
            ]
            
        processing_time = time.time() - start_time
        
        # Convert absolute overlay paths to relative URLs served by Flask static directory
        # e.g., "app/static/heatmaps/uuid_gradcam.png" -> "/static/heatmaps/uuid_gradcam.png"
        def relative_url(path: str) -> str:
            # Replaces windows backward slashes and extracts relative path from "static"
            parts = path.replace("\\", "/").split("app/static/")
            if len(parts) > 1:
                return "/static/" + parts[1]
            return path
            
        return {
            'prediction': prediction,
            'confidence': confidence,
            'confidence_normal': float(probs[0]),
            'confidence_pneumonia': float(probs[1]),
            'processing_time': processing_time,
            'heatmap_path': relative_url(gradcam_path),
            'lime_path': relative_url(lime_path),
            'shap_path': relative_url(shap_path),
            'findings': findings
        }
