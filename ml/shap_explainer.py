import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import shap
from skimage.segmentation import slic
from typing import Tuple, Optional

def generate_shap_overlay(
    model: torch.nn.Module,
    img_rgb: np.ndarray,  # Original image as RGB [H, W, 3]
    output_path: str,
    target_class_idx: int,
    nsamples: int = 80
) -> str:
    """Generates a SHAP explanation overlay using superpixel segmentations and KernelExplainer.
    
    Args:
        model: Loaded PyTorch model
        img_rgb: Input image in RGB format [H, W, 3]
        output_path: Path to save the SHAP overlay image
        target_class_idx: Class index (0: Normal, 1: Pneumonia) to explain
        nsamples: Number of perturbation samples for SHAP (default 80)
        
    Returns:
        output_path: File path of the saved SHAP overlay
    """
    # Resize image to model input size (224, 224)
    h, w, c = img_rgb.shape
    img_224 = cv2.resize(img_rgb, (224, 224))
    
    # Ensure float range [0, 1]
    if img_224.dtype == np.uint8:
        img_float = img_224.astype(np.float32) / 255.0
    else:
        img_float = img_224.astype(np.float32)
        if np.max(img_float) > 1.0:
            img_float = img_float / 255.0

    # 1. Segment image into superpixels (using SLIC)
    # 25-30 superpixels is a good trade-off between detail and computation speed
    num_superpixels = 25
    segments = slic(img_float, n_segments=num_superpixels, compactness=10, sigma=1)
    unique_segments = np.unique(segments)
    num_active_superpixels = len(unique_segments)

    # 2. Define prediction function for SHAP
    # SHAP will pass binary vector coalitions of shape (batch, num_active_superpixels)
    def model_predict_superpixels(coalitions: np.ndarray) -> np.ndarray:
        model.eval()
        device = next(model.parameters()).device
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        batch_images = []
        for coalition in coalitions:
            # Reconstruct image based on active/inactive superpixels
            temp_img = img_float.copy()
            # If coalition bit is 0, replace that superpixel with background (black)
            for idx, superpixel_id in enumerate(unique_segments):
                if coalition[idx] == 0:
                    temp_img[segments == superpixel_id] = 0.0
            
            # Normalize and transpose
            norm_img = (temp_img - mean) / std
            norm_img = norm_img.transpose((2, 0, 1))
            batch_images.append(norm_img)
            
        # Predict
        batch_tensor = torch.from_numpy(np.stack(batch_images)).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = F.softmax(logits, dim=1)
            
        return probs.cpu().numpy()

    # 3. Setup Kernel Explainer
    # Background dataset is all superpixels turned off (zeros)
    background = np.zeros((1, num_active_superpixels))
    explainer = shap.KernelExplainer(model_predict_superpixels, background)
    
    # 4. Compute Shapley values
    # Input is all superpixels active (ones)
    input_coalition = np.ones((1, num_active_superpixels))
    shap_values = explainer.shap_values(input_coalition, nsamples=nsamples)
    
    # 5. Extract Shapley values for the target class
    # shap_values can be a list (older version) or np.ndarray
    if isinstance(shap_values, list):
        target_shap = shap_values[target_class_idx][0]
    elif isinstance(shap_values, np.ndarray):
        # Shape could be (1, num_features, num_classes) or (num_classes, 1, num_features)
        # Let's write a robust parsing
        if shap_values.ndim == 3:
            # Usually (1, num_features, num_classes)
            if shap_values.shape[2] == 2:  # 2 classes
                target_shap = shap_values[0, :, target_class_idx]
            else:
                target_shap = shap_values[target_class_idx, 0, :]
        else:
            target_shap = shap_values[0]
    else:
        target_shap = np.array(shap_values)

    # 6. Map Shapley values back to pixels
    shap_heatmap = np.zeros_like(segments, dtype=np.float32)
    for idx, superpixel_id in enumerate(unique_segments):
        shap_heatmap[segments == superpixel_id] = target_shap[idx]
        
    # We only visualize positive contributions (features supporting target class)
    shap_heatmap = np.maximum(shap_heatmap, 0)
    
    # Normalize to [0, 255]
    h_max = np.max(shap_heatmap)
    if h_max > 0:
        shap_heatmap = (shap_heatmap / h_max) * 255
    shap_heatmap = np.uint8(shap_heatmap)
    
    # Resize map to original image dimensions
    shap_heatmap_resized = cv2.resize(shap_heatmap, (w, h))
    
    # Apply colormap (JET) and blend
    heatmap_colored = cv2.applyColorMap(shap_heatmap_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    overlay = cv2.addWeighted(img_rgb, 0.70, heatmap_rgb, 0.30, 0)
    
    # Save output (BGR for cv2)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    return output_path
