import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries
from typing import Tuple, Optional

def generate_lime_overlay(
    model: torch.nn.Module,
    img_rgb: np.ndarray,  # Original image as RGB numpy array, values 0-255 or 0-1
    output_path: str,
    target_class_idx: int,
    num_samples: int = 100
) -> str:
    """Generates a LIME explanation image highlighting superpixels that support the prediction.
    
    Args:
        model: Loaded PyTorch model
        img_rgb: Input image in RGB format [H, W, 3], range 0-255 or 0-1
        output_path: Path to save the output LIME overlay image
        target_class_idx: Class index (0: Normal, 1: Pneumonia) to explain
        num_samples: Number of perturb samples (lower is faster, default 100)
        
    Returns:
        output_path: File path of the saved LIME overlay
    """
    # Resize image to model input size (224, 224) if needed
    h, w, c = img_rgb.shape
    img_224 = cv2.resize(img_rgb, (224, 224))
    
    # Ensure image values are in [0, 1] floats for LIME
    if img_224.dtype == np.uint8:
        img_float = img_224.astype(np.float32) / 255.0
    else:
        img_float = img_224.astype(np.float32)
        if np.max(img_float) > 1.0:
            img_float = img_float / 255.0

    # Define prediction function for LIME
    # LIME passes a batch of numpy images of shape (batch, 224, 224, 3) in range [0, 1]
    def batch_predict(numpy_images: np.ndarray) -> np.ndarray:
        model.eval()
        device = next(model.parameters()).device
        
        # Preprocess each image in the batch using the same normalization as training
        # Mean/Std standard normalization for ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        normalized_images = []
        for img in numpy_images:
            norm_img = (img - mean) / std
            # Transpose to channels-first (3, 224, 224)
            norm_img = norm_img.transpose((2, 0, 1))
            normalized_images.append(norm_img)
            
        # Stack into batch tensor
        batch_tensor = torch.from_numpy(np.stack(normalized_images)).to(device)
        
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = F.softmax(logits, dim=1)
            
        return probs.cpu().numpy()

    # Initialize LIME Explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Run the explanation
    explanation = explainer.explain_instance(
        img_float,
        batch_predict,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples
    )
    
    # Get image and mask for positive influence
    # temp is the original image, mask is a binary mask of positive superpixels
    temp, mask = explanation.get_image_and_mask(
        target_class_idx,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    # Generate boundary outline (highlighting positive superpixels in green)
    # skimage.segmentation.mark_boundaries returns a float image in [0, 1]
    lime_boundary = mark_boundaries(temp, mask, color=(0, 1, 0), outline_color=(0, 0.8, 0))
    
    # Resize back to original image size for display consistency
    lime_boundary_resized = cv2.resize(lime_boundary, (w, h))
    
    # Convert to [0, 255] uint8 RGB image
    lime_boundary_uint8 = np.uint8(lime_boundary_resized * 255)
    
    # Save image (convert to BGR for cv2)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(lime_boundary_uint8, cv2.COLOR_RGB2BGR))
    
    return output_path
