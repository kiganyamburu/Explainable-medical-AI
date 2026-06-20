import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) generator for PyTorch CNNs."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register forward and backward hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        
        # Backward hook syntax depends on PyTorch version
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
        else:
            self.backward_hook = self.target_layer.register_backward_hook(self._save_gradient)
            
    def _save_activation(self, module: nn.Module, input_tensor: Tuple[torch.Tensor, ...], output_tensor: torch.Tensor) -> None:
        self.activations = output_tensor
        
    def _save_gradient(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """Generates the normalized Grad-CAM heatmap for a given input tensor and class index."""
        self.model.eval()
        outputs = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()
            
        self.model.zero_grad()
        class_score = outputs[0, class_idx]
        class_score.backward()
        
        if self.gradients is None or self.activations is None:
            raise ValueError("Failed to capture gradients or activations. Verify hooks and model graph.")
            
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients (weights)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of forward activation maps
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # Apply ReLU to keep only features that positively influence the target class
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize between 0 and 1
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
            
        return heatmap, outputs

    def remove_hooks(self) -> None:
        """Removes the forward and backward hooks to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def generate_gradcam_overlay(
    model: nn.Module,
    img_tensor: torch.Tensor,
    original_img: np.ndarray,
    output_path: str,
    target_class_idx: Optional[int] = None
) -> Tuple[str, float]:
    """Coordinates Grad-CAM execution, overlays the heatmap onto the original image, and saves it.
    
    Args:
        model: Loaded PyTorch model
        img_tensor: Preprocessed image tensor [1, 3, 224, 224]
        original_img: Original image as numpy array [H, W, 3] in RGB
        output_path: Destination file path for the overlaid image
        target_class_idx: Target index to explain (0: Normal, 1: Pneumonia)
        
    Returns:
        output_path: File path where overlay is saved
        heatmap_max_val: Maximum weight value (measure of heatmap density)
    """
    # Assuming target layer is model.conv4 for our custom CNN
    if hasattr(model, 'conv4'):
        target_layer = model.conv4
    else:
        # Fallback to last conv layer if not matching PneumoniaCNN
        conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        if not conv_layers:
            raise ValueError("Model has no Conv2d layers.")
        target_layer = conv_layers[-1]

    grad_cam = GradCAM(model, target_layer)
    try:
        heatmap, logits = grad_cam.generate(img_tensor, class_idx=target_class_idx)
        
        # Post-process heatmap to original image dimensions
        h, w, _ = original_img.shape
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply Colormap JET
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend original image and heatmap (0.65 original, 0.35 heatmap)
        overlay = cv2.addWeighted(original_img, 0.65, heatmap_colored_rgb, 0.35, 0)
        
        # Save output in BGR for OpenCV
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
    finally:
        grad_cam.remove_hooks()
        
    return output_path, float(np.max(heatmap))
