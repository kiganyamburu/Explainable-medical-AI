import os
import sys
from typing import Tuple
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np

# Add parent directory to path to ensure proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.cnn import PneumoniaCNN

def preprocess_image(image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
    """Reads, resizes, and normalizes a chest X-ray image for PyTorch CNN inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from: {image_path}")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Normalize with standard ImageNet statistics
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std
    
    # Transpose to Channel-First (3, 224, 224) and add batch dim (1, 3, 224, 224)
    img_tensor = img_normalized.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)
    
    return img_tensor, img_rgb

def predict(image_path: str, model_path: str) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    # Instantiate and load model weights
    model = PneumoniaCNN()
    if not os.path.exists(model_path):
        print(f"Error: Model weights file not found at {model_path}")
        print("Please train a model first or check path configuration.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Preprocess image
    try:
        img_tensor, _ = preprocess_image(image_path)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)
        
    # Run prediction
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
    classes = ['Normal', 'Pneumonia']
    pred_idx = int(np.argmax(probs))
    prediction = classes[pred_idx]
    confidence = probs[pred_idx]
    
    print("\n==================================")
    print("      DIAGNOSTIC REPORT (CLI)     ")
    print("==================================")
    print(f"Image File   : {os.path.basename(image_path)}")
    print(f"Prediction   : {prediction}")
    print(f"Confidence   : {confidence * 100:.2f}%")
    print("----------------------------------")
    print("Prediction Probabilities:")
    print(f"  - Normal     : {probs[0] * 100:.2f}%")
    print(f"  - Pneumonia  : {probs[1] * 100:.2f}%")
    print("==================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pneumonia Detection CLI Inference")
    parser.add_argument('--image', type=str, required=True, help='Path to chest X-ray image')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to model weights checkpoint (.pth)')
    
    args = parser.parse_args()
    predict(args.image, args.model)
