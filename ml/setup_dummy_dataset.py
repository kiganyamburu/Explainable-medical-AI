import os
import numpy as np
import cv2

def create_synthetic_xray(label: str) -> np.ndarray:
    """Generates a synthetic chest X-ray representation.
    Normal: Clear black lung fields with standard rib cage outlines.
    Pneumonia: Opacified lung fields with foggy white spots (consolidation).
    """
    # Create black canvas
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # 1. Chest outline / rib cage
    # Draw thoracic cavity boundary
    cv2.ellipse(img, (112, 110), (80, 100), 0, 0, 360, (60, 60, 60), -1)
    # Draw central mediastinum (heart/spine center column)
    cv2.ellipse(img, (112, 120), (25, 60), 0, 0, 360, (180, 180, 180), -1)
    
    # 2. Draw lung fields (two dark lobes)
    # Left Lung
    cv2.ellipse(img, (80, 110), (22, 70), 0, 0, 360, (15, 15, 15), -1)
    # Right Lung
    cv2.ellipse(img, (144, 110), (22, 70), 0, 0, 360, (15, 15, 15), -1)
    
    # 3. Draw rib bones (horizontal curved lines)
    for y in range(40, 190, 20):
        # Draw rib lines across the left and right lung lobes
        cv2.ellipse(img, (60, y), (40, 10), -10, 0, 180, (140, 140, 140), 2)
        cv2.ellipse(img, (164, y), (40, 10), 10, 0, 180, (140, 140, 140), 2)
        
    # 4. Apply disease markers if label is PNEUMONIA
    if label.upper() == 'PNEUMONIA':
        # Add hazy consolidation patches in lower lobes (white blurred circles)
        # Patch A
        cv2.circle(img, (75, 145), 20, (100, 100, 100), -1)
        # Patch B
        cv2.circle(img, (135, 150), 25, (120, 120, 120), -1)
        # Blur the image to simulate fluid density consolidation
        img = cv2.GaussianBlur(img, (15, 15), 0)
    else:
        # Normal lung: clear and sharp
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
    return img

def setup_dataset_structure(base_dir: str):
    splits = {
        'train': {'NORMAL': 16, 'PNEUMONIA': 16},
        'val': {'NORMAL': 4, 'PNEUMONIA': 4},
        'test': {'NORMAL': 4, 'PNEUMONIA': 4}
    }
    
    print(f"Setting up dummy chest X-ray dataset at: {os.path.abspath(base_dir)}")
    
    for split, classes in splits.items():
        for class_name, count in classes.items():
            class_path = os.path.join(base_dir, split, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            for idx in range(count):
                img = create_synthetic_xray(class_name)
                filename = f"synthetic_{split}_{class_name.lower()}_{idx}.jpeg"
                filepath = os.path.join(class_path, filename)
                cv2.imwrite(filepath, img)
                
    print("Dataset setup completed successfully!")

if __name__ == '__main__':
    # Determine default dataset dir matching active config settings
    import sys
    base_proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_proj_dir, 'dataset')
    setup_dataset_structure(dataset_dir)
