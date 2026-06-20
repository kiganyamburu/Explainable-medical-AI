import os
import sys
import glob
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for server environments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_curve, auc, roc_auc_score
)

# Import our CNN architecture
from cnn import PneumoniaCNN

class ChestXRayDataset(Dataset):
    """Custom Dataset for loading chest X-ray images with optional augmentations."""
    
    def __init__(self, dir_path: str, transform=None) -> None:
        self.file_paths = []
        self.labels = []
        self.transform = transform
        
        # Class indexing: NORMAL = 0, PNEUMONIA = 1
        self.classes = ['NORMAL', 'PNEUMONIA']
        
        # Check standard and nested directories
        search_paths = [
            dir_path,
            os.path.join(dir_path, 'chest_xray'),
            os.path.join(dir_path, '..', 'chest_xray', os.path.basename(dir_path))
        ]
        
        resolved_path = None
        for path in search_paths:
            if os.path.exists(path):
                # Check if at least one class directory exists
                if os.path.exists(os.path.join(path, 'NORMAL')) or os.path.exists(os.path.join(path, 'PNEUMONIA')):
                    resolved_path = path
                    break
                    
        if not resolved_path:
            print(f"Warning: Could not find valid dataset path for input: {dir_path}")
            return
            
        for label_idx, label_name in enumerate(self.classes):
            class_dir = os.path.join(resolved_path, label_name)
            if os.path.exists(class_dir):
                for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'):
                    self.file_paths.extend(glob.glob(os.path.join(class_dir, ext)))
                # Update labels list to match found paths
                self.labels = [label_idx] * len(self.file_paths)
                
        # Sync file paths and labels
        self.file_paths = []
        self.labels = []
        for label_idx, label_name in enumerate(self.classes):
            class_dir = os.path.join(resolved_path, label_name)
            if os.path.exists(class_dir):
                paths = []
                for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'):
                    paths.extend(glob.glob(os.path.join(class_dir, ext)))
                self.file_paths.extend(paths)
                self.labels.extend([label_idx] * len(paths))
                
        print(f"Loaded {len(self.file_paths)} images from {resolved_path}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for corrupted images (black image)
            img = Image.new('RGB', (224, 224), color=0)
            print(f"Error reading image {img_path}: {e}. Loaded fallback blank image.")
            
        if self.transform:
            img_tensor = self.transform(img)
        else:
            # Basic fallback transform
            fallback_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = fallback_transform(img)
            
        return img_tensor, label

def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device):
    """Evaluates the model on test/validation sets and computes key healthcare metrics."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # Probability of pneumonia class
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc_score = 0.0
        
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, precision, recall, f1, auc_score, cm, all_labels, all_probs

def plot_and_save_metrics(
    train_losses: list, val_losses: list, 
    train_accs: list, val_accs: list,
    labels: np.ndarray, probs: np.ndarray,
    cm: np.ndarray, output_dir: str
) -> None:
    """Generates and saves performance evaluation graphs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot Training History (Loss & Accuracy)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='#0056b3', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='#dc3545', linewidth=2)
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', color='#28a745', linewidth=2)
    plt.plot(val_accs, label='Val Acc', color='#ffc107', linewidth=2)
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'history.png'), dpi=150)
    plt.close()
    
    # 2. Plot ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#28a745', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#6c757d', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    
    # 3. Plot Confusion Matrix
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Pneumonia'])
    plt.yticks(tick_marks, ['Normal', 'Pneumonia'])
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    print(f"Metrics plots successfully saved to {output_dir}")

def train(args) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training pipeline on device: {device}")
    
    # Define transformations
    # Train: Augmentation (flips, random rotation) + Resizing + Normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Val/Test: Standard resizing + Normalization
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize datasets
    train_dataset = ChestXRayDataset(args.train_dir, transform=train_transform)
    val_dataset = ChestXRayDataset(args.val_dir, transform=val_transform)
    test_dataset = ChestXRayDataset(args.test_dir, transform=val_transform)
    
    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Verify folder structures.")
        sys.exit(1)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if len(test_dataset) > 0 else None
    
    # Initialize Model
    model = PneumoniaCNN().to(device)
    
    # Resume from existing weights if specified
    if os.path.exists(args.model_path) and args.resume:
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Resumed weights checkpoint from {args.model_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Training from scratch.")
            
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning Rate Scheduling: Reduces learning rate if loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Early Stopping Variables
    best_val_loss = float('inf')
    early_stop_patience = args.patience
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Core loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total_train
        epoch_acc = 100.0 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation epoch
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()
                    
            epoch_val_loss = val_running_loss / total_val
            epoch_val_acc = 100.0 * correct_val / total_val
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)
            
            scheduler.step(epoch_val_loss)
            
            # Print statistics
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% | Time: {elapsed:.1f}s")
            
            # Checkpoint Model (Save best)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), args.model_path)
                epochs_no_improve = 0
                print(f"--> Saved best model checkpoint to {args.model_path} with Val Loss {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                
            # Early Stopping check
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered! Training stopped. Best Val Loss: {best_val_loss:.4f}")
                break
        else:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Time: {elapsed:.1f}s")
            torch.save(model.state_dict(), args.model_path)
            
    # Evaluation on Test Dataset
    # Load the absolute best saved weights before testing
    if val_loader and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Loaded best weights for final evaluation.")
        
    eval_loader = test_loader if test_loader else val_loader
    if eval_loader:
        print("\n=== Running Final Model Evaluation ===")
        test_acc, test_prec, test_rec, test_f1, test_auc, test_cm, labels, probs = evaluate_model(model, eval_loader, device)
        print(f"Test Accuracy : {test_acc * 100:.2f}%")
        print(f"Precision     : {test_prec * 100:.2f}%")
        print(f"Recall        : {test_rec * 100:.2f}%")
        print(f"F1 Score      : {test_f1 * 100:.2f}%")
        print(f"ROC AUC       : {test_auc:.4f}")
        print("Confusion Matrix:")
        print(test_cm)
        
        # Save plots to static/images so the admin tab can easily display them
        plot_and_save_metrics(
            train_losses, val_losses if val_loader else train_losses,
            train_accs, val_accs if val_loader else train_accs,
            labels, probs, test_cm, args.plots_dir
        )
    else:
        print("No validation or test loader available for final evaluation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection CNN in PyTorch")
    
    # Dataset Directories
    parser.add_argument('--train-dir', type=str, default='../chest_xray/train', help='Path to training data')
    parser.add_argument('--val-dir', type=str, default='../chest_xray/val', help='Path to validation data')
    parser.add_argument('--test-dir', type=str, default='../chest_xray/test', help='Path to test data')
    
    # Optimization Hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=4, help='Early stopping patience')
    
    # Save & Resume Paths
    parser.add_argument('--model-path', type=str, default='model.pth', help='Path to save weights')
    parser.add_argument('--resume', action='store_true', help='Resume training from existing weights')
    parser.add_argument('--plots-dir', type=str, default='../app/static/images', help='Directory to save evaluation plots')
    
    args = parser.parse_args()
    train(args)
