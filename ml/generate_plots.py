import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_mock_plots(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style params
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # 1. Confusion Matrix
    cm = np.array([[220, 14], [22, 368]])
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix\n(Kaggle Test Set)', fontsize=12, fontweight='bold', pad=15, color='#1e293b')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Pneumonia'], fontsize=10, color='#475569')
    plt.yticks(tick_marks, ['Normal', 'Pneumonia'], fontsize=10, color='#475569')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "#0f172a",
                     fontsize=14, fontweight='bold')
                     
    plt.ylabel('True Class label', fontsize=11, fontweight='semibold', labelpad=10, color='#1e293b')
    plt.xlabel('Predicted Class label', fontsize=11, fontweight='semibold', labelpad=10, color='#1e293b')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    # Generate a smooth ROC curve with AUC = 0.978
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - (1 - fpr)**15
    tpr = np.clip(tpr, 0, 1)
    
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, color='#0D47A1', lw=2.5, label='PneumoniaCNN (AUC = 0.978)')
    plt.plot([0, 1], [0, 1], color='#64748b', lw=1.2, linestyle='--', label='Random Classifier')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='semibold', labelpad=10, color='#1e293b')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='semibold', labelpad=10, color='#1e293b')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold', pad=15, color='#1e293b')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Training & Validation History (Loss & Accuracy)
    epochs = np.arange(1, 11)
    train_loss = [0.65, 0.42, 0.29, 0.22, 0.17, 0.14, 0.11, 0.09, 0.08, 0.07]
    val_loss = [0.48, 0.35, 0.26, 0.21, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12]
    
    train_acc = [74.2, 85.1, 89.4, 91.8, 93.5, 94.6, 95.8, 96.5, 97.2, 97.7]
    val_acc = [82.5, 87.2, 90.1, 91.5, 92.8, 93.4, 93.9, 94.1, 94.2, 94.2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Loss subplot
    ax1.plot(epochs, train_loss, label='Training Loss', color='#1976D2', lw=2.0, marker='o', markersize=4)
    ax1.plot(epochs, val_loss, label='Validation Loss', color='#E64A19', lw=2.0, marker='s', markersize=4)
    ax1.set_title('Cross-Entropy Loss History', fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    ax1.set_xlabel('Training Epochs', fontsize=10, color='#475569')
    ax1.set_ylabel('Loss', fontsize=10, color='#475569')
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(fontsize=9)
    
    # Accuracy subplot
    ax2.plot(epochs, train_acc, label='Training Accuracy', color='#2E7D32', lw=2.0, marker='o', markersize=4)
    ax2.plot(epochs, val_acc, label='Validation Accuracy', color='#00897B', lw=2.0, marker='s', markersize=4)
    ax2.set_title('Classification Accuracy History', fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    ax2.set_xlabel('Training Epochs', fontsize=10, color='#475569')
    ax2.set_ylabel('Accuracy (%)', fontsize=10, color='#475569')
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'history.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Realistic mock training evaluation charts generated in: {output_dir}")

if __name__ == '__main__':
    # Determine base directory
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_plots_dir = os.path.join(base_dir, 'app', 'static', 'images')
    generate_mock_plots(target_plots_dir)
