import torch
import torch.nn as nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    """Custom Convolutional Neural Network for binary classification of chest X-ray images (Normal vs. Pneumonia)."""
    
    def __init__(self) -> None:
        super(PneumoniaCNN, self).__init__()
        
        # Convolutional Block 1
        # Input: 3 x 224 x 224
        # Output: 16 x 112 x 112 after pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        
        # Convolutional Block 2
        # Input: 16 x 112 x 112
        # Output: 32 x 56 x 56 after pooling
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        
        # Convolutional Block 3
        # Input: 32 x 56 x 56
        # Output: 64 x 28 x 28 after pooling
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        
        # Convolutional Block 4
        # Input: 64 x 28 x 28
        # Output: 128 x 14 x 14 after pooling
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        
        # Max Pooling (reduces height and width by half)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        # Input dimensions: 128 channels * 14 * 14 spatial height/width
        self.fc1 = nn.Linear(in_features=128 * 14 * 14, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=2)  # 2 classes: Normal (0), Pneumonia (1)
        
        # Regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten the feature map
        x = x.reshape(-1, 128 * 14 * 14)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
