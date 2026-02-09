"""
Medical-specific model architectures.

Includes models commonly used in medical imaging and genomics.

Usage:
    from src.models.medical import MedicalCNN, get_medical_resnet, GeneExpressionMLP
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class MedicalCNN(nn.Module):
    """
    Medical imaging CNN optimized for grayscale or RGB medical images.
    
    Includes:
    - Attention mechanisms
    - Deep supervision (optional)
    - Dropout for regularization
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,  # Grayscale by default
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_medical_resnet(
    model_name: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    in_channels: int = 1,
) -> nn.Module:
    """
    Get a ResNet adapted for medical imaging.
    
    Handles grayscale input and uses pretrained weights.
    
    Args:
        model_name: resnet18, resnet34, resnet50
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        in_channels: 1 for grayscale, 3 for RGB
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Modify first conv for grayscale
    if in_channels != 3:
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels, 64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize with mean of RGB weights
        if pretrained:
            with torch.no_grad():
                model.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
    
    # Modify classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


class GeneExpressionMLP(nn.Module):
    """
    MLP for gene expression / genomics data.
    
    Designed for high-dimensional inputs (thousands of genes).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] = [1024, 512, 256],
        dropout: float = 0.5,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class UNet(nn.Module):
    """
    U-Net for medical image segmentation.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.downs.append(self._double_conv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._double_conv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self._double_conv(feature * 2, feature))
        
        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _double_conv(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        
        return self.final_conv(x)


class AttentionGate(nn.Module):
    """Attention gate for U-Net."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
