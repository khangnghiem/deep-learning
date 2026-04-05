"""
Reusable model architectures.

Usage:
    from src.models import SimpleCNN, get_pretrained_resnet
    
    model = SimpleCNN(num_classes=10)
    model = get_pretrained_resnet("resnet18", num_classes=100)
"""

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 style images."""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class MLP(nn.Module):
    """Simple MLP for tabular data or flattened images."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128],
        num_classes: int = 10,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        return self.network(x)


def get_pretrained_resnet(
    model_name: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Get a pretrained ResNet with custom classifier.
    
    Args:
        model_name: One of "resnet18", "resnet34", "resnet50", "resnet101"
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze feature extraction layers
    
    Returns:
        ResNet model with modified classifier
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
    elif model_name == "resnet101":
        model = models.resnet101(weights=weights)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def get_pretrained_vit(
    model_name: str = "vit_b_16",
    num_classes: int = 10,
    pretrained: bool = True
) -> nn.Module:
    """
    Get a pretrained Vision Transformer.
    
    Args:
        model_name: One of "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        ViT model with modified head
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    
    if model_name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
    elif model_name == "vit_b_32":
        model = models.vit_b_32(weights=weights)
    elif model_name == "vit_l_16":
        model = models.vit_l_16(weights=weights)
    elif model_name == "vit_l_32":
        model = models.vit_l_32(weights=weights)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Replace head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model
