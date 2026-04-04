"""
=============================================================================
CIFAR-10 CNN EXPERIMENT - LOCAL TRAINING SCRIPT
=============================================================================

For LOCAL execution only. For Colab, use train.ipynb instead.

Usage:
    cd experiments/001_cifar10_cnn
    python train.py

View results:
    cd "/path/to/My Drive/mlflow"
    mlflow ui --backend-store-uri file://./mlruns --port 5000
"""

import sys
from pathlib import Path

# =============================================================================
# LOCAL SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# Import shared utilities
from shared_config.paths import MLFLOW_TRACKING_URI, BRONZE, TRAINED, setup_mlflow, get_env_info
from src.models import SimpleCNN, get_pretrained_resnet
from src.data.transforms import get_cifar_transforms
from src.training.trainer import Trainer


# =============================================================================
# CONFIGURATION
# =============================================================================
def load_config(config_path: str = "config.yaml") -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# MODEL SELECTION
# =============================================================================
def get_model(config: dict) -> nn.Module:
    """Build model based on configuration."""
    architecture = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"].get("pretrained", True)
    
    if architecture == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes)
        print(f"Created SimpleCNN with {num_classes} classes")
    elif architecture == "resnet18":
        model = get_pretrained_resnet(
            model_name="resnet18",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=False
        )
        print(f"Created ResNet18 (pretrained={pretrained}) with {num_classes} classes")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    return model


# =============================================================================
# DATA LOADING
# =============================================================================
def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for CIFAR-10."""
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    data_dir = BRONZE / "cifar10"
    
    train_transform = get_cifar_transforms(train=True)
    val_transform = get_cifar_transforms(train=False)
    
    print(f"Loading CIFAR-10 from {data_dir}")
    
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")
    
    return train_loader, val_loader


# =============================================================================
# MAIN
# =============================================================================
def main():
    config = load_config()
    
    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']} (LOCAL)")
    print("=" * 60)
    
    # MLflow setup
    mlflow = setup_mlflow()
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Model & Data
    model = get_model(config).to(device)
    train_loader, val_loader = get_dataloaders(config)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        experiment_name=config["mlflow"]["experiment_name"]
    )

    save_dir = TRAINED / config["experiment"]["name"]

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        run_name=config["mlflow"].get("run_name"),
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()
