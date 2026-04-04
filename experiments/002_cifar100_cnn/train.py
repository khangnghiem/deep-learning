"""
=============================================================================
CIFAR-100 CNN EXPERIMENT - LOCAL TRAINING SCRIPT
=============================================================================

For LOCAL execution only. For Colab, use train.ipynb instead.

Usage:
    cd experiments/002_cifar100_cnn
    python train.py
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
from shared_config.paths import MLFLOW_TRACKING_URI, BRONZE, TRAINED, setup_mlflow
from src.models import SimpleCNN, get_pretrained_resnet
from src.data.transforms import get_cifar_transforms


# =============================================================================
# CONFIGURATION
# =============================================================================
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# MODEL SELECTION
# =============================================================================
def get_model(config: dict) -> nn.Module:
    """Build model based on configuration."""
    architecture = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]  # 100 for CIFAR-100
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
# DATA LOADING - CIFAR-100
# =============================================================================
def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for CIFAR-100."""
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    
    # CIFAR-100 data directory
    data_dir = BRONZE / "cifar100"
    
    # Use same transforms as CIFAR-10 (same image size)
    train_transform = get_cifar_transforms(train=True)
    val_transform = get_cifar_transforms(train=False)
    
    print(f"Loading CIFAR-100 from {data_dir}")
    
    # CIFAR-100 has 100 classes
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR100(
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
    print(f"Classes: 100 (fine-grained categories)")
    
    return train_loader, val_loader


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc="Training", leave=False, unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating", leave=False, unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), correct / total


# =============================================================================
# MAIN
# =============================================================================
def main():
    config = load_config()
    
    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']} (LOCAL)")
    print(f"Description: {config['experiment']['description']}")
    print("=" * 60)
    
    mlflow = setup_mlflow()
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = get_model(config).to(device)
    train_loader, val_loader = get_dataloaders(config)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    with mlflow.start_run(run_name=config["mlflow"].get("run_name")):
        mlflow.log_params({
            "model": config["model"]["architecture"],
            "epochs": config["training"]["epochs"],
            "lr": config["training"]["learning_rate"],
            "batch_size": config["data"]["batch_size"],
            "dataset": "cifar100",
            "num_classes": 100,
        })
        
        best_acc = 0
        
        for epoch in range(config["training"]["epochs"]):
            print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            }, step=epoch)
            
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = TRAINED / config["experiment"]["name"] / "best_model.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"✓ New best model saved! (acc: {best_acc:.4f})")
                mlflow.log_artifact(str(save_path))
        
        mlflow.log_metric("best_val_acc", best_acc)
        print(f"\n✅ Training complete! Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
