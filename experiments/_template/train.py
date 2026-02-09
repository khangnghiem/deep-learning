"""
=============================================================================
EXPERIMENT TEMPLATE - LOCAL TRAINING SCRIPT
=============================================================================

For LOCAL execution only. For Colab, use train.ipynb instead.

Usage:
    cd experiments/{EXPERIMENT_NAME}
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
from tqdm import tqdm

# Import shared utilities
from shared_config.paths import MLFLOW_TRACKING_URI, BRONZE, TRAINED, setup_mlflow
# TODO: Import your model and transforms
# from src.models import YourModel
# from src.data.transforms import get_transforms


# =============================================================================
# CONFIGURATION
# =============================================================================
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# MODEL (TODO: Implement)
# =============================================================================
def get_model(config: dict) -> nn.Module:
    """Build model based on configuration."""
    # TODO: Implement model creation
    raise NotImplementedError("Implement get_model()")


# =============================================================================
# DATA LOADING (TODO: Implement)
# =============================================================================
def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    # TODO: Implement data loading
    raise NotImplementedError("Implement get_dataloaders()")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc="Training"):
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
        for inputs, targets in tqdm(loader, desc="Validating"):
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
    print("=" * 60)
    
    mlflow = setup_mlflow()
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
                print(f"✓ New best! (acc: {best_acc:.4f})")
                mlflow.log_artifact(str(save_path))
        
        mlflow.log_metric("best_val_acc", best_acc)
        print(f"\n✅ Training complete! Best: {best_acc:.4f}")


if __name__ == "__main__":
    main()
