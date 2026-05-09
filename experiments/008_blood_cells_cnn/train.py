"""
=============================================================================
BLOOD CELLS CNN EXPERIMENT - LOCAL TRAINING SCRIPT
=============================================================================

For LOCAL execution only. For Colab, use 008_blood_cells_cnn_train.ipynb instead.

Usage:
    cd experiments/008_blood_cells_cnn
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
REPOS_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))


import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Import shared utilities
from src.config.paths import SILVER, TRAINED, setup_mlflow
from src.models.medical import get_medical_resnet
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


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
    in_channels = config["model"].get("in_channels", 3)

    model = get_medical_resnet(
        model_name=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
    )
    print(f"Created {architecture} (pretrained={pretrained}) with {num_classes} classes")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    return model


# =============================================================================
# DATA LOADING
# =============================================================================
def get_transforms(config: dict):
    """Get train and validation transforms."""
    img_size = config["data"].get("image_size", 224)

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_transform


def get_dataloaders(config: dict) -> tuple:
    """Create train, validation, and test DataLoaders."""
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    data_dir = SILVER / "blood-cells"

    train_transform, val_transform = get_transforms(config)

    print(f"Loading blood-cells from {data_dir}")

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch."""
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
    """Evaluate model on validation/test set."""
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
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0),
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )

    # Training loop with MLflow
    run_name = config["mlflow"].get("run_name")
    if not run_name or str(run_name).lower() == "none" or str(run_name).lower() == "null":
        arch = config.get("model", {}).get("architecture", "model")
        lr = config.get("training", {}).get("learning_rate", "lr")
        ep = config.get("training", {}).get("epochs", "ep")
        run_name = f"{arch}_lr{lr}_ep{ep}"
        
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": config["model"]["architecture"],
            "epochs": config["training"]["epochs"],
            "lr": config["training"]["learning_rate"],
            "batch_size": config["data"]["batch_size"],
            "image_size": config["data"].get("image_size", 224),
            "num_classes": config["model"]["num_classes"],
            "classes": str(class_names),
        })

        best_acc = 0
        patience_counter = 0
        patience = config["training"].get("early_stopping_patience", 10)

        for epoch in range(config["training"]["epochs"]):
            print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                save_path = TRAINED / config["experiment"]["name"] / "best_model.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"✓ New best model saved! (acc: {best_acc:.4f})")
                mlflow.log_artifact(str(save_path))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Final test evaluation
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        mlflow.log_metric("best_val_acc", best_acc)

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best Val Acc: {best_acc:.4f}")
        print(f"Test Acc:     {test_acc:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
