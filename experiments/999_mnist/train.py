"""
=============================================================================
EXPERIMENT TEMPLATE - TRAINING SCRIPT
=============================================================================

Works both locally and in Colab (via template_reference_train.ipynb).
Writes completed.json on finish for batch monitoring.

Usage:
    cd experiments/{EXPERIMENT_NAME}
    python train.py
"""

import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

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
from tqdm import tqdm

# Import shared utilities
from src.config.paths import MLFLOW_TRACKING_URI, BRONZE, TRAINED, setup_mlflow
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
# COMPLETION MARKER (for batch monitoring)
# =============================================================================
def write_completion_marker(config: dict, best_acc: float, duration: float, success: bool, error: str = None):
    """Write completed.json so batch_status.ipynb can track run results."""
    marker = {
        "experiment": config["experiment"]["name"],
        "completed_at": datetime.now().isoformat(),
        "success": success,
        "duration_seconds": round(duration, 1),
        "best_val_acc": round(best_acc, 4) if best_acc else None,
        "model": config["model"]["architecture"],
        "epochs": config["training"]["epochs"],
        "error": error,
    }
    marker_path = Path("completed.json")
    marker_path.write_text(json.dumps(marker, indent=2))
    print(f"\ud83d\udcc4 Completion marker written to {marker_path}")


# =============================================================================
# MODEL (TODO: Implement)
# =============================================================================
def get_model(config: dict) -> nn.Module:
    """Build model based on configuration."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, config["model"]["num_classes"])
    )


# =============================================================================
# DATA LOADING (TODO: Implement)
# =============================================================================
def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    from torch.utils.data import TensorDataset
    # Dummy dataset for fast local test
    x_train = torch.randn(100, 1, 28, 28)
    y_train = torch.randint(0, config["model"]["num_classes"], (100,))
    x_val = torch.randn(20, 1, 28, 28)
    y_val = torch.randint(0, config["model"]["num_classes"], (20,))
    
    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=config["data"]["batch_size"])
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=config["data"]["batch_size"])
    return train_dl, val_dl


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
    start_time = time.time()
    best_acc = 0

    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Reproducibility — set seed before anything else
    # ------------------------------------------------------------------
    seed = config["training"].get("seed")
    if seed is not None:
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Random seed: {seed}")

    try:
        mlflow = setup_mlflow()
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = get_model(config).to(device)
        train_loader, val_loader = get_dataloaders(config)

        criterion = nn.CrossEntropyLoss()

        # ------------------------------------------------------------------
        # Optimizer — controlled by config.training.optimizer
        # ------------------------------------------------------------------
        lr = config["training"]["learning_rate"]
        opt_name = config["training"].get("optimizer", "adam").lower()
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        else:  # adam (default)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ------------------------------------------------------------------
        # Scheduler — controlled by config.training.scheduler
        # ------------------------------------------------------------------
        scheduler = None
        sched_name = config["training"].get("scheduler", "null")
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["training"]["epochs"]
            )
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # ------------------------------------------------------------------
        # Early stopping — controlled by config.training.early_stopping_patience
        # ------------------------------------------------------------------
        patience = config["training"].get("early_stopping_patience")
        es_counter = 0
        es_best_loss = float("inf")

        run_name = config["mlflow"].get("run_name")
        if not run_name or str(run_name).lower() in ("none", "null"):
            arch = config.get("model", {}).get("architecture", "model")
            run_name = f"{arch}_{opt_name}_lr{lr}_ep{config['training']['epochs']}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": config["model"]["architecture"],
                "epochs": config["training"]["epochs"],
                "lr": lr,
                "optimizer": opt_name,
                "scheduler": sched_name,
                "batch_size": config["data"]["batch_size"],
                "seed": seed,
            })

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

                if scheduler:
                    scheduler.step()

                if val_acc > best_acc:
                    best_acc = val_acc
                    save_path = TRAINED / config["experiment"]["name"] / "best_model.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    print(f"\N{CHECK MARK} New best! (acc: {best_acc:.4f})")
                    mlflow.log_artifact(str(save_path))

                # Early stopping check
                if patience:
                    if val_loss < es_best_loss:
                        es_best_loss = val_loss
                        es_counter = 0
                    else:
                        es_counter += 1
                        print(f"Early stopping: {es_counter}/{patience}")
                        if es_counter >= patience:
                            print("Early stopping triggered.")
                            break

            mlflow.log_metric("best_val_acc", best_acc)
            print(f"\N{WHITE HEAVY CHECK MARK} Training complete! Best val_acc: {best_acc:.4f}")

        duration = time.time() - start_time
        write_completion_marker(config, best_acc, duration, success=True)

    except Exception as e:
        duration = time.time() - start_time
        write_completion_marker(config, best_acc, duration, success=False, error=str(e))
        raise


if __name__ == "__main__":
    main()
