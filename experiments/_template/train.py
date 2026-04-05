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
from collections import Counter

# =============================================================================
# LOCAL SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOS_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))


import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Auto-release Colab GPU runtime when script finishes
import atexit
try:
    from google.colab import runtime
    # atexit will fire when the python interpreter shuts down.
    # Note: When using `%run train.py` in IPython, this may not fire until the kernel dies.
    atexit.register(runtime.unassign)
except ImportError:
    pass  # Not in Colab — no-op

# Import shared utilities
from src.config.paths import MLFLOW_TRACKING_URI, BRONZE, TRAINED, setup_mlflow
from src.training import EarlyStopping, save_checkpoint, load_checkpoint
from src.utils.metrics import precision_recall_f1, get_confusion_matrix
from src.utils.visualization import plot_confusion_matrix, show_predictions
# TODO: Import your model and transforms
# from src.models import YourModel
# from src.data.transforms import get_transforms


# =============================================================================
# CONFIGURATION
# =============================================================================
REQUIRED_CONFIG = {
    "experiment": ["name"],
    "data": ["dataset", "batch_size", "num_workers"],
    "model": ["architecture", "num_classes"],
    "training": ["epochs", "learning_rate"],
    "mlflow": ["experiment_name"],
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and validate experiment configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: dict):
    """Check that all required config sections and keys exist."""
    for section, keys in REQUIRED_CONFIG.items():
        if section not in config:
            raise ValueError(f"Missing config section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing config key: '{section}.{key}'")


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
    print(f"📄 Completion marker written to {marker_path}")


# =============================================================================
# MODEL (TODO: Implement for your experiment)
# =============================================================================
def get_model(config: dict) -> nn.Module:
    """Build model based on configuration."""
    # TODO: Implement model creation for your experiment
    # Example:
    #   from src.models import SimpleCNN, get_pretrained_resnet
    #   if config["model"]["architecture"] == "simple_cnn":
    #       return SimpleCNN(num_classes=config["model"]["num_classes"])
    raise NotImplementedError("Implement get_model() for your experiment")


# =============================================================================
# DATA LOADING (TODO: Implement for your experiment)
# =============================================================================
def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    # TODO: Implement data loading for your experiment
    # Example:
    #   from torchvision import datasets
    #   from src.data.transforms import get_cifar_transforms
    #   train_ds = datasets.CIFAR10(root=BRONZE/"cifar10", train=True, ...)
    #   val_ds = datasets.CIFAR10(root=BRONZE/"cifar10", train=False, ...)
    raise NotImplementedError("Implement get_dataloaders() for your experiment")


# =============================================================================
# DATA VALIDATION
# =============================================================================
def validate_data(train_loader, val_loader, mlflow):
    """Verify data integrity: class distribution, sample shape, value range."""
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    # Class distribution (works for datasets with .targets attribute)
    if hasattr(train_dataset, "targets"):
        train_dist = Counter(train_dataset.targets)
        print(f"Train class distribution: {dict(sorted(train_dist.items()))}")
    if hasattr(val_dataset, "targets"):
        val_dist = Counter(val_dataset.targets)
        print(f"Val   class distribution: {dict(sorted(val_dist.items()))}")

    # Verify sample shape
    sample, label = train_dataset[0]
    print(f"Sample shape: {sample.shape}, min: {sample.min():.3f}, max: {sample.max():.3f}, label: {label}")

    # Log dataset info to MLflow
    mlflow.log_params({
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    })


# =============================================================================
# SANITY CHECK
# =============================================================================
def sanity_check(model, train_loader, criterion, device, steps=50):
    """Verify model can overfit a single batch — catches broken pipelines fast."""
    print("\n--- Sanity Check: Overfit One Batch ---")
    model.train()
    batch = next(iter(train_loader))
    inputs, targets = batch[0].to(device), batch[1].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    _, predicted = outputs.max(1)
    acc = predicted.eq(targets).float().mean().item()
    print(f"  After {steps} steps → loss: {loss.item():.4f}, acc: {acc:.4f}")

    if acc < 0.8:
        raise RuntimeError(
            f"Sanity check FAILED: model can't overfit one batch (acc={acc:.2f}). "
            "Check model architecture, loss function, or data transforms."
        )
    print("  ✅ Sanity check PASSED — model can learn\n")


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
    """Evaluate model on validation set."""
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
# POST-TRAINING EVALUATION
# =============================================================================
def evaluate(model, val_loader, device, config, mlflow, class_names=None):
    """Full post-training evaluation: confusion matrix, per-class F1."""
    print("\n--- Post-Training Evaluation ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            outputs = model(inputs.to(device))
            _, preds = outputs.max(1)
            all_preds.append(preds.cpu())
            all_labels.append(targets)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Weighted metrics
    precision, recall, f1 = precision_recall_f1(all_labels, all_preds, average="weighted")
    print(f"Weighted — Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    mlflow.log_metrics({"precision": precision, "recall": recall, "f1": f1})

    # Confusion matrix
    cm = get_confusion_matrix(all_labels, all_preds)
    eval_dir = TRAINED / config["experiment"]["name"]
    eval_dir.mkdir(parents=True, exist_ok=True)

    cm_path = eval_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, save_path=cm_path)
    mlflow.log_artifact(str(cm_path))
    print(f"📊 Confusion matrix saved to {cm_path}")


# =============================================================================
# OPTIMIZER / SCHEDULER / EARLY STOPPING FACTORIES
# =============================================================================
def create_optimizer(model, config):
    """Create optimizer from config."""
    lr = config["training"]["learning_rate"]
    opt_name = config["training"].get("optimizer", "adam").lower()
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4), opt_name
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2), opt_name
    else:  # adam (default)
        return torch.optim.Adam(model.parameters(), lr=lr), "adam"


def create_scheduler(optimizer, config):
    """Create LR scheduler from config."""
    sched_name = config["training"].get("scheduler", "null")
    if sched_name and sched_name != "null":
        sched_name = str(sched_name).lower()
    else:
        sched_name = "null"

    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"]
        ), sched_name
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1), sched_name
    return None, sched_name


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
        lr = config["training"]["learning_rate"]
        optimizer, opt_name = create_optimizer(model, config)
        scheduler, sched_name = create_scheduler(optimizer, config)

        # Early stopping
        patience = config["training"].get("early_stopping_patience")
        early_stopping = None
        if patience:
            early_stopping = EarlyStopping(patience=patience, mode="min")

        # Sanity check — verify model can learn before committing
        sanity_check(model, train_loader, criterion, device, steps=50)

        # Re-initialize model after sanity check
        model = get_model(config).to(device)
        optimizer, opt_name = create_optimizer(model, config)
        scheduler, sched_name = create_scheduler(optimizer, config)

        # MLflow run
        run_name = config["mlflow"].get("run_name")
        if not run_name or str(run_name).lower() in ("none", "null"):
            arch = config.get("model", {}).get("architecture", "model")
            run_name = f"{arch}_{opt_name}_lr{lr}_ep{config['training']['epochs']}"

        with mlflow.start_run(run_name=run_name):
            # Log all hyperparameters + environment
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_params({
                "model": config["model"]["architecture"],
                "epochs": config["training"]["epochs"],
                "lr": lr,
                "optimizer": opt_name,
                "scheduler": sched_name,
                "batch_size": config["data"]["batch_size"],
                "seed": seed,
                "early_stopping_patience": patience,
                "trainable_params": num_params,
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            })

            # Validate data
            validate_data(train_loader, val_loader, mlflow)

            best_epoch = 0
            save_dir = TRAINED / config["experiment"]["name"]
            save_dir.mkdir(parents=True, exist_ok=True)

            for epoch in range(config["training"]["epochs"]):
                epoch_start = time.time()
                print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)

                epoch_time = time.time() - epoch_start

                mlflow.log_metrics({
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_time_sec": epoch_time,
                }, step=epoch)

                print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                print(f"LR: {optimizer.param_groups[0]['lr']:.6f} | Epoch time: {epoch_time:.1f}s")

                if scheduler:
                    scheduler.step()

                # Save best model (full checkpoint)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1
                    best_path = save_dir / "best_model.pt"
                    save_checkpoint(
                        model, optimizer, epoch, best_path,
                        scheduler=scheduler,
                        metrics={"val_acc": val_acc, "val_loss": val_loss},
                    )
                    print(f"✓ New best! (acc: {best_acc:.4f})")
                    mlflow.log_artifact(str(best_path))

                # Always save last checkpoint (for crash recovery)
                last_path = save_dir / "last_model.pt"
                save_checkpoint(
                    model, optimizer, epoch, last_path,
                    scheduler=scheduler,
                    metrics={"val_acc": val_acc, "val_loss": val_loss},
                )

                # Early stopping check
                if early_stopping:
                    if early_stopping(val_loss):
                        print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
                        mlflow.log_param("early_stopped_epoch", epoch + 1)
                        break
                    elif early_stopping.counter > 0:
                        print(f"Early stopping: {early_stopping.counter}/{patience}")

            mlflow.log_metrics({
                "best_val_acc": best_acc,
                "best_epoch": best_epoch,
            })
            print(f"\n✅ Training complete! Best val_acc: {best_acc:.4f} (epoch {best_epoch})")

            # Post-training evaluation
            evaluate(model, val_loader, device, config, mlflow)

        duration = time.time() - start_time
        write_completion_marker(config, best_acc, duration, success=True)

    except Exception as e:
        duration = time.time() - start_time
        write_completion_marker(config, best_acc, duration, success=False, error=str(e))
        raise


if __name__ == "__main__":
    main()
