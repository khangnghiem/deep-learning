"""
=============================================================================
CIFAR-100 CNN EXPERIMENT - TRAINING SCRIPT
=============================================================================

Works both locally and in Colab (via 002_cifar100_cnn_train.ipynb).
Writes completed.json on finish for batch monitoring.

Usage:
    cd experiments/002_cifar100_cnn
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
from torchvision import datasets
from tqdm import tqdm

from src.config.paths import MLFLOW_TRACKING_URI, BRONZE, TRAINED, setup_mlflow
from src.models import SimpleCNN, get_pretrained_resnet
from src.data.transforms import get_cifar_transforms
from src.training import EarlyStopping, save_checkpoint
from src.utils.metrics import precision_recall_f1, get_confusion_matrix
from src.utils.visualization import plot_confusion_matrix


# =============================================================================
# CONFIGURATION
# =============================================================================
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def write_completion_marker(config, best_acc, duration, success, error=None):
    marker = {
        "experiment": config["experiment"]["name"],
        "completed_at": datetime.now().isoformat(),
        "success": success, "duration_seconds": round(duration, 1),
        "best_val_acc": round(best_acc, 4) if best_acc else None,
        "model": config["model"]["architecture"],
        "epochs": config["training"]["epochs"], "error": error,
    }
    Path("completed.json").write_text(json.dumps(marker, indent=2))
    print(f"📄 Completion marker written")


# =============================================================================
# MODEL
# =============================================================================
def get_model(config):
    architecture = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"].get("pretrained", True)

    if architecture == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes)
    elif architecture == "resnet18":
        model = get_pretrained_resnet("resnet18", num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    print(f"Created {architecture} with {num_classes} classes, "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    return model


# =============================================================================
# DATA
# =============================================================================
def get_dataloaders(config):
    from src.data.gold import GoldClassificationDataset
    import torchvision.transforms as transforms
    
    # We could dynamically load transforms here, but for simplicity we keep minimal normalization
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_ds = GoldClassificationDataset(experiment_name=config["experiment"]["name"], split="train", transform=transform)
    val_ds = GoldClassificationDataset(experiment_name=config["experiment"]["name"], split="val", transform=transform)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
    
    print(f"Loaded: {len(train_ds)} train, {len(val_ds)} val")
    return train_loader, val_loader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
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
    total_loss, correct, total = 0, 0, 0
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


def sanity_check(model, train_loader, criterion, device, steps=50):
    print("\n--- Sanity Check ---")
    model.train()
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(steps):
        opt.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        opt.step()
    acc = (model(inputs).argmax(1) == targets).float().mean().item()
    print(f"  loss={loss.item():.4f}, acc={acc:.4f}")
    if acc < 0.5:  # Lower threshold for 100 classes
        raise RuntimeError(f"Sanity check FAILED (acc={acc:.2f})")
    print("  ✅ Passed\n")


def evaluate(model, val_loader, device, config, mlflow):
    print("\n--- Evaluation ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating", leave=False, unit="batch"):
            preds = model(inputs.to(device)).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(targets)
    all_preds, all_labels = torch.cat(all_preds), torch.cat(all_labels)

    p, r, f1 = precision_recall_f1(all_labels, all_preds, average="weighted")
    print(f"Weighted — P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
    mlflow.log_metrics({"precision": p, "recall": r, "f1": f1})

    cm = get_confusion_matrix(all_labels, all_preds)
    eval_dir = TRAINED / config["experiment"]["name"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    cm_path = eval_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, save_path=cm_path)
    mlflow.log_artifact(str(cm_path))


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

    seed = config["training"].get("seed")
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Seed: {seed}")

    try:
        mlflow = setup_mlflow()
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model = get_model(config).to(device)
        train_loader, val_loader = get_dataloaders(config)
        criterion = nn.CrossEntropyLoss()

        lr = config["training"]["learning_rate"]
        opt_name = config["training"].get("optimizer", "adam").lower()
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            opt_name = "adam"

        sched_name = config["training"].get("scheduler", "null")
        scheduler = None
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        patience = config["training"].get("early_stopping_patience")
        early_stopping = EarlyStopping(patience=patience, mode="min") if patience else None

        sanity_check(model, train_loader, criterion, device)
        model = get_model(config).to(device)
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        run_name = config["mlflow"].get("run_name")
        if not run_name or str(run_name).lower() in ("none", "null"):
            run_name = f"{config['model']['architecture']}_{opt_name}_lr{lr}_ep{config['training']['epochs']}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": config["model"]["architecture"], "epochs": config["training"]["epochs"],
                "lr": lr, "optimizer": opt_name, "scheduler": sched_name,
                "batch_size": config["data"]["batch_size"], "seed": seed,
                "early_stopping_patience": patience,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            })

            best_epoch = 0
            save_dir = TRAINED / config["experiment"]["name"]
            save_dir.mkdir(parents=True, exist_ok=True)

            for epoch in range(config["training"]["epochs"]):
                t0 = time.time()
                print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                dt = time.time() - t0

                mlflow.log_metrics({
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"], "epoch_time_sec": dt,
                }, step=epoch)
                print(f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | {dt:.1f}s")

                if scheduler: scheduler.step()

                if val_acc > best_acc:
                    best_acc, best_epoch = val_acc, epoch + 1
                    save_checkpoint(model, optimizer, epoch, save_dir / "best_model.pt",
                                    scheduler=scheduler, metrics={"val_acc": val_acc})
                    print(f"✓ New best! ({best_acc:.4f})")
                    mlflow.log_artifact(str(save_dir / "best_model.pt"))

                save_checkpoint(model, optimizer, epoch, save_dir / "last_model.pt",
                                scheduler=scheduler, metrics={"val_acc": val_acc})

                if early_stopping and early_stopping(val_loss):
                    print(f"⏹️ Early stopping at epoch {epoch + 1}")
                    break

            mlflow.log_metrics({"best_val_acc": best_acc, "best_epoch": best_epoch})
            print(f"\n✅ Done! Best: {best_acc:.4f} (epoch {best_epoch})")
            evaluate(model, val_loader, device, config, mlflow)

        write_completion_marker(config, best_acc, time.time() - start_time, True)
    except Exception as e:
        write_completion_marker(config, best_acc, time.time() - start_time, False, str(e))
        raise


if __name__ == "__main__":
    main()
