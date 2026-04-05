"""
Checkpoint utilities for saving and loading models.

Usage:
    from src.training.checkpoint import save_checkpoint, load_checkpoint, ModelCheckpoint
    
    save_checkpoint(model, optimizer, epoch, "model.pt")
    model, optimizer, epoch = load_checkpoint("model.pt", model, optimizer)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_config.paths import TRAINED


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    metrics: Optional[dict] = None,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        path: Save path
        scheduler: Optional LR scheduler
        metrics: Optional dict of metrics
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
) -> tuple:
    """
    Load a training checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load to
    
    Returns:
        Tuple of (model, optimizer, epoch, metrics)
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    
    return model, optimizer, epoch, metrics


class ModelCheckpoint:
    """
    Callback to save best model during training.
    
    Usage:
        checkpoint = ModelCheckpoint(save_dir="checkpoints", monitor="val_acc", mode="max")
        
        for epoch in range(epochs):
            ...
            checkpoint(model, optimizer, epoch, {"val_acc": val_acc})
    """
    
    def __init__(
        self,
        save_dir: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True,
        save_top_k: int = 1,
    ):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_top_k = save_top_k
        
        self.best_score = None
        self.best_path = None
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> bool:
        """
        Check if model should be saved.
        
        Returns:
            True if a new best model was saved
        """
        score = metrics.get(self.monitor)
        if score is None:
            return False
        
        is_best = False
        
        if self.best_score is None:
            is_best = True
        elif self.mode == "min" and score < self.best_score:
            is_best = True
        elif self.mode == "max" and score > self.best_score:
            is_best = True
        
        if is_best:
            self.best_score = score
            self.best_path = self.save_dir / f"best_model_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, self.best_path, scheduler, metrics)
        
        if self.save_last:
            last_path = self.save_dir / "last_model.pt"
            save_checkpoint(model, optimizer, epoch, last_path, scheduler, metrics)
        
        return is_best
