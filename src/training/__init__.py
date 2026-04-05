"""
Training utilities: losses, schedulers, checkpointing, early stopping.

These are composable building blocks — use them directly in your
experiment's train.py as needed. There is no Trainer framework;
each experiment owns its full training loop.

Quick imports:
    from src.training import EarlyStopping
    from src.training import FocalLoss, DiceLoss, LabelSmoothingCE
    from src.training import WarmupCosineScheduler
    from src.training import save_checkpoint, load_checkpoint
"""

from .early_stopping import EarlyStopping
from .losses import FocalLoss, LabelSmoothingCE, DiceLoss, ContrastiveLoss
from .schedulers import WarmupCosineScheduler, LinearWarmupScheduler, OneCycleLR
from .checkpoint import save_checkpoint, load_checkpoint, ModelCheckpoint

__all__ = [
    # Early stopping
    "EarlyStopping",
    # Losses
    "FocalLoss",
    "LabelSmoothingCE",
    "DiceLoss",
    "ContrastiveLoss",
    # Schedulers
    "WarmupCosineScheduler",
    "LinearWarmupScheduler",
    "OneCycleLR",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "ModelCheckpoint",
]
