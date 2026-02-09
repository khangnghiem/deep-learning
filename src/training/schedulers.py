"""
Custom learning rate schedulers.

Usage:
    from src.training.schedulers import WarmupCosineScheduler, LinearWarmupScheduler
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    
    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by constant LR.
    
    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        return self.base_lrs


class OneCycleLR(_LRScheduler):
    """
    Simplified 1cycle learning rate policy.
    
    Args:
        optimizer: Optimizer
        max_lr: Maximum learning rate
        total_steps: Total training steps
        pct_start: Percentage of cycle spent increasing LR
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.total_steps * self.pct_start:
            # Increasing phase
            progress = step / (self.total_steps * self.pct_start)
            return [self.initial_lr + (self.max_lr - self.initial_lr) * progress]
        else:
            # Decreasing phase
            progress = (step - self.total_steps * self.pct_start) / \
                       (self.total_steps * (1 - self.pct_start))
            return [self.max_lr - (self.max_lr - self.final_lr) * progress]
