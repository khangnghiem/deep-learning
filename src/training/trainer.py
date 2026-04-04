"""
Training utilities: trainer, callbacks, schedulers.

Usage:
    from src.training import Trainer, EarlyStopping
    
    trainer = Trainer(model, optimizer, criterion)
    trainer.fit(train_loader, val_loader, epochs=50)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable
from tqdm import tqdm
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_config.paths import setup_mlflow, TRAINED


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class Trainer:
    """Simple trainer with MLflow integration."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "auto",
        experiment_name: str = "deep-learning",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.experiment_name = experiment_name
        
        # Setup MLflow
        self.mlflow = setup_mlflow()
        self.mlflow.set_experiment(experiment_name)
    
    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """Single training epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.detach().item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(loader), correct / total
    
    def validate(self, loader: DataLoader) -> tuple[float, float]:
        """Validation loop."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(loader), correct / total
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping: Optional[EarlyStopping] = None,
        run_name: Optional[str] = None,
        save_dir: Optional[Path] = None,
    ) -> dict:
        """
        Train the model.
        
        Returns:
            Dict with training history
        """
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_acc = 0
        
        with self.mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            self.mlflow.log_params({
                "epochs": epochs,
                "optimizer": type(self.optimizer).__name__,
                "lr": self.optimizer.param_groups[0]["lr"],
            })
            
            for epoch in range(epochs):
                logger.info(f"\nEpoch {epoch + 1}/{epochs}")
                
                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc = self.validate(val_loader)
                
                # Log metrics
                self.mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, step=epoch)
                
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                
                # Learning rate scheduler
                if self.scheduler:
                    self.scheduler.step()
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    if save_dir:
                        save_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.state_dict(), save_dir / "best_model.pt")
                
                # Early stopping
                if early_stopping and early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            self.mlflow.log_metric("best_val_acc", best_acc)
        
        return history
