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
from src.training.trainer import Trainer


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
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        experiment_name=config["mlflow"]["experiment_name"]
    )

    save_dir = TRAINED / config["experiment"]["name"]

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        run_name=config["mlflow"].get("run_name"),
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()
