"""
SVHN CNN Experiment - LOCAL TRAINING (Street View House Numbers)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from shared_config.paths import BRONZE, TRAINED, setup_mlflow
from src.models import SimpleCNN
from src.training.trainer import Trainer


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    
    data_dir = BRONZE / "svhn"
    
    train_ds = datasets.SVHN(str(data_dir), split='train', download=True, transform=transform)
    val_ds = datasets.SVHN(str(data_dir), split='test', download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
    
    print(f"SVHN: {len(train_ds)} train, {len(val_ds)} val")
    return train_loader, val_loader


def main():
    config = load_config()
    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']}")
    print("=" * 60)
    
    mlflow = setup_mlflow()
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = SimpleCNN(num_classes=10).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()
