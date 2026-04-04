"""
STL-10 CNN Experiment - LOCAL TRAINING (96x96 images)
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
from src.models import get_pretrained_resnet
from src.training.trainer import Trainer


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_dataloaders(config):
    # STL10 has 96x96 images - resize to 224 for pretrained ResNet
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    data_dir = BRONZE / "stl10"
    
    train_ds = datasets.STL10(str(data_dir), split='train', download=True, transform=train_transform)
    val_ds = datasets.STL10(str(data_dir), split='test', download=True, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
    
    print(f"STL10: {len(train_ds)} train, {len(val_ds)} val (96x96 → 224x224)")
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
    
    model = get_pretrained_resnet("resnet18", num_classes=10, pretrained=True).to(device)
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
