"""
STL-10 CNN Experiment - LOCAL TRAINING (96x96 images)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOS_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))


import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.config.paths import BRONZE, TRAINED, setup_mlflow
from src.models import get_pretrained_resnet


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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total


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
    
    with mlflow.start_run():
        mlflow.log_params({"model": "ResNet18", "epochs": config["training"]["epochs"], "lr": config["training"]["learning_rate"], "dataset": "stl10"})
        
        best_acc = 0
        for epoch in range(config["training"]["epochs"]):
            print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            print(f"Train: {train_acc:.4f} | Val: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = TRAINED / config["experiment"]["name"] / "best_model.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"✓ Best: {best_acc:.4f}")
        
        mlflow.log_metric("best_val_acc", best_acc)
        print(f"\n✅ Complete! Best: {best_acc:.4f}")


if __name__ == "__main__":
    main()
