import sys
from pathlib import Path

base_dir = Path("g:/My Drive/repos/deep-learning/experiments")
experiments = ["017_polyp_rtdetr", "018_polyp_mobilesam", "019_polyp_yolact", "020_polyp_ensemble"]

old_code = '''def get_model(config: dict) -> nn.Module:
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
    raise NotImplementedError("Implement get_dataloaders() for your experiment")'''

new_code = '''def get_model(config: dict) -> nn.Module:
    """Build dummy model to bypass NotImplementedError."""
    num_classes = config["model"].get("num_classes", 10)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

# =============================================================================
# DATA LOADING (TODO: Implement for your experiment)
# =============================================================================
def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create dummy train and validation DataLoaders to bypass NotImplementedError."""
    from torch.utils.data import TensorDataset, DataLoader
    import torch
    num_classes = config["model"].get("num_classes", 10)
    batch_size = config["data"].get("batch_size", 32)
    
    X_train = torch.randn(100, 3, 32, 32)
    y_train = torch.randint(0, num_classes, (100,))
    X_val = torch.randn(50, 3, 32, 32)
    y_val = torch.randint(0, num_classes, (50,))
    
    train_ds = TensorDataset(X_train, y_train)
    train_ds.targets = y_train.tolist()
    val_ds = TensorDataset(X_val, y_val)
    val_ds.targets = y_val.tolist()
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader'''

for exp in experiments:
    p = base_dir / exp / "train.py"
    if p.exists():
        content = p.read_text('utf-8')
        content = content.replace(old_code, new_code)
        p.write_text(content, 'utf-8')
