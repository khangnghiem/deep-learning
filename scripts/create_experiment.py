"""
Auto-generate experiment folders from the dataset catalog.

Usage:
    python create_experiment.py intel-image           # Create single experiment
    python create_experiment.py --batch vision        # Create all vision experiments
    python create_experiment.py --template transformer intel-image  # Use transformer template
    python create_experiment.py --list                # List available datasets
    python create_experiment.py --list-templates      # List available templates
"""

import sys
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shared_config.catalog import DATASETS

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
TEMPLATE_DIR = EXPERIMENTS_DIR / "_template"
TEMPLATES_DIR = EXPERIMENTS_DIR / "_templates"  # Custom templates directory

# Built-in template configurations
TEMPLATES = {
    "cnn": {
        "description": "Standard CNN for image classification",
        "categories": ["vision", "medical"],
        "architecture": "simple_cnn",
    },
    "resnet": {
        "description": "Pretrained ResNet for image classification",
        "categories": ["vision", "medical"],
        "architecture": "resnet18",
    },
    "transformer": {
        "description": "Transformer/BERT for text classification",
        "categories": ["nlp"],
        "architecture": "distilbert",
    },
    "tabular": {
        "description": "MLP for tabular data",
        "categories": ["tabular"],
        "architecture": "mlp",
    },
    "audio": {
        "description": "CNN for spectrogram/audio classification",
        "categories": ["audio"],
        "architecture": "audio_cnn",
    },
    "timeseries": {
        "description": "LSTM/Transformer for time series forecasting",
        "categories": ["timeseries"],
        "architecture": "lstm",
    },
}


def get_next_experiment_number():
    """Get next available experiment number."""
    existing = [d.name for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and d.name[:3].isdigit()]
    if not existing:
        return 7  # Start after 006
    numbers = [int(d[:3]) for d in existing if d[:3].isdigit()]
    return max(numbers) + 1


def create_experiment(dataset_name: str, number: int = None, template: str = None):
    """Create a new experiment folder for a dataset."""
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        return None
    
    info = DATASETS[dataset_name]
    category = info.get("category", "vision")
    
    # Auto-select template if not specified
    if template is None:
        template = get_default_template(category)
    
    if template and template not in TEMPLATES:
        print(f"⚠️  Unknown template: {template}")
        print(f"   Available: {', '.join(TEMPLATES.keys())}")
        template = get_default_template(category)
        print(f"   Using default: {template}")
    
    if number is None:
        number = get_next_experiment_number()
    
    # Generate experiment name
    safe_name = dataset_name.replace("-", "_")
    arch = TEMPLATES.get(template, {}).get("architecture", "cnn")
    exp_name = f"{number:03d}_{safe_name}_{arch}"
    exp_dir = EXPERIMENTS_DIR / exp_name
    
    if exp_dir.exists():
        print(f"⏭️  {exp_name} already exists, skipping...")
        return exp_dir
    
    # Copy base template
    if TEMPLATE_DIR.exists():
        shutil.copytree(TEMPLATE_DIR, exp_dir)
    else:
        exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate config.yaml
    config_content = generate_config(dataset_name, exp_name, info, template)
    (exp_dir / "config.yaml").write_text(config_content)
    
    # Generate train.py based on template
    train_content = generate_train_script(dataset_name, info, template)
    (exp_dir / "train.py").write_text(train_content)
    
    # Generate README.md
    readme_content = generate_readme(dataset_name, exp_name, info, template)
    (exp_dir / "README.md").write_text(readme_content)
    
    print(f"✅ Created {exp_name} (template: {template})")
    return exp_dir


def get_default_template(category: str) -> str:
    """Get the default template for a category."""
    category_map = {
        "vision": "cnn",
        "medical": "cnn",
        "nlp": "transformer",
        "tabular": "tabular",
        "audio": "audio",
        "timeseries": "timeseries",
        "detection": "resnet",
        "generative": "resnet",
        "video": "resnet",
    }
    return category_map.get(category, "cnn")


def generate_config(dataset_name, exp_name, info, template="cnn"):
    """Generate config.yaml for the experiment."""
    num_classes = info.get("classes", 10)
    size_mb = _parse_size(info.get("size", "100MB"))
    
    # Get architecture from template
    arch = TEMPLATES.get(template, {}).get("architecture", "simple_cnn")
    
    # Adjust hyperparameters based on dataset size and template
    if template == "transformer":
        epochs, batch_size, lr = 5, 16, 2e-5
    elif template == "tabular":
        epochs, batch_size, lr = 50, 256, 0.001
    elif size_mb < 100:
        epochs, batch_size, lr = 15, 64, 0.001
    elif size_mb < 500:
        epochs, batch_size, lr = 20, 64, 0.001
    elif size_mb < 2000:
        epochs, batch_size, lr = 25, 32, 0.0001
    else:
        epochs, batch_size, lr = 30, 16, 0.0001
    
    return f'''# {exp_name}

experiment:
  name: "{exp_name}"
  description: "{dataset_name} classification"
  template: "{template}"

data:
  dataset: "{dataset_name}"
  batch_size: {batch_size}
  num_workers: 4

model:
  architecture: "{arch}"
  pretrained: true
  num_classes: {num_classes}

training:
  epochs: {epochs}
  learning_rate: {lr}
  optimizer: "adam"

mlflow:
  experiment_name: "{dataset_name}-{arch.replace('_', '-')}"
  run_name: null
'''


def generate_train_script(dataset_name, info, template="cnn"):
    """Generate train.py based on template."""
    category = info.get("category", "vision")
    source = info.get("source", "torchvision")
    
    if template == "transformer":
        return _generate_transformer_train(dataset_name, info)
    elif template == "tabular":
        return _generate_tabular_train(dataset_name, info)
    elif template == "audio":
        return _generate_audio_train(dataset_name, info)
    elif template == "timeseries":
        return _generate_timeseries_train(dataset_name, info)
    else:
        return _generate_vision_train(dataset_name, info, template)


def _generate_vision_train(dataset_name, info, template="cnn"):
    """Generate train.py based on dataset category."""
    category = info.get("category", "vision")
    source = info.get("source", "torchvision")
    num_classes = info.get("classes", 10)
    
    return f'''"""
{dataset_name.upper()} Experiment - AUTO-GENERATED
Category: {category} | Source: {source}
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from shared_config.paths import get_bronze_path, TRAINED, setup_mlflow
from src.models import SimpleCNN, get_pretrained_resnet


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model(config):
    arch = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]
    
    if arch == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    else:
        return get_pretrained_resnet(arch, num_classes=num_classes, pretrained=True)


def get_dataloaders(config):
    """Load {dataset_name} dataset."""
    from torchvision import datasets as tv_datasets
    from torch.utils.data import random_split
    
    data_dir = get_bronze_path("{category}") / "{dataset_name}"
    batch_size = config["data"]["batch_size"]
    
    # Standard transforms for {category}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # TODO: Customize data loading for {dataset_name}
    # Option 1: ImageFolder (if organized as class folders)
    dataset = tv_datasets.ImageFolder(data_dir, transform=transform)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Dataset: {{len(dataset)}} images, {{config['model']['num_classes']}} classes")
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False, unit="batch"):
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
        for x, y in tqdm(loader, desc="Val", leave=False, unit="batch"):
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
    print(f"Experiment: {{config['experiment']['name']}}")
    print("=" * 60)
    
    mlflow = setup_mlflow()
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {{device}}")
    
    model = get_model(config).to(device)
    train_loader, val_loader = get_dataloaders(config)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    with mlflow.start_run():
        mlflow.log_params({{
            "model": config["model"]["architecture"],
            "epochs": config["training"]["epochs"],
            "lr": config["training"]["learning_rate"],
            "dataset": "{dataset_name}"
        }})
        
        best_acc = 0
        for epoch in range(config["training"]["epochs"]):
            print(f"\\nEpoch {{epoch+1}}/{{config['training']['epochs']}}")
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            mlflow.log_metrics({{"train_loss": train_loss, "train_acc": train_acc, 
                                "val_loss": val_loss, "val_acc": val_acc}}, step=epoch)
            print(f"Train: {{train_acc:.4f}} | Val: {{val_acc:.4f}}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = TRAINED / config["experiment"]["name"] / "best_model.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"✓ Best: {{best_acc:.4f}}")
        
        mlflow.log_metric("best_val_acc", best_acc)
        print(f"\\n✅ Complete! Best: {{best_acc:.4f}}")


if __name__ == "__main__":
    main()
'''


def _generate_transformer_train(dataset_name, info):
    """Generate train.py for NLP/transformer models."""
    return f'''"""
{dataset_name.upper()} Experiment - TRANSFORMER TEMPLATE
Category: nlp | Architecture: DistilBERT
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from shared_config.paths import get_bronze_path, TRAINED, setup_mlflow


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    print("=" * 60)
    print(f"Experiment: {{config['experiment']['name']}}")
    print("=" * 60)
    
    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config["model"]["num_classes"]
    )
    
    # Load dataset (customize for your dataset)
    data_dir = get_bronze_path("nlp") / "{dataset_name}"
    # TODO: Customize data loading for {dataset_name}
    
    # Training args
    training_args = TrainingArguments(
        output_dir=str(TRAINED / config["experiment"]["name"]),
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["data"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Use HuggingFace Trainer
    # trainer = Trainer(model=model, args=training_args, ...)
    
    print("TODO: Complete transformer training loop for {dataset_name}")


if __name__ == "__main__":
    main()
'''


def _generate_tabular_train(dataset_name, info):
    """Generate train.py for tabular/MLP models."""
    return f'''"""
{dataset_name.upper()} Experiment - TABULAR TEMPLATE
Category: tabular | Architecture: MLP
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from shared_config.paths import get_bronze_path, TRAINED, setup_mlflow


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    print("=" * 60)
    print(f"Experiment: {{config['experiment']['name']}}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    data_dir = get_bronze_path("tabular") / "{dataset_name}"
    # TODO: Customize for {dataset_name}
    # df = pd.read_csv(data_dir / "data.csv")
    
    print("TODO: Complete tabular training loop for {dataset_name}")
    print("1. Load CSV/parquet data")
    print("2. Encode categorical features")
    print("3. Split train/val")
    print("4. Train MLP")


if __name__ == "__main__":
    main()
'''


def _generate_audio_train(dataset_name, info):
    """Generate train.py for audio classification."""
    return f'''"""
{dataset_name.upper()} Experiment - AUDIO TEMPLATE
Category: audio | Architecture: Audio CNN (Spectrogram)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from shared_config.paths import get_bronze_path, TRAINED, setup_mlflow


class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    print("=" * 60)
    print(f"Experiment: {{config['experiment']['name']}}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load audio data
    data_dir = get_bronze_path("audio") / "{dataset_name}"
    
    # Convert audio to mel spectrogram
    # mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=128)
    
    print("TODO: Complete audio training loop for {dataset_name}")
    print("1. Load audio files")
    print("2. Convert to mel spectrograms")
    print("3. Train CNN on spectrograms")


if __name__ == "__main__":
    main()
'''


def _generate_timeseries_train(dataset_name, info):
    """Generate train.py for time series forecasting."""
    return f'''"""
{dataset_name.upper()} Experiment - TIMESERIES TEMPLATE
Category: timeseries | Architecture: LSTM
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from shared_config.paths import get_bronze_path, TRAINED, setup_mlflow


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=30):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return x, y


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    print("=" * 60)
    print(f"Experiment: {{config['experiment']['name']}}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load time series data
    data_dir = get_bronze_path("timeseries") / "{dataset_name}"
    
    # TODO: Customize for {dataset_name}
    # df = pd.read_csv(data_dir / "data.csv")
    # data = df["value"].values
    
    print("TODO: Complete time series training loop for {dataset_name}")
    print("1. Load time series data (CSV/Parquet)")
    print("2. Normalize data")
    print("3. Create sequences with TimeSeriesDataset")
    print("4. Train LSTM model")
    print("5. Evaluate on validation set")


if __name__ == "__main__":
    main()
'''


def generate_readme(dataset_name, exp_name, info, template="cnn"):
    """Generate README.md for the experiment."""
    template_info = TEMPLATES.get(template, {})
    return f'''# {exp_name}

## Goal
{dataset_name} classification using {template_info.get("description", "neural network")}.

## Dataset
- **Source**: {info.get("source", "unknown")}
- **Size**: {info.get("size", "?")}
- **Classes**: {info.get("classes", "?")}
- **Category**: {info.get("category", "vision")}

## Template
Using **{template}** template with `{template_info.get("architecture", "?")}` architecture.

## Run

```bash
# Download data first
python ../data-ingestion/scripts/ingest.py {dataset_name}

# Train
python train.py
```

## Results
| Model | Best Val Acc | Epochs |
|-------|--------------|--------|
| | | |
'''


def _parse_size(size_str):
    size_str = size_str.upper().replace(" ", "")
    if "GB" in size_str:
        return float(size_str.replace("GB", "")) * 1024
    elif "MB" in size_str:
        return float(size_str.replace("MB", ""))
    return 0


def batch_create(category: str = None, template: str = None):
    """Create experiments for all datasets in a category."""
    datasets_to_create = []
    for name, info in DATASETS.items():
        cat = info.get("category", "other")
        size = _parse_size(info.get("size", "0"))
        
        # Filter by category and size (<2GB for practical training)
        if category and cat != category:
            continue
        if size > 2048:  # Skip >2GB datasets
            continue
        
        datasets_to_create.append(name)
    
    print(f"Creating {len(datasets_to_create)} experiments...")
    for name in sorted(datasets_to_create):
        create_experiment(name, template=template)


def list_templates():
    """List available templates."""
    print("\n📋 Available Templates\n")
    for name, info in TEMPLATES.items():
        cats = ", ".join(info["categories"])
        print(f"  {name:12} - {info['description']} (for: {cats})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate experiment folders from dataset catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_experiment.py intel-image              # Create single experiment
  python create_experiment.py --template resnet mnist  # Use resnet template
  python create_experiment.py --batch vision           # All vision experiments
  python create_experiment.py --list-templates         # Show available templates
        """
    )
    
    parser.add_argument("dataset", nargs="?", help="Dataset name to create experiment for")
    parser.add_argument("--list", "-l", action="store_true", help="List available datasets")
    parser.add_argument("--list-templates", action="store_true", help="List available templates")
    parser.add_argument("--batch", metavar="CATEGORY", help="Create experiments for all datasets in category")
    parser.add_argument("--template", "-t", choices=list(TEMPLATES.keys()),
                       help="Template to use (cnn, resnet, transformer, tabular, audio)")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable datasets for experiments:")
        for cat in ["vision", "medical", "nlp", "audio", "tabular"]:
            datasets = [n for n, i in DATASETS.items() if i.get("category") == cat]
            print(f"\n{cat.upper()}: {len(datasets)} datasets")
        print(f"\nTotal: {len(DATASETS)} datasets")
    elif args.list_templates:
        list_templates()
    elif args.batch:
        batch_create(args.batch, template=args.template)
    elif args.dataset:
        create_experiment(args.dataset, template=args.template)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
