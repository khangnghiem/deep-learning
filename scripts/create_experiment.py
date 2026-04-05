"""
Create a new experiment folder by copying the template and patching config.yaml.

The template (_template/) is the single source of truth. This script copies it
and patches config.yaml with dataset info. Everything else — model architecture,
data loading, training loop — is yours to fill in.

Usage:
    python create_experiment.py intel-image              # Create single experiment
    python create_experiment.py intel-image --number 12  # Force experiment number
    python create_experiment.py --list                   # List available datasets
    python create_experiment.py --list-pending           # Datasets without an experiment yet
"""

import sys
import yaml
import shutil
import argparse
from pathlib import Path

import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent / "shared_config"))

if os.name == "nt" and "DRIVE_ROOT" not in os.environ:
    os.environ["DRIVE_ROOT"] = "G:\\My Drive"

from shared_config.catalog import DATASETS

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
TEMPLATE_DIR = EXPERIMENTS_DIR / "_template"


# =============================================================================
# HELPERS
# =============================================================================

def get_next_experiment_number() -> int:
    """Get next available experiment number."""
    existing = [
        d.name for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir() and d.name[:3].isdigit()
    ]
    if not existing:
        return 1
    return max(int(d[:3]) for d in existing) + 1


def _smart_hyperparams(info: dict) -> dict:
    """Pick sensible default hyperparameters based on dataset size and category."""
    size_str = info.get("size", "100MB")
    # Parse size string like "60MB", "1.8GB"
    try:
        num = float("".join(c for c in size_str if c.isdigit() or c == "."))
        unit = "".join(c for c in size_str if c.isalpha()).upper()
        size_mb = num * 1024 if unit == "GB" else num
    except (ValueError, AttributeError):
        size_mb = 100

    category = info.get("category", "vision")

    if category == "nlp":
        return {"epochs": 5, "batch_size": 16, "lr": 2e-5, "optimizer": "adamw"}
    elif category == "tabular":
        return {"epochs": 50, "batch_size": 256, "lr": 1e-3, "optimizer": "adam"}
    elif size_mb < 100:
        return {"epochs": 15, "batch_size": 64, "lr": 1e-3, "optimizer": "adam"}
    elif size_mb < 500:
        return {"epochs": 20, "batch_size": 64, "lr": 1e-3, "optimizer": "adam"}
    elif size_mb < 2000:
        return {"epochs": 25, "batch_size": 32, "lr": 1e-4, "optimizer": "adamw"}
    else:
        return {"epochs": 30, "batch_size": 16, "lr": 1e-4, "optimizer": "adamw"}


# =============================================================================
# CORE: CREATE
# =============================================================================

def create_experiment(dataset_name: str, number: int = None) -> Path:
    """
    Copy the template and patch config.yaml for a given dataset.

    Returns:
        Path to the created experiment directory.
    """
    if dataset_name not in DATASETS:
        available = "\n  ".join(sorted(DATASETS.keys()))
        print(f"Unknown dataset: '{dataset_name}'\n\nAvailable:\n  {available}")
        sys.exit(1)

    info = DATASETS[dataset_name]

    if number is None:
        number = get_next_experiment_number()

    # Build experiment directory name
    safe_name = dataset_name.replace("-", "_")
    exp_name = f"{number:03d}_{safe_name}"
    exp_dir = EXPERIMENTS_DIR / exp_name

    if exp_dir.exists():
        print(f"Already exists: {exp_dir.name} — skipping.")
        return exp_dir

    # ----------------------------------------------------------------
    # Copy template verbatim
    # ----------------------------------------------------------------
    if not TEMPLATE_DIR.exists():
        print(f"Template not found: {TEMPLATE_DIR}")
        sys.exit(1)

    shutil.copytree(TEMPLATE_DIR, exp_dir)
    print(f"Copied template -> {exp_dir.name}/")

    # ----------------------------------------------------------------
    # Create mapped exploration notebook
    # ----------------------------------------------------------------
    EXPLORATIONS_DIR = PROJECT_ROOT / "explorations"
    template_nb = EXPLORATIONS_DIR / "_template.ipynb"
    target_nb = EXPLORATIONS_DIR / f"{exp_name}.ipynb"
    
    if template_nb.exists() and not target_nb.exists():
        nb_text = template_nb.read_text(encoding="utf-8")
        nb_text = nb_text.replace("{DATASET_NAME}", dataset_name)
        target_nb.write_text(nb_text, encoding="utf-8")
        print(f"Created paired exploration -> explorations/{target_nb.name}")

    # ----------------------------------------------------------------
    # Patch config.yaml
    # ----------------------------------------------------------------
    config_path = exp_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    hp = _smart_hyperparams(info)

    config["experiment"]["name"] = exp_name
    config["experiment"]["description"] = info.get("description", f"{dataset_name} experiment")
    config["data"]["dataset"] = dataset_name
    config["data"]["batch_size"] = hp["batch_size"]
    config["model"]["num_classes"] = info.get("classes", 10)
    config["training"]["epochs"] = hp["epochs"]
    config["training"]["learning_rate"] = hp["lr"]
    config["training"]["optimizer"] = hp["optimizer"]
    config["mlflow"]["experiment_name"] = dataset_name.replace("_", "-")

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Patched config.yaml:")
    print(f"  experiment:  {exp_name}")
    print(f"  dataset:     {dataset_name}  ({info.get('classes', '?')} classes)")
    print(f"  epochs:      {hp['epochs']}")
    print(f"  batch_size:  {hp['batch_size']}")
    print(f"  lr:          {hp['lr']}")
    print(f"  optimizer:   {hp['optimizer']}")

    # ----------------------------------------------------------------
    # Update README placeholder
    # ----------------------------------------------------------------
    readme_path = exp_dir / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        readme = readme.replace("{EXPERIMENT_NAME}", exp_name)
        readme = readme.replace("{DATASET_NAME}", dataset_name)
        readme = readme.replace("experiment_name", exp_name)
        readme_path.write_text(readme, encoding="utf-8")

    print(f"\nNext steps:")
    print(f"  1. Edit {exp_name}/train.py — implement get_model() and get_dataloaders()")
    print(f"  2. Edit {exp_name}/requirements.txt — add experiment-specific packages")
    print(f"  3. Run via Colab: open {exp_name}/template_reference_train.ipynb")

    return exp_dir


# =============================================================================
# COMMANDS
# =============================================================================

def cmd_list():
    """List all available datasets."""
    print(f"{'Dataset':<30} {'Category':<15} {'Classes':<10} {'Size':<10} Description")
    print("-" * 90)
    for name, info in sorted(DATASETS.items()):
        # Check if experiment already exists
        existing = [
            d.name for d in EXPERIMENTS_DIR.iterdir()
            if d.is_dir() and f"_{name.replace('-', '_')}" in d.name
        ]
        indicator = " [exists]" if existing else ""
        print(
            f"{name:<30} {info.get('category','?'):<15} "
            f"{info.get('classes','?'):<10} {info.get('size','?'):<10} "
            f"{info.get('description','')[:40]}{indicator}"
        )


def cmd_list_pending():
    """List datasets that don't have an experiment yet."""
    print("Datasets without an experiment:\n")
    for name, info in sorted(DATASETS.items()):
        safe = name.replace("-", "_")
        existing = [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and safe in d.name]
        if not existing:
            print(f"  {name:<30} {info.get('category','?'):<15} {info.get('description','')[:50]}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create a new experiment from the template.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", nargs="?", help="Dataset name from the catalog")
    parser.add_argument("--number", "-n", type=int, help="Force experiment number (default: auto)")
    parser.add_argument("--list", "-l", action="store_true", help="List available datasets")
    parser.add_argument("--list-pending", action="store_true", help="List datasets with no experiment")

    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.list_pending:
        cmd_list_pending()
    elif args.dataset:
        create_experiment(args.dataset, number=args.number)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
