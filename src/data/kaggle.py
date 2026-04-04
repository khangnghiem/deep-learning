"""
Kaggle Datasets Helper

Download datasets from Kaggle competitions and datasets.

Setup:
    1. Create Kaggle account
    2. Go to Account -> Create New API Token
    3. Save kaggle.json to My Drive/.secrets/kaggle.json (for Colab)
       or ~/.kaggle/kaggle.json (for local)
    
Usage:
    from src.data.kaggle import download_dataset, download_competition
    
    # Download a dataset
    download_dataset("uciml/iris")
    
    # Download competition data
    download_competition("titanic")
"""

import os
import subprocess
import re
from pathlib import Path
import sys

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_config.paths import get_bronze_path


def check_kaggle_auth():
    """Check if Kaggle API is configured. Checks .secrets first, then .kaggle."""
    # Check .secrets folder first (copied from Drive in Colab)
    secrets_json = Path.home() / ".secrets" / "kaggle.json"
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if secrets_json.exists():
        # Copy to ~/.kaggle for Kaggle CLI compatibility
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(secrets_json, kaggle_json)
        os.chmod(kaggle_json, 0o600)
        return
    
    if kaggle_json.exists():
        os.chmod(kaggle_json, 0o600)
        return
    
    raise FileNotFoundError(
        "Kaggle API not configured. Download kaggle.json from "
        "https://www.kaggle.com/settings and save to ~/.secrets/kaggle.json"
    )


def download_dataset(
    dataset: str,
    output_dir: Path = None,
    unzip: bool = True,
    category: str = "tabular"
) -> Path:
    """
    Download a Kaggle dataset.
    
    Args:
        dataset: Dataset slug (e.g., "uciml/iris", "hmendes/brasil-coronavirus")
        output_dir: Where to save. Defaults to category bronze folder / dataset_name
        unzip: Whether to unzip after download
        category: Dataset category for path routing (vision, tabular, nlp, etc.)
    
    Returns:
        Path to downloaded data
    """
    # Security: Validate dataset format strictly to prevent argument injection
    for part in dataset.split("/"):
        if not re.match(r'^[a-zA-Z0-9_][a-zA-Z0-9_-]*$', part):
            raise ValueError(f"Invalid dataset identifier part: {part}")

    check_kaggle_auth()
    
    dataset_name = dataset.split("/")[-1]
    if output_dir is None:
        output_dir = get_bronze_path(category) / dataset_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset,
        "-p", str(output_dir),
    ]
    if unzip:
        cmd.append("--unzip")
    
    print(f"Downloading {dataset} to {output_dir}")
    subprocess.run(cmd, check=True)
    
    return output_dir


def download_competition(
    competition: str,
    output_dir: Path = None,
    unzip: bool = True,
    category: str = "tabular"
) -> Path:
    """
    Download Kaggle competition data.
    
    Args:
        competition: Competition name (e.g., "titanic", "house-prices-advanced-regression")
        output_dir: Where to save. Defaults to category bronze folder / competition
        unzip: Whether to unzip after download
        category: Dataset category for path routing (vision, tabular, medical, etc.)
    
    Returns:
        Path to downloaded data
    """
    # Security: Validate competition format strictly to prevent argument injection
    if not re.match(r'^[a-zA-Z0-9_][a-zA-Z0-9_-]*$', competition):
        raise ValueError(f"Invalid competition identifier: {competition}")

    check_kaggle_auth()
    
    if output_dir is None:
        output_dir = get_bronze_path(category) / competition
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "kaggle", "competitions", "download",
        "-c", competition,
        "-p", str(output_dir),
    ]
    if unzip:
        cmd.append("--unzip")
    
    print(f"Downloading {competition} to {output_dir}")
    subprocess.run(cmd, check=True)
    
    return output_dir


def list_popular_competitions() -> list[str]:
    """Return list of popular beginner competitions."""
    return [
        "titanic",
        "house-prices-advanced-regression-techniques",
        "digit-recognizer",
        "spaceship-titanic",
        "store-sales-time-series-forecasting",
    ]


if __name__ == "__main__":
    print("Checking Kaggle authentication...")
    try:
        check_kaggle_auth()
        print("✓ Kaggle API configured")
    except FileNotFoundError as e:
        print(f"✗ {e}")
