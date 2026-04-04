"""
HuggingFace Datasets Helper

Easy access to 50K+ datasets from HuggingFace Hub.

Usage:
    from src.data.huggingface import load_hf_dataset, list_popular_datasets
    
    # Load a dataset
    dataset = load_hf_dataset("cifar10")
    
    # Load with specific split
    train_ds = load_hf_dataset("mnist", split="train")
"""

from pathlib import Path
from datasets import load_dataset
import sys
import logging

logger = logging.getLogger(__name__)

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_config.paths import get_bronze_path


def load_hf_dataset(
    name: str,
    split: str = None,
    cache_dir: Path = None,
    category: str = "vision",
    **kwargs
):
    """
    Load a dataset from HuggingFace Hub.
    
    Args:
        name: Dataset name (e.g., "cifar10", "mnist", "imagenet-1k")
        split: Optional split ("train", "test", "validation")
        cache_dir: Where to cache. Defaults to category bronze folder / "huggingface"
        category: Dataset category for path routing (vision, nlp, audio, etc.)
        **kwargs: Additional args passed to load_dataset
    
    Returns:
        Dataset or DatasetDict
    """
    if cache_dir is None:
        cache_dir = get_bronze_path(category) / "huggingface"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    return load_dataset(
        name,
        split=split,
        cache_dir=str(cache_dir),
        **kwargs
    )


def list_popular_datasets() -> list[str]:
    """Return a list of popular image datasets."""
    return [
        # Image Classification
        "cifar10",
        "cifar100",
        "mnist",
        "fashion_mnist",
        "imagenet-1k",
        "food101",
        "oxford_flowers102",
        "stanford_dogs",
        
        # Object Detection
        "coco",
        "voc2007",
        
        # Medical
        "marmal88/skin_cancer",
        "alkzar90/NIH-Chest-X-ray-dataset",
        
        # NLP (for reference)
        "imdb",
        "glue",
        "squad",
    ]


def get_image_classification_datasets() -> dict:
    """Return dict of image classification datasets with metadata."""
    return {
        "cifar10": {"classes": 10, "size": "60K images", "resolution": "32x32"},
        "cifar100": {"classes": 100, "size": "60K images", "resolution": "32x32"},
        "mnist": {"classes": 10, "size": "70K images", "resolution": "28x28"},
        "fashion_mnist": {"classes": 10, "size": "70K images", "resolution": "28x28"},
        "imagenet-1k": {"classes": 1000, "size": "1.2M images", "resolution": "variable"},
        "food101": {"classes": 101, "size": "101K images", "resolution": "variable"},
    }


if __name__ == "__main__":
    print("Popular datasets:")
    for ds in list_popular_datasets():
        print(f"  - {ds}")
