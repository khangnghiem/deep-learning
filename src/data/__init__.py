"""
Data loading, acquisition, and transforms.
"""

try:
    from .huggingface import load_hf_dataset, list_popular_datasets
except ImportError:
    load_hf_dataset = None
    list_popular_datasets = None

from .kaggle import download_dataset, download_competition
from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_cifar_transforms,
    get_mnist_transforms,
)
from .loaders import create_dataloaders, get_class_weights, create_imbalanced_sampler
from .medical import (
    get_medical_datasets,
    download_medical_dataset,
    list_medical_datasets,
)
from .gold import GoldClassificationDataset, GoldSegmentationDataset
from .mlflow_tracker import (
    compute_file_sha256,
    create_gold_manifest,
    log_medallion_dataset,
)

__all__ = [
    # HuggingFace
    "load_hf_dataset",
    "list_popular_datasets",
    # Kaggle
    "download_dataset",
    "download_competition",
    # Transforms
    "get_train_transforms",
    "get_val_transforms",
    "get_cifar_transforms",
    "get_mnist_transforms",
    # Loaders
    "create_dataloaders",
    "get_class_weights",
    "create_imbalanced_sampler",
    # Medical
    "get_medical_datasets",
    "download_medical_dataset",
    "list_medical_datasets",
    # Gold layer
    "GoldClassificationDataset",
    "GoldSegmentationDataset",
    # MLflow lineage tracking
    "compute_file_sha256",
    "create_gold_manifest",
    "log_medallion_dataset",
]
