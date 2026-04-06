"""
Data loading, acquisition, and transforms.
"""

from .huggingface import load_hf_dataset, list_popular_datasets
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
]
