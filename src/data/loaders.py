"""
DataLoader utilities and wrappers.

Usage:
    from src.data.loaders import create_dataloaders, get_class_weights
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from typing import Optional, Tuple
import numpy as np


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split a dataset and create train/val/test dataloaders.
    
    Args:
        dataset: Full dataset
        batch_size: Batch size
        val_split: Validation set proportion
        test_split: Test set proportion
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (faster GPU transfer)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    total = len(dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")
    
    return train_loader, val_loader, test_loader


def get_class_weights(
    dataset: Dataset,
    num_classes: int,
    label_key: str = None,
    precomputed_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        dataset: Dataset with labels
        num_classes: Number of classes
        label_key: Key to access labels if dataset returns dict
        precomputed_labels: Optional tensor of all labels, to skip iterating dataset.
    
    Returns:
        Tensor of class weights (inversely proportional to frequency)
    """
    if precomputed_labels is not None:
        class_counts = torch.bincount(precomputed_labels, minlength=num_classes).float()
    else:
        class_counts = torch.zeros(num_classes)
        for item in dataset:
            if isinstance(item, tuple):
                label = item[1]  # Assume (input, label) format
            elif isinstance(item, dict) and label_key:
                label = item[label_key]
            else:
                raise ValueError("Cannot extract label from dataset item")
            class_counts[label] += 1
    
    # Inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # Normalize
    
    return weights


def create_imbalanced_sampler(
    dataset: Dataset,
    num_classes: int,
) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a weighted sampler for imbalanced datasets.
    
    Args:
        dataset: Dataset with labels
        num_classes: Number of classes
    
    Returns:
        WeightedRandomSampler
    """
    # ⚡ Bolt Optimization: Extract labels in a single pass to prevent
    # redundant, expensive dataset iterations during sampler initialization.
    labels = []
    for item in dataset:
        if isinstance(item, tuple):
            label = item[1]
        else:
            raise ValueError("Expected (input, label) format")
        labels.append(label)

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    class_weights = get_class_weights(
        dataset,
        num_classes,
        precomputed_labels=labels_tensor
    )
    
    # ⚡ Bolt Optimization: Use advanced PyTorch tensor indexing instead of list
    # comprehensions and unidiomatic list-to-tensor conversions to efficiently
    # map sample labels to their corresponding weights.
    sample_weights = class_weights[labels_tensor]
    
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
