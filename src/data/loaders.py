"""
DataLoader utilities and wrappers.

Usage:
    from src.data.loaders import create_dataloaders, get_class_weights
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional, Tuple


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
    dataset: Optional[Dataset] = None,
    num_classes: int = None,
    label_key: str = None,
    *,
    labels: Optional[list] = None,
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        dataset: Dataset with labels
        num_classes: Number of classes
        label_key: Key to access labels if dataset returns dict
        labels: Optional precomputed list of labels to avoid iterating dataset
    
    Returns:
        Tensor of class weights (inversely proportional to frequency)
    """
    if num_classes is None:
        raise ValueError("num_classes must be provided")

    class_counts = torch.zeros(num_classes)
    
    if labels is not None:
        for label in labels:
            class_counts[int(label)] += 1
    elif dataset is not None:
        for item in dataset:
            if isinstance(item, tuple):
                label = item[1]  # Assume (input, label) format
            elif isinstance(item, dict) and label_key:
                label = item[label_key]
            else:
                raise ValueError("Cannot extract label from dataset item")

            class_counts[int(label)] += 1
    else:
        raise ValueError("Must provide either dataset or labels")
    
    # Inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # Normalize
    
    return weights


def create_imbalanced_sampler(
    dataset: Dataset,
    num_classes: int,
    label_key: str = None,
) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a weighted sampler for imbalanced datasets.
    
    Args:
        dataset: Dataset with labels
        num_classes: Number of classes
        label_key: Key to access labels if dataset returns dict
    
    Returns:
        WeightedRandomSampler
    """
    # Extract labels in a single pass to avoid redundant dataset iteration
    labels = []
    for item in dataset:
        if isinstance(item, tuple):
            label = item[1]
        elif isinstance(item, dict) and label_key:
            label = item[label_key]
        else:
            raise ValueError("Expected (input, label) format or dict with label_key")
        labels.append(label)

    # Pass precomputed labels to avoid second dataset iteration
    class_weights = get_class_weights(num_classes=num_classes, labels=labels)
    
    # Efficiently assign sample weights using advanced tensor indexing
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    sample_weights = class_weights[labels_tensor]
    
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
