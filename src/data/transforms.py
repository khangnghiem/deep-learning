"""
Common image transforms and augmentation pipelines.

Usage:
    from src.data.transforms import get_train_transforms, get_val_transforms
    
    train_transform = get_train_transforms(image_size=224)
    val_transform = get_val_transforms(image_size=224)
"""

import torch
from torchvision import transforms


def get_train_transforms(
    image_size: int = 224,
    normalize: bool = True,
    augmentation: str = "standard"
) -> transforms.Compose:
    """
    Get training transforms with augmentation.
    
    Args:
        image_size: Target size for images
        normalize: Whether to normalize with ImageNet stats
        augmentation: Level of augmentation ("none", "standard", "heavy")
    
    Returns:
        Composed transform
    """
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Augmentation
    if augmentation == "standard":
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    elif augmentation == "heavy":
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
        ])
    
    # To tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return transforms.Compose(transform_list)


def get_val_transforms(
    image_size: int = 224,
    normalize: bool = True
) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target size for images
        normalize: Whether to normalize with ImageNet stats
    
    Returns:
        Composed transform
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return transforms.Compose(transform_list)


def get_cifar_transforms(train: bool = True) -> transforms.Compose:
    """CIFAR-10/100 specific transforms (32x32 images)."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])


def get_mnist_transforms(train: bool = True) -> transforms.Compose:
    """MNIST/FashionMNIST specific transforms (28x28 grayscale)."""
    if train:
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
