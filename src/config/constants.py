"""
Global constants for the repository.

Usage:
    from src.config.constants import EPSILON, IMAGENET_MEAN, IMAGENET_STD
"""

# Numerical stability
EPSILON = 1e-6

# ImageNet Normalization Statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CIFAR Normalization Statistics
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]

# MNIST Normalization Statistics
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
