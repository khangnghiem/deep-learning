"""
Shared constants for ML repos.

Usage:
    from src.config.constants import EPSILON, IMAGENET_MEAN, IMAGENET_STD
"""

# Numerical Stability
EPSILON = 1e-6

# Dataset Statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
