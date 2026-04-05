"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_image_batch():
    """Create a sample batch of images (B, C, H, W)."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_grayscale_batch():
    """Create a sample batch of grayscale images."""
    return torch.randn(4, 1, 224, 224)


@pytest.fixture
def sample_labels():
    """Create sample labels for 10-class classification."""
    return torch.randint(0, 10, (4,))


@pytest.fixture
def sample_gene_expression():
    """Create sample gene expression data (B, num_genes)."""
    return torch.randn(8, 20000)


@pytest.fixture
def tiny_dataset():
    """Create a tiny dataset for integration tests."""
    from torch.utils.data import TensorDataset
    
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    
    return TensorDataset(images, labels)
