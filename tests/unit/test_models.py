"""
Unit tests for model architectures.

Run with: pytest tests/unit/test_models.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import SimpleCNN, MLP, get_pretrained_resnet
from src.models.medical import MedicalCNN, get_medical_resnet, GeneExpressionMLP, UNet


class TestSimpleCNN:
    """Tests for SimpleCNN."""
    
    def test_forward_pass_shape(self, sample_image_batch):
        """Model output shape matches num_classes."""
        model = SimpleCNN(num_classes=10)
        output = model(sample_image_batch)
        
        assert output.shape == (4, 10)
    
    def test_different_num_classes(self):
        """Model works with different class counts."""
        for num_classes in [2, 5, 100]:
            model = SimpleCNN(num_classes=num_classes)
            x = torch.randn(2, 3, 32, 32)
            output = model(x)
            
            assert output.shape == (2, num_classes)


class TestMLP:
    """Tests for MLP."""
    
    def test_forward_pass_shape(self):
        """MLP outputs correct shape."""
        model = MLP(input_dim=784, num_classes=10)
        x = torch.randn(4, 784)
        output = model(x)
        
        assert output.shape == (4, 10)
    
    def test_flattens_images(self):
        """MLP can flatten image input."""
        model = MLP(input_dim=784, num_classes=10)
        x = torch.randn(4, 1, 28, 28)  # Image shape
        output = model(x)
        
        assert output.shape == (4, 10)


class TestPretrainedResNet:
    """Tests for pretrained ResNet."""
    
    def test_resnet18_shapes(self, sample_image_batch):
        """ResNet18 produces correct output shape."""
        model = get_pretrained_resnet("resnet18", num_classes=10, pretrained=False)
        output = model(sample_image_batch)
        
        assert output.shape == (4, 10)
    
    def test_resnet50_shapes(self, sample_image_batch):
        """ResNet50 produces correct output shape."""
        model = get_pretrained_resnet("resnet50", num_classes=10, pretrained=False)
        output = model(sample_image_batch)
        
        assert output.shape == (4, 10)


class TestMedicalCNN:
    """Tests for MedicalCNN."""
    
    def test_grayscale_input(self, sample_grayscale_batch):
        """MedicalCNN handles grayscale input."""
        model = MedicalCNN(num_classes=2, in_channels=1)
        output = model(sample_grayscale_batch)
        
        assert output.shape == (4, 2)
    
    def test_rgb_input(self, sample_image_batch):
        """MedicalCNN handles RGB input."""
        model = MedicalCNN(num_classes=4, in_channels=3)
        output = model(sample_image_batch)
        
        assert output.shape == (4, 4)


class TestMedicalResNet:
    """Tests for medical ResNet."""
    
    def test_grayscale_resnet(self, sample_grayscale_batch):
        """Medical ResNet handles grayscale."""
        model = get_medical_resnet("resnet18", num_classes=2, in_channels=1, pretrained=False)
        output = model(sample_grayscale_batch)
        
        assert output.shape == (4, 2)


class TestGeneExpressionMLP:
    """Tests for genomics MLP."""
    
    def test_high_dimensional_input(self, sample_gene_expression):
        """GeneExpressionMLP handles high-dimensional input."""
        model = GeneExpressionMLP(input_dim=20000, num_classes=5)
        output = model(sample_gene_expression)
        
        assert output.shape == (8, 5)


class TestUNet:
    """Tests for U-Net segmentation."""
    
    def test_segmentation_output_shape(self):
        """U-Net output matches input spatial dimensions."""
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, 256, 256)
        output = model(x)
        
        assert output.shape == (2, 1, 256, 256)
    
    def test_multiclass_segmentation(self):
        """U-Net handles multi-class segmentation."""
        model = UNet(in_channels=3, out_channels=4)
        x = torch.randn(2, 3, 128, 128)
        output = model(x)
        
        assert output.shape == (2, 4, 128, 128)
