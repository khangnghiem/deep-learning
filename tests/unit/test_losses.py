"""
Unit tests for loss functions.
"""

import pytest
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.losses import FocalLoss, LabelSmoothingCE, DiceLoss


class TestFocalLoss:
    """Tests for FocalLoss."""
    
    def test_loss_non_negative(self):
        """Focal loss should be non-negative."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert loss >= 0
    
    def test_perfect_prediction_low_loss(self):
        """Perfect prediction should have low loss."""
        loss_fn = FocalLoss(gamma=2.0)
        
        # Create confident correct predictions
        logits = torch.zeros(4, 3)
        logits[0, 0] = 10.0  # High confidence for class 0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        logits[3, 0] = 10.0
        targets = torch.tensor([0, 1, 2, 0])
        
        loss = loss_fn(logits, targets)
        
        assert loss < 0.1  # Very low for confident correct


class TestLabelSmoothingCE:
    """Tests for LabelSmoothingCE."""
    
    def test_loss_non_negative(self):
        """Label smoothing CE should be non-negative."""
        loss_fn = LabelSmoothingCE(smoothing=0.1)
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert loss >= 0
    
    def test_smoothing_increases_loss(self):
        """Smoothing should increase loss vs standard CE."""
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        
        smooth_loss = LabelSmoothingCE(smoothing=0.1)(logits, targets)
        no_smooth_loss = LabelSmoothingCE(smoothing=0.0)(logits, targets)
        
        # Smoothing should generally increase loss slightly
        assert smooth_loss != no_smooth_loss


class TestDiceLoss:
    """Tests for DiceLoss."""
    
    def test_loss_range(self):
        """Dice loss should be in [0, 1]."""
        loss_fn = DiceLoss()
        pred = torch.randn(4, 1, 32, 32)
        target = torch.randint(0, 2, (4, 1, 32, 32)).float()
        
        loss = loss_fn(pred, target)
        
        assert 0 <= loss <= 1
    
    def test_perfect_overlap(self):
        """Perfect overlap should have near-zero loss."""
        loss_fn = DiceLoss()
        target = torch.ones(2, 1, 16, 16)
        pred = torch.ones(2, 1, 16, 16) * 100  # High logits -> sigmoid near 1
        
        loss = loss_fn(pred, target)
        
        assert loss < 0.1
