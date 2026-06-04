"""
Integration tests for training pipeline.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import SimpleCNN
from src.training import EarlyStopping


class TestTrainingLoop:
    """Integration tests for training."""
    
    def test_one_epoch_no_crash(self, tiny_dataset):
        """One training epoch completes without error."""
        loader = DataLoader(tiny_dataset, batch_size=8)
        
        model = SimpleCNN(num_classes=10, in_channels=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for batch in loader:
            inputs, targets = batch
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # If we get here, training didn't crash
        assert True
    
    def test_loss_decreases(self, tiny_dataset):
        """Loss should generally decrease over epochs."""
        loader = DataLoader(tiny_dataset, batch_size=8)
        
        model = SimpleCNN(num_classes=10, in_channels=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        for epoch in range(3):
            epoch_loss = 0
            for batch in loader:
                inputs, targets = batch
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
        
        # Last loss should be less than first (on tiny dataset)
        assert losses[-1] < losses[0]


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""
    
    def test_stops_after_patience(self):
        """Early stopping triggers after patience epochs."""
        early_stop = EarlyStopping(patience=3, mode="min")
        
        # Simulate improving loss
        for val in [1.0, 0.9, 0.8]:
            early_stop(val)
            assert not early_stop.should_stop
        
        # Simulate stagnation
        for val in [0.85, 0.86, 0.87]:
            early_stop(val)
        
        assert early_stop.should_stop
    
    def test_resets_on_improvement(self):
        """Counter resets when metric improves."""
        early_stop = EarlyStopping(patience=3, mode="min")
        
        early_stop(1.0)
        early_stop(0.95)  # No improvement
        early_stop(0.96)  # No improvement (counter = 2)
        early_stop(0.8)   # Improvement! (counter resets)
        
        assert early_stop.counter == 0
        assert not early_stop.should_stop
