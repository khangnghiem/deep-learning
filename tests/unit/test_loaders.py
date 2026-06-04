import pytest
import torch
from torch.utils.data import Dataset
from src.data.loaders import create_imbalanced_sampler, get_class_weights

class DummyDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels
        self.access_count = 0
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        self.access_count += 1
        return torch.zeros(1), self.labels[idx]

def test_single_pass():
    dataset = DummyDataset([0, 1, 1, 2, 2, 2])
    sampler = create_imbalanced_sampler(dataset, 3)
    assert dataset.access_count == 6
