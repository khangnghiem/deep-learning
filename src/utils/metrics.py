"""
Custom metrics for evaluation.

Usage:
    from src.utils.metrics import accuracy, precision_recall_f1
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate accuracy."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return accuracy_score(y_true, y_pred)


def precision_recall_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    average: str = "weighted"
) -> tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'micro', 'macro', 'weighted', or None
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    return precision, recall, f1


def get_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor
) -> np.ndarray:
    """Get confusion matrix."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return confusion_matrix(y_true, y_pred)


def top_k_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        outputs: Model outputs (logits or probabilities)
        targets: True labels
        k: Top-k value
    
    Returns:
        Top-k accuracy
    """
    with torch.no_grad():
        _, top_k_preds = outputs.topk(k, dim=1)
        correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
        return correct.any(dim=1).float().mean().item()
