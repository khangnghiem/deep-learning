"""
Visualization utilities.

Usage:
    from src.utils.visualization import plot_training_history, show_predictions
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional
from pathlib import Path


def plot_training_history(
    history: dict,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 4)
) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dict with train_loss, train_acc, val_loss, val_acc lists
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()


def show_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: Optional[list[str]] = None,
    n_samples: int = 16,
    figsize: tuple = (12, 12),
    save_path: Optional[Path] = None
) -> None:
    """
    Show a grid of images with true and predicted labels.
    
    Args:
        images: Tensor of images (N, C, H, W)
        labels: True labels
        predictions: Predicted labels
        class_names: Optional list of class names
        n_samples: Number of samples to show
        figsize: Figure size
        save_path: Optional path to save figure
    """
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_samples):
        img = images[i].cpu().numpy()
        
        # Handle grayscale vs RGB
        if img.shape[0] == 1:
            img = img.squeeze(0)
            cmap = "gray"
        else:
            img = np.transpose(img, (1, 2, 0))
            # Denormalize if needed
            img = (img - img.min()) / (img.max() - img.min())
            cmap = None
        
        axes[i].imshow(img, cmap=cmap)
        
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        
        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
            title = f"True: {true_name}\nPred: {pred_name}"
        else:
            title = f"True: {true_label}\nPred: {pred_label}"
        
        color = "green" if true_label == pred_label else "red"
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis("off")
    
    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list[str]] = None,
    figsize: tuple = (10, 8),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (N x N)
        class_names: Optional list of class names
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, cmap="Blues")
    
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color)
    
    plt.colorbar(im)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()
