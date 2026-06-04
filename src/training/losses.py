from typing import Optional
"""
Custom loss functions.

Usage:
    from src.training.losses import FocalLoss, LabelSmoothingCE
    
    criterion = FocalLoss(gamma=2.0)
    criterion = LabelSmoothingCE(smoothing=0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces loss for well-classified examples, focusing on hard examples.
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights (optional)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy with label smoothing.
    
    Prevents overconfident predictions by softening labels.
    
    Args:
        smoothing: Smoothing factor (0.0 = no smoothing, 0.1 = common choice)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(inputs)
            smooth_labels.fill_(self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for similarity learning.
    
    Args:
        margin: Margin for negative pairs
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings1, embeddings2: Pair of embeddings
            labels: 1 for similar pairs, 0 for dissimilar
        """
        distances = F.pairwise_distance(embeddings1, embeddings2)
        
        loss = labels * distances.pow(2) + \
               (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        return loss.mean()

class StructureLoss(nn.Module):
    """
    Structure Loss for polyp segmentation (Weighted BCE + Weighted IoU).
    Penalizes errors linearly more on the boundaries.
    As proposed in PraNet / PNS+ papers.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred and mask shape: [B, 1, H, W]
        # Weight map generation (higher weights at boundaries)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        
        # Weighted BCE
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # Weighted IoU
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        
        return (wbce + wiou).mean()
