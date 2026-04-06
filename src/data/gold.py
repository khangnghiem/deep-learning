import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

# We assume this script lives in g:/My Drive/repos/deep-learning/src/data
# so we locate GOLD relative to this, or use the centralized paths module.
from src.config.paths import GOLD

class GoldClassificationDataset(ImageFolder):
    """
    Standardizes Classification data loading from the Gold layer.
    Inherits from torchvision's ImageFolder.
    """
    def __init__(self, experiment_name: str, split: str = "train", transform=None, target_transform=None):
        gold_dir = str(GOLD / experiment_name / split)
        if not os.path.exists(gold_dir):
            raise FileNotFoundError(f"Gold data not found for {experiment_name} split {split}. "
                                    f"Run ingest.ipynb first to generate {gold_dir}")
            
        super().__init__(root=gold_dir, transform=transform, target_transform=target_transform)

class GoldSegmentationDataset(Dataset):
    """
    Standardizes Segmentation data loading from the Gold layer.
    Expects structure:
      03_gold/experiment_name/split/images/...
      03_gold/experiment_name/split/masks/...
    """
    def __init__(self, experiment_name: str, split: str = "train", transform=None):
        self.gold_dir = GOLD / experiment_name / split
        if not self.gold_dir.exists():
            raise FileNotFoundError(f"Gold data not found for {experiment_name} split {split}. "
                                    f"Run ingest.ipynb first to generate {self.gold_dir}")
                                    
        self.img_dir = self.gold_dir / "images"
        self.mask_dir = self.gold_dir / "masks"
        
        # Sort to ensure matching order
        self.images = sorted(list(self.img_dir.glob("*.*")))
        self.masks = sorted(list(self.mask_dir.glob("*.*")))
        
        self.transform = transform
        
        if len(self.images) != len(self.masks):
            print(f"⚠️ Warning: Mismatch in number of images ({len(self.images)}) "
                  f"and masks ({len(self.masks)}) in {self.gold_dir}")

    def __len__(self):
        return min(len(self.images), len(self.masks))
        
    def __getitem__(self, idx):
        # We read as numpy arrays so typical albumentations transforms work well
        img_path = str(self.images[idx])
        mask_path = str(self.masks[idx])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        if self.transform:
            # Check if it's an albumentations transform (dict return)
            # or a standard torchvision transform
            try:
                # Albumentations signature check (roughly)
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            except TypeError:
                # Fallback to standard transforms
                # Usually standard transforms take PIL images
                image = Image.fromarray(image)
                mask = Image.fromarray(mask)
                if hasattr(self.transform, '__call__'):
                    # Custom paired transform
                    image, mask = self.transform(image, mask)
                    
        # Ensure mask is a tensor if it isn't already, e.g. albumentations with ToTensorV2
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32) / 255.0
        else:
            if mask.max() > 1.5:  # e.g., max is 255
                mask = mask / 255.0
                
        # Binarize
        mask = (mask > 0.5).float()
        
        # Add channel dimension if 2D
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask
