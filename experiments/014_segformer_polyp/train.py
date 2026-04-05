import os
import sys
import yaml
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation

# Adjust path to find src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.training.losses import StructureLoss
from src.training import WarmupCosineScheduler, EarlyStopping, ModelCheckpoint
from shared_config.paths import setup_mlflow

# Auto-release Colab GPU runtime when script finishes
import atexit
try:
    from google.colab import runtime
    atexit.register(runtime.unassign)
except ImportError:
    pass  # Not in Colab — no-op

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SingleImagePolypDataset(Dataset):
    """Dataset for training single-frame segmentation using real medical datasets."""
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Read as NumPy arrays for Albumentations
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = ToTensorV2()(image=image)['image'] / 255.0
            mask = ToTensorV2()(image=mask)['mask'] / 255.0
            
        mask = (mask > 0.5).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask

def calculate_metrics(pred, mask, threshold=0.5):
    """Calculates Dice and IoU (batch average)."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    inter = (pred_bin * mask).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    
    dice = (2. * inter) / (union + 1e-5)
    iou = inter / (union - inter + 1e-5)
    
    return dice.mean().item(), iou.mean().item()

def get_data_splits(val_ratio=0.2, seed=42):
    """Collects Kvasir for train/val and uses CVC for test."""
    from shared_config.paths import BRONZE_MEDICAL
    kvasir_images_dir = str(BRONZE_MEDICAL / 'kvasir-seg/Kvasir-SEG/Kvasir-SEG/images')
    kvasir_masks_dir = str(BRONZE_MEDICAL / 'kvasir-seg/Kvasir-SEG/Kvasir-SEG/masks')
    
    cvc_images_dir = str(BRONZE_MEDICAL / 'cvc_datasetninja/images')
    cvc_masks_dir = str(BRONZE_MEDICAL / 'cvc_datasetninja/masks')
    
    kvasir_images = []
    kvasir_masks = []
    cvc_images = []
    cvc_masks = []
    
    # Kvasir
    if os.path.exists(kvasir_images_dir):
        for img_name in sorted(os.listdir(kvasir_images_dir)):
            if img_name.endswith(('.jpg', '.png')):
                mask_candidate_1 = img_name.replace('.jpg', '.png')
                mask_candidate_2 = img_name.replace('.png', '.jpg')
                mask_name = img_name
                
                if os.path.exists(os.path.join(kvasir_masks_dir, mask_candidate_1)):
                    mask_name = mask_candidate_1
                elif os.path.exists(os.path.join(kvasir_masks_dir, mask_candidate_2)):
                    mask_name = mask_candidate_2

                if os.path.exists(os.path.join(kvasir_masks_dir, mask_name)):
                    kvasir_images.append(os.path.join(kvasir_images_dir, img_name))
                    kvasir_masks.append(os.path.join(kvasir_masks_dir, mask_name))
                    
    # CVC Ninja
    if os.path.exists(cvc_images_dir):
        for img_name in sorted(os.listdir(cvc_images_dir)):
            if img_name.endswith(('.jpg', '.png')):
                if os.path.exists(os.path.join(cvc_masks_dir, img_name)):
                    cvc_images.append(os.path.join(cvc_images_dir, img_name))
                    cvc_masks.append(os.path.join(cvc_masks_dir, img_name))

    if len(kvasir_images) == 0:
        logger.error("No data found! Check if Google Drive is mounted and paths are correct.")
        return [], [], [], [], [], []

    np.random.seed(seed)
    indices = np.random.permutation(len(kvasir_images))
    split = int((1.0 - val_ratio) * len(kvasir_images))
    
    train_imgs = [kvasir_images[i] for i in indices[:split]]
    train_masks = [kvasir_masks[i] for i in indices[:split]]
    val_imgs = [kvasir_images[i] for i in indices[split:]]
    val_masks = [kvasir_masks[i] for i in indices[split:]]
    test_imgs = cvc_images
    test_masks = cvc_masks
    
    logger.info(f"Loaded {len(train_imgs)} train images, {len(val_imgs)} val images, and {len(test_imgs)} test images.")
    return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Use SOTA SegFormer-B2 from HuggingFace
    # SegFormer returns logits of shape (batch, num_labels, H/4, W/4).
    # We will interpolate them to original size before loss calculation.
    model = SegformerForSemanticSegmentation.from_pretrained(
        config['model'].get('architecture', 'nvidia/mit-b2'),
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['training'].get('learning_rate', 6e-5)), 
        weight_decay=float(config['training'].get('weight_decay', 1e-4))
    )
    criterion = StructureLoss().to(device)

    # Albumentations Training Pipeline with Stronger Augmentations (Session 3)
    train_transform = A.Compose([
        A.Resize(352, 352),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Stronger spatial augmentations
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(352, 352),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks = get_data_splits()
    
    if len(train_imgs) == 0:
        return
        
    train_dataset = SingleImagePolypDataset(train_imgs, train_masks, transform=train_transform)
    val_dataset = SingleImagePolypDataset(val_imgs, val_masks, transform=val_transform)
    if len(test_imgs) > 0:
        test_dataset = SingleImagePolypDataset(test_imgs, test_masks, transform=val_transform)
    else:
        test_dataset = val_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=config['training'].get('batch_size', 16), shuffle=True, num_workers=config['data'].get('num_workers', 2))
    val_loader = DataLoader(val_dataset, batch_size=config['training'].get('batch_size', 16), shuffle=False, num_workers=config['data'].get('num_workers', 2))
    test_loader = DataLoader(test_dataset, batch_size=config['training'].get('batch_size', 16), shuffle=False, num_workers=config['data'].get('num_workers', 2))

    epochs = config['training'].get('epochs', 25)
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=2,
        total_epochs=epochs,
        min_lr=1e-6
    )

    early_stopper = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 10),
        mode='max'
    )
    
    # Checkpointing
    from shared_config.paths import REPOS
    chkpt_dir = str(REPOS / 'deep-learning/experiments/014_segformer_polyp/checkpoints')
    os.makedirs(chkpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        save_dir=chkpt_dir,
        monitor='val_dice',
        mode='max',
        save_last=True
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    mlflow_config = setup_mlflow()
    mlflow_config.set_experiment(config['mlflow'].get('experiment_name', '014_segformer_polyp'))
    
    with mlflow_config.start_run():
        mlflow_config.log_params(config['training'])
        
        for epoch in range(epochs):
            # --- Training Phase ---
            model.train()
            train_loss = 0.0
            train_dice = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    # Segformer outputs logits at 1/4 resolution, so interpolate:
                    logits = nn.functional.interpolate(
                        outputs.logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    loss = criterion(logits, masks)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                dice, _ = calculate_metrics(logits, masks)
                train_loss += loss.item()
                train_dice += dice
                
                pbar.set_postfix({'loss': loss.item(), 'dice': dice})
                
            avg_train_loss = train_loss / len(train_loader)
            avg_train_dice = train_dice / len(train_loader)
            
            # --- Validation Phase ---
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            val_iou = 0.0
            
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for images, masks in pbar_val:
                    images, masks = images.to(device), masks.to(device)
                    
                    outputs = model(images)
                    logits = nn.functional.interpolate(
                        outputs.logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    
                    loss = criterion(logits, masks)
                    
                    dice, iou = calculate_metrics(logits, masks)
                    val_loss += loss.item()
                    val_dice += dice
                    val_iou += iou
                    pbar_val.set_postfix({'val_loss': loss.item(), 'val_dice': dice})
                    
            avg_val_loss = val_loss / len(val_loader)
            avg_val_dice = val_dice / len(val_loader)
            avg_val_iou = val_iou / len(val_loader)
            
            mlflow_config.log_metrics({
                'train_loss': avg_train_loss,
                'train_dice': avg_train_dice,
                'val_loss': avg_val_loss,
                'val_dice': avg_val_dice,
                'val_iou': avg_val_iou
            }, step=epoch)
            
            logger.info(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val IoU: {avg_val_iou:.4f}")
            
            # Save Best Model
            is_best = checkpoint_callback(
                model, 
                optimizer, 
                epoch, 
                {'val_dice': avg_val_dice, 'val_loss': avg_val_loss}, 
                scheduler=scheduler
            )
            if is_best:
                logger.info(f"Saved new best model with Val Dice: {avg_val_dice:.4f}")
            
            scheduler.step()
            mlflow_config.log_metrics({'lr': scheduler.get_last_lr()[0]}, step=epoch)

            if early_stopper(avg_val_dice):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        # --- Testing Phase ---
        if checkpoint_callback.best_path and os.path.exists(checkpoint_callback.best_path):
            checkpoint = torch.load(checkpoint_callback.best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best model for testing.")

        model.eval()
        test_dice, test_iou = 0.0, 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                logits = nn.functional.interpolate(
                    outputs.logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                dice, iou = calculate_metrics(logits, masks)
                test_dice += dice
                test_iou += iou
        avg_test_dice = test_dice / len(test_loader)
        avg_test_iou = test_iou / len(test_loader)
        mlflow_config.log_metrics({'test_dice': avg_test_dice, 'test_iou': avg_test_iou})
        logger.info(f"Test Dice: {avg_test_dice:.4f}, Test IoU: {avg_test_iou:.4f}")
        
        mlflow_config.log_artifact(str(checkpoint_callback.best_path))
        logger.info(f"Training complete. Best Val Dice: {checkpoint_callback.best_score:.4f}")

if __name__ == "__main__":
    main()
