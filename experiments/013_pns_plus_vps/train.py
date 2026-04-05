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
import segmentation_models_pytorch as smp

# Adjust path to find src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.training.losses import StructureLoss
from shared_config.paths import setup_mlflow

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
    
    dice = (2. * inter + 1e-5) / (union + 1e-5)
    iou = (inter + 1e-5) / (union - inter + 1e-5)
    
    return dice.mean().item(), iou.mean().item()

def get_data_splits(val_ratio=0.2, seed=42):
    """Collects Kvasir and CVC dataset images and creates an 80/20 train/val split."""
    kvasir_images_dir = '/content/drive/MyDrive/data_lake/01_bronze_medical/kvasir-seg/Kvasir-SEG/Kvasir-SEG/images'
    kvasir_masks_dir = '/content/drive/MyDrive/data_lake/01_bronze_medical/kvasir-seg/Kvasir-SEG/Kvasir-SEG/masks'
    
    cvc_images_dir = '/content/drive/MyDrive/data_lake/01_bronze_medical/cvc_datasetninja/images'
    cvc_masks_dir = '/content/drive/MyDrive/data_lake/01_bronze_medical/cvc_datasetninja/masks'
    
    all_images = []
    all_masks = []
    
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
                    all_images.append(os.path.join(kvasir_images_dir, img_name))
                    all_masks.append(os.path.join(kvasir_masks_dir, mask_name))
                    
    # CVC Ninja
    if os.path.exists(cvc_images_dir):
        for img_name in sorted(os.listdir(cvc_images_dir)):
            if img_name.endswith(('.jpg', '.png')):
                if os.path.exists(os.path.join(cvc_masks_dir, img_name)):
                    all_images.append(os.path.join(cvc_images_dir, img_name))
                    all_masks.append(os.path.join(cvc_masks_dir, img_name))

    if len(all_images) == 0:
        logger.error("No data found! Check if Google Drive is mounted and paths are correct.")
        return [], [], [], []

    np.random.seed(seed)
    indices = np.random.permutation(len(all_images))
    split = int((1.0 - val_ratio) * len(all_images))
    
    train_imgs = [all_images[i] for i in indices[:split]]
    train_masks = [all_masks[i] for i in indices[:split]]
    val_imgs = [all_images[i] for i in indices[split:]]
    val_masks = [all_masks[i] for i in indices[split:]]
    
    logger.info(f"Loaded {len(train_imgs)} train images and {len(val_imgs)} val images.")
    return train_imgs, train_masks, val_imgs, val_masks

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Use a robust U-Net from SMP rather than the mock PNS+
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['training'].get('learning_rate', 1e-4)), 
        weight_decay=float(config['training'].get('weight_decay', 1e-4))
    )
    criterion = StructureLoss().to(device)

    # Albumentations Training Pipeline
    train_transform = A.Compose([
        A.Resize(352, 352),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(352, 352),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    train_imgs, train_masks, val_imgs, val_masks = get_data_splits()
    
    if len(train_imgs) == 0:
        return
        
    train_dataset = SingleImagePolypDataset(train_imgs, train_masks, transform=train_transform)
    val_dataset = SingleImagePolypDataset(val_imgs, val_masks, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training'].get('batch_size', 16), shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['training'].get('batch_size', 16), shuffle=False, num_workers=2)

    epochs = config['training'].get('epochs', 15)
    
    mlflow_config = setup_mlflow()
    mlflow_config.set_experiment(config['mlflow'].get('experiment_name', '013_pns_plus_vps'))
    
    best_dice = 0.0
    os.makedirs(os.path.join(os.path.dirname(__file__), 'checkpoints'), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pt')
    
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
                preds = model(images)
                
                loss = criterion(preds, masks)
                loss.backward()
                optimizer.step()
                
                dice, _ = calculate_metrics(preds, masks)
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
                    
                    preds = model(images)
                    loss = criterion(preds, masks)
                    
                    dice, iou = calculate_metrics(preds, masks)
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
            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved new best model with Val Dice: {best_dice:.4f}")
            
        mlflow_config.log_artifact(model_path)
        logger.info(f"Training complete. Best Val Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
