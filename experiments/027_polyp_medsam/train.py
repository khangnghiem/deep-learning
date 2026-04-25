"""
=============================================================================
027_polyp_medsam — MedSAM Domain Adaptation
=============================================================================

Custom PyTorch training loop to fine-tune the SAM (Segment Anything Model)
mask decoder on polyp data. This serves as Domain Adaptation for medical imaging,
specifically targeting polyp boundary delineation.

Usage (Colab):
    %run train.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from transformers import SamModel, SamProcessor
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "datasets", "monai"])
    from transformers import SamModel, SamProcessor

import monai.losses as monai_losses

# =============================================================================
# LOCAL SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import atexit
try:
    from google.colab import runtime
    atexit.register(runtime.unassign)
except ImportError:
    pass

from src.config.paths import setup_mlflow

# =============================================================================
# DATASET
# =============================================================================
def yolo_polygon_to_mask(label_path: Path, img_h: int, img_w: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not label_path.exists():
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            coords = list(map(float, parts[1:]))
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask

class YOLOToSAMDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor):
        self.image_paths = sorted(list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png")))
        self.label_dir = Path(label_dir)
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        label_path = self.label_dir / (img_path.stem + ".txt")
        gt_mask = yolo_polygon_to_mask(label_path, h, w)

        # Extract bounding box from mask for prompt
        y_indices, x_indices = np.where(gt_mask > 0)
        if len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturb
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(w, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(h, y_max + np.random.randint(0, 20))
            prompt_box = [x_min, y_min, x_max, y_max]
        else:
            prompt_box = [0, 0, w, h] # fallback

        inputs = self.processor(image, input_boxes=[[[prompt_box]]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Add labels (resize to 256x256 to allow batch stacking and match SAM pred_masks shape)
        gt_mask_resized = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        inputs["ground_truth_mask"] = torch.tensor(gt_mask_resized, dtype=torch.float32).unsqueeze(0)
        return inputs

def collate_fn(batch):
    keys = batch[0].keys()
    collated = {k: torch.stack([x[k] for x in batch]) for k in keys}
    return collated

# =============================================================================
# MAIN
# =============================================================================
def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    mlflow = setup_mlflow()
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Model & Processor
    model_name = config["model"].get("architecture", "facebook/sam-vit-base")
    logger.info(f"Loading {model_name}...")
    try:
        processor = SamProcessor.from_pretrained(model_name)
        model = SamModel.from_pretrained(model_name).to(device)
    except Exception as e:
        logger.warning(f"Failed to load {model_name}. Fallback to facebook/sam-vit-base. Error: {e}")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

    # Freeze vision encoder and prompt encoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad = False
    
    # Optimizer
    lr = config["training"].get("learning_rate", 1e-4)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Loss: DiceCELoss (Includes boundary awareness via Dice)
    seg_loss = monai_losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # Data
    ds_yaml_path = Path(config["data"]["dataset_yaml"])
    with open(ds_yaml_path) as f:
        ds_cfg = yaml.safe_load(f)
    val_img_dir = ds_yaml_path.parent / ds_cfg.get("val", "val/images")
    val_lbl_dir = val_img_dir.parent.parent / "labels"
    # Wait, YOLO dataset separates images and labels. If val goes to images/val or val/images.
    # usually dataset.yaml has 'val: images/val' or 'val'. Let's just robustly assume:
    # 016_polyp_fast_diag_dataset directly has 'train/images', 'train/labels'
    
    train_img_dir = ds_yaml_path.parent / "train" / "images"
    train_lbl_dir = ds_yaml_path.parent / "train" / "labels"
    val_img_dir = ds_yaml_path.parent / "val" / "images"
    val_lbl_dir = ds_yaml_path.parent / "val" / "labels"
    
    if not train_img_dir.exists(): train_img_dir = ds_yaml_path.parent / "images" / "train"
    if not train_lbl_dir.exists(): train_lbl_dir = ds_yaml_path.parent / "labels" / "train"
    if not val_img_dir.exists(): val_img_dir = ds_yaml_path.parent / "images" / "val"
    if not val_lbl_dir.exists(): val_lbl_dir = ds_yaml_path.parent / "labels" / "val"

    train_ds = YOLOToSAMDataset(train_img_dir, train_lbl_dir, processor)
    val_ds = YOLOToSAMDataset(val_img_dir, val_lbl_dir, processor)
    
    batch_size = config["data"].get("batch_size", 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    epochs = config["training"].get("epochs", 10)
    
    run_name = config["mlflow"].get("run_name", f"medsam_finetune_ep{epochs}")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config["training"])
        
        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(device)
                input_boxes = batch["input_boxes"].to(device)
                gt = batch["ground_truth_mask"].to(device)

                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                
                # Output logits shape: (B, 1, 1, H, W). Squeeze the extra dim.
                pred_masks = outputs.pred_masks.squeeze(1) # now (B, 1, H, W)
                
                # We resized GT to 256x256 in __getitem__, matching pred_masks shape.
                loss = seg_loss(pred_masks, gt)
                
                # Optimize memory bandwidth and footprint
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    pixel_values = batch["pixel_values"].to(device)
                    input_boxes = batch["input_boxes"].to(device)
                    gt = batch["ground_truth_mask"].to(device)
                    
                    outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                    pred_masks = outputs.pred_masks.squeeze(1)
                    v_loss = seg_loss(pred_masks, gt)
                    val_loss += v_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            mlflow.log_metrics({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch)
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_dir = Path(config["paths"]["output_dir"]) / "weights"
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_dir / "best_medsam")
                processor.save_pretrained(save_dir / "best_medsam")
                logger.info(f"Saved new best model at {save_dir / 'best_medsam'}")

        # Save completion marker
        marker = {"success": True, "completed_at": datetime.now().isoformat()}
        Path("completed.json").write_text(json.dumps(marker, indent=2))
        logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
