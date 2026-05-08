import numpy as np
import torch
import cv2
import os
import glob
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from peft import PeftModel

def yolo_to_mask(txt_path, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(txt_path): return mask
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) > 5:
                pts = np.array(parts[1:]).reshape(-1, 2)
                pts[:, 0] *= w; pts[:, 1] *= h
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask

def main():
    test_dir = "/content/drive/MyDrive/data_lake/03_gold/016_polyp_fast_diag_dataset/test"
    img_paths = sorted(glob.glob(os.path.join(test_dir, "images", "*.*")))
    
    yolo_model = YOLO("/content/drive/MyDrive/models/trained/025_polyp_yolov12x_aug/train/weights/best.pt")
    
    sam2_checkpoint = "sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    
    # Load LoRA adapter
    sam2_model = PeftModel.from_pretrained(sam2_model, "weights/lora_finetuned")
    sam2_model.eval()
    
    predictor = SAM2ImagePredictor(sam2_model)
    
    ious, dices = [], []
    for img_path in tqdm(img_paths, desc="Evaluating"):
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        gt_mask = yolo_to_mask(os.path.join(test_dir, "labels", Path(img_path).stem + ".txt"), h, w)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            results = yolo_model(img, conf=0.25, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
        
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if len(boxes) > 0:
            with torch.no_grad():
                predictor.set_image(img_rgb)
                masks, _, _ = predictor.predict(box=boxes, multimask_output=False)
            for m in masks:
                pred_mask = np.logical_or(pred_mask, m[0]).astype(np.uint8)
                
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        ious.append(1.0 if union == 0 else intersection / union)
        denom = gt_mask.sum() + pred_mask.sum()
        dices.append(1.0 if denom == 0 else 2 * intersection / denom)
        
    mIoU, mDice = np.mean(ious), np.mean(dices)
    print(f"mIoU: {mIoU:.4f}\nmDice: {mDice:.4f}")
    
    out_dir = Path("/content/drive/MyDrive/models/trained") / Path(os.getcwd()).name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"mIoU: {mIoU:.4f}\nmDice: {mDice:.4f}\n")

if __name__ == "__main__": main()
