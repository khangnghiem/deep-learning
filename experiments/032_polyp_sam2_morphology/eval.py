import os
import yaml
import glob
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO, SAM

def yolo_to_mask(label_path, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            coords = list(map(float, parts[1:]))
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    yolo_model = YOLO(config['model']['yolo_weights'])
    sam_model = SAM(config['model']['sam_weights'])

    test_images = sorted(glob.glob(os.path.join(config['data']['test_images_dir'], '*.*')))
    test_labels_dir = config['data']['test_labels_dir']
    
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    ious = []
    dices = []

    # Define morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    for img_path in tqdm(test_images, desc="Evaluating SAM2 + Morphology"):
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        label_path = os.path.join(test_labels_dir, Path(img_path).stem + ".txt")
        gt_mask = yolo_to_mask(label_path, h, w)

        # 1. YOLO predicts boxes (with TTA enabled)
        yolo_results = yolo_model(img, conf=config['evaluation']['conf_threshold'], augment=True, verbose=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if len(boxes) > 0:
            for box in boxes:
                sam_results = sam_model(img, bboxes=box.tolist(), verbose=False)[0]
                if sam_results.masks is not None:
                    mask_data = sam_results.masks.data.cpu().numpy()
                    if mask_data.ndim == 3: mask_data = mask_data[0] # Take first instance
                    if mask_data.shape != (h, w):
                        mask_data = cv2.resize(mask_data.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_mask = np.logical_or(pred_mask, mask_data > 0).astype(np.uint8)
            
            # Post-processing: Morphological Closing to fill holes & smooth edges
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Calculate metrics
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        
        if union == 0:
            iou = 1.0 # True negative
        else:
            iou = intersection / union

        # Dice score
        denom = gt_mask.sum() + pred_mask.sum()
        if denom == 0:
            dice = 1.0
        else:
            dice = 2 * intersection / denom

        ious.append(iou)
        dices.append(dice)
        
    mIoU = np.mean(ious)
    mDice = np.mean(dices)
    print(f"Testing completed.")
    print(f"mIoU: {mIoU:.4f}")
    print(f"mDice: {mDice:.4f}")
    
    # Save results
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"mIoU: {mIoU:.4f}\n")
        f.write(f"mDice: {mDice:.4f}\n")

if __name__ == "__main__":
    main()
