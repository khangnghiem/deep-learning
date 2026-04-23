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

def get_feedback_points(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
        
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return []
        
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    pts = [extLeft, extRight, extTop, extBot]
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        pts.append((cX, cY))
        
    return [list(p) for p in pts]

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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    for img_path in tqdm(test_images, desc="Evaluating SAM2 Iterative"):
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        label_path = os.path.join(test_labels_dir, Path(img_path).stem + ".txt")
        gt_mask = yolo_to_mask(label_path, h, w)

        yolo_results = yolo_model(img, conf=config['evaluation']['conf_threshold'], augment=True, verbose=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if len(boxes) > 0:
            for box in boxes:
                # PASS 1
                res1 = sam_model(img, bboxes=box.tolist(), verbose=False)[0]
                mask1 = np.zeros((h, w), dtype=np.uint8)
                if res1.masks is not None:
                    md1 = res1.masks.data.cpu().numpy()
                    if md1.size > 0:
                        if md1.ndim == 3: md1 = md1[0]
                        if md1.shape != (h, w):
                            md1 = cv2.resize(md1.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                        mask1 = (md1 > 0).astype(np.uint8)
                
                # Extract feedback points
                feedback_pts = get_feedback_points(mask1)
                
                # PASS 2 (Iterative refinement)
                if len(feedback_pts) > 0:
                    labels = [1] * len(feedback_pts)
                    res2 = sam_model(img, bboxes=box.tolist(), points=feedback_pts, labels=labels, verbose=False)[0]
                    if res2.masks is not None:
                        md2 = res2.masks.data.cpu().numpy()
                        if md2.size > 0:
                            if md2.ndim == 3: md2 = md2[0]
                            if md2.shape != (h, w):
                                md2 = cv2.resize(md2.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                            mask1 = (md2 > 0).astype(np.uint8)
                
                pred_mask = np.logical_or(pred_mask, mask1).astype(np.uint8)
            
            # Post-processing
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = 1.0 if union == 0 else intersection / union

        denom = gt_mask.sum() + pred_mask.sum()
        dice = 1.0 if denom == 0 else 2 * intersection / denom

        ious.append(iou)
        dices.append(dice)
        
    mIoU = np.mean(ious)
    mDice = np.mean(dices)
    print(f"Testing completed.")
    print(f"mIoU: {mIoU:.4f}")
    print(f"mDice: {mDice:.4f}")
    
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"mIoU: {mIoU:.4f}\n")
        f.write(f"mDice: {mDice:.4f}\n")

if __name__ == "__main__":
    main()
