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

def extract_mask(sam_results, h, w):
    if sam_results.masks is not None:
        mask_data = sam_results.masks.data.cpu().numpy()
        if mask_data.size > 0:
            if mask_data.ndim == 3: mask_data = mask_data[0]
            if mask_data.shape != (h, w):
                mask_data = cv2.resize(mask_data.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            return (mask_data > 0).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)

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

    for img_path in tqdm(test_images, desc="Evaluating SAM2 K-Means Prompt"):
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
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                cx, cy = x1 + (x2-x1)//2, y1 + (y2-y1)//2
                
                box_img = img[y1:y2, x1:x2]
                if box_img.size > 0:
                    pixels = box_img.reshape((-1, 3)).astype(np.float32)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    K = 3
                    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    
                    brightness = np.sum(centers, axis=1)
                    glare_idx = np.argmax(brightness)
                    
                    counts = np.bincount(labels.flatten())
                    counts[glare_idx] = 0
                    tissue_idx = np.argmax(counts)
                    
                    tissue_mask = (labels.flatten() == tissue_idx).reshape(box_img.shape[:2])
                    M = cv2.moments(tissue_mask.astype(np.uint8))
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + x1
                        cy = int(M["m01"] / M["m00"]) + y1

                # 1. Bbox Prompt
                res_box = sam_model(img, bboxes=box.tolist(), verbose=False)[0]
                m_box = extract_mask(res_box, h, w)
                
                # 2. Glare-Avoidant Point Prompt
                res_pt = sam_model(img, points=[[cx, cy]], labels=[1], verbose=False)[0]
                m_pt = extract_mask(res_pt, h, w)
                
                # Ensemble
                m_ensemble = np.logical_or(m_box, m_pt).astype(np.uint8)
                pred_mask = np.logical_or(pred_mask, m_ensemble).astype(np.uint8)

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
