import os, yaml, glob
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm
from ultralytics import YOLO, SAM

def yolo_to_mask(label_path, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(label_path): return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            coords = list(map(float, parts[1:]))
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= w; pts[:, 1] *= h
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    yolo_model = YOLO(config['model']['yolo_weights'])
    sam_model = SAM(config['model']['sam_weights'])
    test_images = sorted(glob.glob(os.path.join(config['data']['test_images_dir'], '*.*')))
    output_dir = Path(config['paths']['output_dir']); output_dir.mkdir(parents=True, exist_ok=True)
    ious, dices = [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for img_path in tqdm(test_images, desc="Evaluating SAM2 Bilateral Morph"):
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        # Apply bilateral filter to image before SAM2 inference
        img_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        gt_mask = yolo_to_mask(os.path.join(config['data']['test_labels_dir'], Path(img_path).stem + ".txt"), h, w)
        # YOLO on original image (TTA), SAM2 on bilateral-filtered image
        yolo_results = yolo_model(img, conf=config['evaluation']['conf_threshold'], augment=True, verbose=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if len(boxes) > 0:
            for box in boxes:
                res = sam_model(img_filtered, bboxes=box.tolist(), verbose=False)[0]
                if res.masks is not None:
                    md = res.masks.data.cpu().numpy()
                    if md.size > 0:
                        if md.ndim == 3: md = md[0]
                        if md.shape != (h, w): md = cv2.resize(md.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                        pred_mask = np.logical_or(pred_mask, md > 0).astype(np.uint8)
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        ious.append(1.0 if union == 0 else intersection / union)
        denom = gt_mask.sum() + pred_mask.sum()
        dices.append(1.0 if denom == 0 else 2 * intersection / denom)
    mIoU, mDice = np.mean(ious), np.mean(dices)
    print(f"Testing completed.\nmIoU: {mIoU:.4f}\nmDice: {mDice:.4f}")
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"mIoU: {mIoU:.4f}\nmDice: {mDice:.4f}\n")

if __name__ == "__main__": main()
