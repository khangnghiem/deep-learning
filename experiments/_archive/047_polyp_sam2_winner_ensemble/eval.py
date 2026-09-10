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

def extract_mask(sam_results, h, w):
    if sam_results.masks is not None:
        md = sam_results.masks.data.cpu().numpy()
        if md.size > 0:
            if md.ndim == 3: md = md[0]
            if md.shape != (h, w): md = cv2.resize(md.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            return (md > 0).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    yolo_model = YOLO(config['model']['yolo_weights'])
    sam_model = SAM(config['model']['sam_weights'])
    test_images = sorted(glob.glob(os.path.join(config['data']['test_images_dir'], '*.*')))
    output_dir = Path(config['paths']['output_dir']); output_dir.mkdir(parents=True, exist_ok=True)
    ious, dices = [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    for img_path in tqdm(test_images, desc="Evaluating SAM2 Winner Ensemble"):
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        gt_mask = yolo_to_mask(os.path.join(config['data']['test_labels_dir'], Path(img_path).stem + ".txt"), h, w)

        yolo_results = yolo_model(img, conf=config['evaluation']['conf_threshold'], augment=True, verbose=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()

        # Pipeline A: 032-style (bbox -> SAM2 -> morphology)
        mask_a = np.zeros((h, w), dtype=np.uint8)
        # Pipeline B: 044-style (bbox + K-Means glare-free point -> SAM2 -> morphology)
        mask_b = np.zeros((h, w), dtype=np.uint8)

        if len(boxes) > 0:
            for box in boxes:
                # --- Pipeline A: Standard bbox prompt ---
                mask_a = np.logical_or(mask_a, extract_mask(
                    sam_model(img, bboxes=box.tolist(), verbose=False)[0], h, w)).astype(np.uint8)

                # --- Pipeline B: K-Means glare-avoidant prompt ---
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cx, cy = x1 + (x2-x1)//2, y1 + (y2-y1)//2
                box_img = img[y1:y2, x1:x2]
                if box_img.size > 0:
                    pixels = box_img.reshape((-1, 3)).astype(np.float32)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    glare_idx = np.argmax(np.sum(centers, axis=1))
                    counts = np.bincount(labels.flatten()); counts[glare_idx] = 0
                    tissue_idx = np.argmax(counts)
                    tissue_mask = (labels.flatten() == tissue_idx).reshape(box_img.shape[:2])
                    M = cv2.moments(tissue_mask.astype(np.uint8))
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + x1
                        cy = int(M["m01"] / M["m00"]) + y1
                m_box = extract_mask(sam_model(img, bboxes=box.tolist(), verbose=False)[0], h, w)
                m_pt = extract_mask(sam_model(img, points=[[cx, cy]], labels=[1], verbose=False)[0], h, w)
                mask_b = np.logical_or(mask_b, np.logical_or(m_box, m_pt)).astype(np.uint8)

            mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_CLOSE, kernel)
            mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, kernel)

        # Winner ensemble: UNION of both pipelines
        pred_mask = np.logical_or(mask_a, mask_b).astype(np.uint8)

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
