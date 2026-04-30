import os, glob
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm
from ultralytics import YOLO, SAM
import yaml

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
    
    for img_path in tqdm(test_images, desc="SAM2 with CLAHE Pre-processing"):
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        gt_mask = yolo_to_mask(os.path.join(config['data']['test_labels_dir'], Path(img_path).stem + ".txt"), h, w)
        

        # Apply CLAHE to image
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        
        # YOLO gets original image, SAM2 gets enhanced image
        yolo_results = yolo_model(img, conf=config['evaluation']['conf_threshold'], augment=True, verbose=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if len(boxes) > 0:
            for box in boxes:
                res = sam_model(enhanced_img, bboxes=box.tolist(), verbose=False)[0]
                pred_mask = np.logical_or(pred_mask, extract_mask(res, h, w)).astype(np.uint8)

        
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
