import os
import shutil
import cv2
import argparse
from pathlib import Path
import random
from tqdm import tqdm

try:
    from src.config.paths import BRONZE, SILVER, GOLD
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config.paths import BRONZE, SILVER, GOLD

def convert_to_yolo_bbox(bbox, img_w, img_h):
    """Convert [xmin, ymin, xmax, ymax] to [x_center, y_center, w, h] normalized."""
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate dimensions
    w = xmax - xmin
    h = ymax - ymin
    
    # Calculate center
    x_center = xmin + w / 2.0
    y_center = ymin + h / 2.0
    
    # Normalize
    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h
    
    # Clamp to [0, 1] just in case
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return [x_center, y_center, w, h]

def prepare_dataset(val_split=0.2, seed=42):
    random.seed(seed)
    
    silver_dir = SILVER / 'polyp' / 'ldpolypvideo'
    gold_dir = GOLD / 'ldpolypvideo_yolo'
    
    if not silver_dir.exists():
        print(f"Error: Silver directory not found at {silver_dir}")
        return
        
    print(f"Creating YOLO dataset at {gold_dir} ...")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (gold_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (gold_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    # Process TrainValid
    trainval_img_dir = silver_dir / 'TrainValid' / 'Images'
    trainval_ann_dir = silver_dir / 'TrainValid' / 'Annotations'
    
    if trainval_img_dir.exists() and trainval_ann_dir.exists():
        videos = sorted([d for d in os.listdir(trainval_img_dir) if os.path.isdir(trainval_img_dir / d)])
        
        # Shuffle videos for train/val split
        random.shuffle(videos)
        num_val = int(len(videos) * val_split)
        val_videos = videos[:num_val]
        train_videos = videos[num_val:]
        
        print(f"Total videos: {len(videos)} | Train: {len(train_videos)} | Val: {len(val_videos)}")
        
        for video in tqdm(videos, desc="Processing Train/Val"):
            split = 'val' if video in val_videos else 'train'
            v_img_dir = trainval_img_dir / video
            v_ann_dir = trainval_ann_dir / video
            
            if not v_ann_dir.exists():
                continue
                
            images = sorted([img for img in os.listdir(v_img_dir) if img.endswith(('.jpg', '.png'))])
            
            for img_name in images:
                base_name = os.path.splitext(img_name)[0]
                img_path = v_img_dir / img_name
                ann_path = v_ann_dir / f"{base_name}.txt"
                
                # New names using video id to prevent overwrites
                new_basename = f"vid{video}_{base_name}"
                new_img_path = gold_dir / 'images' / split / f"{new_basename}.jpg"
                new_lbl_path = gold_dir / 'labels' / split / f"{new_basename}.txt"
                
                if not ann_path.exists():
                    # No annotation file means background/no polyp
                    shutil.copy(img_path, new_img_path)
                    # create empty label
                    with open(new_lbl_path, 'w') as f:
                        pass
                    continue
                
                # Read image to get dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                
                # Copy image
                shutil.copy(img_path, new_img_path)
                
                # Process annotations
                with open(ann_path, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    with open(new_lbl_path, 'w') as f:
                        pass
                    continue
                    
                # First line is number of polys, subsequent lines are bboxes
                try:
                    num_polyps = int(lines[0].strip())
                    bboxes = []
                    for i in range(1, num_polyps + 1):
                        if i < len(lines):
                            parts = []
                            for p in lines[i].strip().split():
                              if p.isdigit():
                                parts.append(int(p))
                            if len(parts) >= 4:
                                bboxes.append(parts[:4])
                except ValueError:
                    bboxes = []
                    
                # Write YOLO format
                with open(new_lbl_path, 'w') as f:
                    for bbox in bboxes:
                        yolo_bbox = convert_to_yolo_bbox(bbox, img_w, img_h)
                        f.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

    # Process Test sequence
    test_img_dir = silver_dir / 'Test' / 'Images'
    test_ann_dir = silver_dir / 'Test' / 'Annotations'
    
    if test_img_dir.exists():
        print("Processing Test set...")
        videos = sorted([d for d in os.listdir(test_img_dir) if os.path.isdir(test_img_dir / d)])
        for video in tqdm(videos, desc="Processing Test"):
            v_img_dir = test_img_dir / video
            v_ann_dir = test_ann_dir / video
            
            images = sorted([img for img in os.listdir(v_img_dir) if img.endswith(('.jpg', '.png'))])
            for img_name in images:
                base_name = os.path.splitext(img_name)[0]
                img_path = v_img_dir / img_name
                ann_path = v_ann_dir / f"{base_name}.txt" if v_ann_dir.exists() else None
                
                new_basename = f"vid{video}_{base_name}"
                new_img_path = gold_dir / 'images' / 'test' / f"{new_basename}.jpg"
                new_lbl_path = gold_dir / 'labels' / 'test' / f"{new_basename}.txt"
                
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                
                shutil.copy(img_path, new_img_path)
                
                if ann_path and ann_path.exists():
                    with open(ann_path, 'r') as f:
                        lines = f.readlines()
                    try:
                        num_polyps = int(lines[0].strip())
                        bboxes = []
                        for i in range(1, num_polyps + 1):
                            if i < len(lines):
                                parts = []
                                for p in lines[i].strip().split():
                                  if p.isdigit():
                                    parts.append(int(p))
                                if len(parts) >= 4:
                                    bboxes.append(parts[:4])
                    except ValueError:
                        bboxes = []
                    
                    with open(new_lbl_path, 'w') as f:
                        for bbox in bboxes:
                            yolo_bbox = convert_to_yolo_bbox(bbox, img_w, img_h)
                            f.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                else:
                    with open(new_lbl_path, 'w') as f:
                        pass
                        
    # Create dataset.yaml
    yaml_content = f"""path: {gold_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: polyp"""
  
    with open(gold_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
        
    print(f"Preparation complete! Data stored in {gold_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(val_split=args.val_split, seed=args.seed)
