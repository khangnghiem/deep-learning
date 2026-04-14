import argparse
import random
import cv2
import glob
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def mask_to_yolo_lines(mask_path, img_w, img_h, category_id=0):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25: continue
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        flat = approx.flatten()
        if len(flat) < 6: continue
        normalized = []
        for i in range(0, len(flat), 2):
            x = max(0.0, min(1.0, flat[i] / float(img_w)))
            y = max(0.0, min(1.0, flat[i+1] / float(img_h)))
            normalized.append(f"{x:.6f} {y:.6f}")
        if normalized:
            lines.append(f"{category_id} " + " ".join(normalized))
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, required=True, help="Folder containing extracted images and masks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    
    # We will search for all images and try to find matching masks somewhere.
    # Typically ldpolypvideo has 'images/' and 'masks/' or 'ground_truth/'
    # Or they are side by side.
    print(f"Scanning {args.extracted_root} for images...")
    all_images = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(args.extracted_root.rglob(f"*{ext}"))
    
    print(f"Found {len(all_images)} potential images.")
    
    entries = []
    for img_path in all_images:
        # Avoid masks being detected as images. Usually masks are in "masks" or "Ground Truth" or end with "_mask"
        if 'mask' in str(img_path).lower() or 'ground truth' in str(img_path).lower():
            continue
            
        # Try to find corresponding mask
        stem = img_path.stem
        # check parent dirs, swap 'images' to 'masks'
        mask_path = Path(str(img_path).replace('images', 'masks'))
        if not mask_path.exists():
            mask_path = Path(str(img_path).replace('image', 'mask'))
        if not mask_path.exists():
            mask_path = Path(str(img_path).replace('Original', 'Ground Truth'))
        
        # Sometimes mask is a PNG while image is JPG
        found_mask = None
        for ext in IMAGE_EXTENSIONS:
            candidate = mask_path.with_suffix(ext)
            if candidate.exists():
                found_mask = candidate
                break
        
        if found_mask:
            # Determine Video ID from path or filename.
            # Example: LDPolypVideo/videos/123/img001.jpg -> video id is '123'
            # Or LDPolypVideo/images/video_12_001.jpg -> video id is 'video_12'
            parent_name = img_path.parent.name
            video_id = parent_name
            if len(stem.split('_')) > 1:
                video_id = "_".join(stem.split('_')[:-1]) # video_12
            
            entries.append({
                "video_id": video_id,
                "image_path": img_path,
                "mask_path": found_mask
            })

    if not entries:
        print("No paired images and masks found!")
        return

    print(f"Matched {len(entries)} image/mask pairs.")
    
    # Group by video
    from collections import defaultdict
    videos = defaultdict(list)
    for entry in entries:
        videos[entry["video_id"]].append(entry)
    
    video_ids = list(videos.keys())
    print(f"Found {len(video_ids)} distinct video sequences.")
    
    # Split by video ID!
    random.seed(42)
    random.shuffle(video_ids)
    
    n_test = int(len(video_ids) * args.test_ratio)
    n_val = int(len(video_ids) * args.val_ratio)
    
    test_vids = video_ids[:n_test]
    val_vids = video_ids[n_test:n_test + n_val]
    train_vids = video_ids[n_test + n_val:]
    
    def process_split(vids, split_name):
        split_entries = [e for v in vids for e in videos[v]]
        print(f"[{split_name}] {len(vids)} videos, {len(split_entries)} images.")
        img_dir = args.output / split_name / "images"
        lbl_dir = args.output / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for entry in split_entries:
            img = cv2.imread(str(entry["image_path"]))
            if img is None: continue
            h, w = img.shape[:2]
            
            out_stem = f"{entry['video_id']}_{entry['image_path'].stem}"
            cv2.imwrite(str(img_dir / f"{out_stem}.jpg"), img)
            
            lines = mask_to_yolo_lines(entry["mask_path"], w, h)
            with open(lbl_dir / f"{out_stem}.txt", "w") as f:
                f.write("\n".join(lines) + "\n" if lines else "")
                
    process_split(train_vids, "train")
    process_split(val_vids, "val")
    process_split(test_vids, "test")
    
    yaml_content = f"""path: {args.output.absolute()}
train: train/images
val: val/images
test: test/images

names:
  0: polyp
"""
    with open(args.output / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    print("Done!")

if __name__ == "__main__":
    main()
