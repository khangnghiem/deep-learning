import argparse, json, random, cv2, numpy as np
from pathlib import Path
from typing import Any

DATASETS = {
    "kvasir-seg": {"images_dir": "kvasir-seg/Kvasir-SEG/Kvasir-SEG/images", "masks_dir": "kvasir-seg/Kvasir-SEG/Kvasir-SEG/masks"},
    "cvc-clinicdb": {"images_dir": "cvc-clinicdb/PNG", "masks_dir": "cvc-clinicdb/TIF"},
    "cvc-colondb": {"images_dir": "cvc-colondb/CVC-ColonDB/CVC-ColonDB/Original", "masks_dir": "cvc-colondb/CVC-ColonDB/CVC-ColonDB/Ground Truth"},
    "etis-larib": {"images_dir": "etis-larib/images", "masks_dir": "etis-larib/masks"},
    "ldpolypvideo": {"images_dir": "ldpolypvideo/images", "masks_dir": "ldpolypvideo/masks"},
    "polypgen": {"images_dir": "polypgen", "masks_dir": "polypgen/masks"},
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def mask_to_yolo_lines(mask_path, img_w, img_h, category_id=0):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 25: continue
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        flat = approx.flatten()
        if len(flat) < 6: continue
        normalized = []
        for i in range(0, len(flat), 2):
            x = max(0.0, min(1.0, flat[i] / float(img_w)))
            y = max(0.0, min(1.0, flat[i+1] / float(img_h)))
            normalized.append(f"{x:.6f} {y:.6f}")
        if normalized: lines.append(f"{category_id} " + " ".join(normalized))
    return lines

def find_matching_mask(image_path, masks_dir):
    for ext in IMAGE_EXTENSIONS:
        candidate = masks_dir / f"{image_path.stem}{ext}"
        if candidate.exists(): return candidate
    return None

def collect_dataset_entries(data_root, dataset_name, dataset_cfg):
    images_dir, masks_dir = data_root / dataset_cfg["images_dir"], data_root / dataset_cfg["masks_dir"]
    if not images_dir.exists():
        print(f"[{dataset_name}] Warning: Images dir not found at {images_dir}")
        return []
    if not masks_dir.exists():
        print(f"[{dataset_name}] Warning: Masks dir not found at {masks_dir}")
        return []
    
    entries = []
    for img_path in sorted(f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS):
        mask_path = find_matching_mask(img_path, masks_dir)
        if mask_path: entries.append({"image_path": img_path, "mask_path": mask_path, "source": dataset_name})
    
    print(f"[{dataset_name}] Found {len(entries)} valid image-mask pairs.")
    return entries

def build_yolo_split(entries, out_img_dir, out_lbl_dir):
    for entry in entries:
        img_path, mask_path, source = entry["image_path"], entry["mask_path"], entry["source"]
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        out_name_stem = f"{source}_{img_path.stem}"
        out_img_path = out_img_dir / f"{out_name_stem}{img_path.suffix}"
        if not out_img_path.exists(): cv2.imwrite(str(out_img_path), img)
        lines = mask_to_yolo_lines(mask_path, w, h)
        out_lbl_path = out_lbl_dir / f"{out_name_stem}.txt"
        with open(out_lbl_path, "w") as f: f.write("\n".join(lines) + "\n" if lines else "")

def save_split(entries, split_name, output_root):
    img_dir, lbl_dir = output_root / split_name / "images", output_root / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    build_yolo_split(entries, img_dir, lbl_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_entries = []
    for name, cfg in DATASETS.items(): all_entries.extend(collect_dataset_entries(args.data_root, name, cfg))
    random.seed(42)
    random.shuffle(all_entries)
    n = len(all_entries)
    if n == 0:
        print("No data found! Exiting.")
        return
    n_test, n_val = int(n * 0.1), int(n * 0.1)
    test, val, train = all_entries[:n_test], all_entries[n_test:n_test+n_val], all_entries[n_test+n_val:]
    args.output.mkdir(parents=True, exist_ok=True)
    save_split(train, "train", args.output); save_split(val, "val", args.output); save_split(test, "test", args.output)
    yaml_path = args.output / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {args.output.absolute()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: polyp\n")
    print(f"Successfully saved dataset splits to {args.output}")

if __name__ == "__main__": main()
