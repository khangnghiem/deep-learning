"""
Dataset preparation — Unify 6 polyp datasets into COCO JSON format.

Converts binary segmentation masks into COCO annotations (bbox + polygon)
for Detectron2 Mask R-CNN training.

Usage:
    python prepare_dataset.py \
        --data-root "G:/My Drive/data_lake/01_bronze_medical" \
        --output ./dataset \
        --val-ratio 0.1 \
        --test-ratio 0.1
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# --- Dataset Definitions ---

DATASETS = {
    "kvasir-seg": {
        "images_dir": "kvasir-seg/Kvasir-SEG/Kvasir-SEG/images",
        "masks_dir": "kvasir-seg/Kvasir-SEG/Kvasir-SEG/masks",
    },
    "cvc-clinicdb": {
        "images_dir": "cvc-clinicdb/PNG/Original",
        "masks_dir": "cvc-clinicdb/PNG/Ground Truth",
    },
    "cvc-colondb": {
        "images_dir": "cvc-colondb/Original",
        "masks_dir": "cvc-colondb/Ground Truth",
    },
    "etis-larib": {
        "images_dir": "etis-larib/Original",
        "masks_dir": "etis-larib/Ground Truth",
    },
    "ldpolypvideo": {
        "images_dir": "ldpolypvideo/images",
        "masks_dir": "ldpolypvideo/masks",
    },
    "polypgen": {
        "images_dir": "polypgen/images",
        "masks_dir": "polypgen/masks",
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CATEGORY = {"id": 0, "name": "polyp"}


# --- Mask Processing ---


def mask_to_yolo_lines(
    mask_path: Path,
    img_w: int,
    img_h: int,
    category_id: int = 0,
) -> list[str]:
    """Convert a binary mask image to YOLO segmentation lines formats.
    
    YOLO seg format: `<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>` 
    (normalized coordinates)
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    lines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25:
            continue
            
        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        flat = approx.flatten()
        if len(flat) < 6:
            continue
            
        # Normalize and dump
        normalized = []
        for i in range(0, len(flat), 2):
            x = flat[i] / float(img_w)
            y = flat[i+1] / float(img_h)
            
            # Clip between 0 and 1
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            
            # Formatted to 6 decimal places for cleanliness
            normalized.append(f"{x:.6f} {y:.6f}")
            
        if normalized:
            lines.append(f"{category_id} " + " ".join(normalized))

    return lines


def find_matching_mask(image_path: Path, masks_dir: Path) -> Path | None:
    """Find the mask file that corresponds to an image file."""
    stem = image_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = masks_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# --- Dataset Collection ---


def collect_dataset_entries(
    data_root: Path,
    dataset_name: str,
    dataset_cfg: dict[str, str],
) -> list[dict[str, Any]]:
    """Collect all image/mask pairs from a single dataset."""
    images_dir = data_root / dataset_cfg["images_dir"]
    masks_dir = data_root / dataset_cfg["masks_dir"]

    if not images_dir.exists():
        print(f"  ⚠️  Skipping {dataset_name}: images dir not found at {images_dir}")
        return []
    if not masks_dir.exists():
        print(f"  ⚠️  Skipping {dataset_name}: masks dir not found at {masks_dir}")
        return []

    entries = []
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in image_files:
        mask_path = find_matching_mask(img_path, masks_dir)
        if mask_path is None:
            continue
        entries.append({
            "image_path": img_path,
            "mask_path": mask_path,
            "source": dataset_name,
        })

    print(f"  ✅ {dataset_name}: {len(entries)} image/mask pairs")
    return entries


# --- COCO JSON Building ---


def build_yolo_split(
    entries: list[dict[str, Any]],
    output_images_dir: Path,
    output_labels_dir: Path,
) -> tuple[int, int]:
    """Process images and build YOLO .txt label files."""
    saved_images = 0
    saved_labels = 0

    for entry in entries:
        img_path: Path = entry["image_path"]
        mask_path: Path = entry["mask_path"]
        source: str = entry["source"]

        # Read image to get dimensions and copy
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        out_name_stem = f"{source}_{img_path.stem}"
        out_img_path = output_images_dir / f"{out_name_stem}{img_path.suffix}"
        
        # Write image
        if not out_img_path.exists():
            cv2.imwrite(str(out_img_path), img)
        saved_images += 1

        # Write labels
        lines = mask_to_yolo_lines(mask_path, w, h)
        out_lbl_path = output_labels_dir / f"{out_name_stem}.txt"
        
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")
        if lines:
            saved_labels += 1

    return saved_images, saved_labels


# --- Split & Save ---


def split_entries(
    entries: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Stratified random split into train/val/test."""
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_set = shuffled[:n_test]
    val_set = shuffled[n_test:n_test + n_val]
    train_set = shuffled[n_test + n_val:]

    return train_set, val_set, test_set


def save_split(
    entries: list[dict[str, Any]],
    split_name: str,
    output_root: Path,
) -> None:
    """Build YOLO dataset structure for a split."""
    images_dir = output_root / split_name / "images"
    labels_dir = output_root / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Building {split_name} split ({len(entries)} images)...")
    n_images, n_labels = build_yolo_split(entries, images_dir, labels_dir)
    print(f"  ✅ Saved {split_name}: {n_images} images, {n_labels} non-empty labels")


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unify polyp datasets into COCO JSON for Detectron2 training."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root path containing polyp dataset folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./dataset"),
        help="Output directory for unified dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific datasets to include (default: all)",
    )
    args = parser.parse_args()

    print("🔍 Collecting datasets...")
    all_entries: list[dict[str, Any]] = []

    selected = args.datasets or list(DATASETS.keys())
    for name in selected:
        if name not in DATASETS:
            print(f"  ⚠️  Unknown dataset: {name}")
            continue
        entries = collect_dataset_entries(args.data_root, name, DATASETS[name])
        all_entries.extend(entries)

    if not all_entries:
        print("❌ No image/mask pairs found. Check --data-root and dataset paths.")
        return

    print(f"\n📊 Total: {len(all_entries)} image/mask pairs from {len(selected)} datasets")

    # Split
    train, val, test = split_entries(
        all_entries, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"   Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Build YOLO structure for each split
    args.output.mkdir(parents=True, exist_ok=True)
    save_split(train, "train", args.output)
    save_split(val, "val", args.output)
    save_split(test, "test", args.output)

    # Generate dataset.yaml
    yaml_content = f"""path: {args.output.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: polyp
"""
    yaml_path = args.output / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n✅ Dataset ready at {args.output}")
    print(f"   Structure:")
    print(f"   {args.output}/")
    print(f"   ├── dataset.yaml      (Ultralytics config)")
    print(f"   ├── train/")
    print(f"   │   ├── images/       (YOLO images)")
    print(f"   │   └── labels/       (YOLO labels)")
    print(f"   ├── val/")
    print(f"   └── test/")


if __name__ == "__main__":
    main()
