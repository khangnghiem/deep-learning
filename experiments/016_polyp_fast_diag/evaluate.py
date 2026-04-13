"""
Evaluation script — Dataset evaluation for YOLO11-seg polyp detection model.

Usage:
    python evaluate.py \
        --model ./output/run/weights/best.pt \
        --dataset ./dataset
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11-seg polyp detection model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (best.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to prepared dataset directory (containing dataset.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for evaluation (e.g., '0', 'cpu'). Default is auto-detect.",
    )
    args = parser.parse_args()

    print("📊 Polyp Detection — Model Evaluation")
    print("=" * 50)

    yaml_path = args.dataset / "dataset.yaml"
    if not yaml_path.exists():
        print(f"❌ Cannot find {yaml_path}")
        return

    model = YOLO(args.model)
    
    print(f"\n📋 Evaluating test split on device: {args.device or 'auto'}...")
    results = model.val(
        data=str(yaml_path.absolute()), 
        split="test",
        device=args.device if args.device else None
    )
    
    print(f"\n✅ mAP50(B): {results.box.map50:.3f}")
    if hasattr(results, 'seg') and results.seg is not None:
        print(f"✅ mAP50(M): {results.seg.map50:.3f}")

if __name__ == "__main__":
    main()
