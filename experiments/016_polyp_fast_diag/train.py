"""
Training script — Fine-tune YOLO11-seg for polyp detection.

Uses YOLO dataset structure produced by prepare_dataset.py.

Usage:
    python train.py \
        --dataset ./dataset \
        --output "G:/My Drive/models/trained/polyp_detection" \
        --epochs 50 \
        --batch-size 16
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO11-seg for polyp detection."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to prepared dataset directory containing dataset.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output"),
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m-seg.pt",
        help="Pretrained model to start from (e.g. yolo11m-seg.pt, yolo11n-seg.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Images per batch (default: 16)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for training (default: 640)",
    )
    args = parser.parse_args()

    yaml_path = args.dataset / "dataset.yaml"
    if not yaml_path.exists():
        print(f"❌ Cannot find {yaml_path}. Did you run prepare_dataset.py?")
        return

    print("🏥 Polyp Detection — YOLO11-seg Training")
    print("=" * 50)
    print(f"  Dataset: {yaml_path}")
    print(f"  Output:  {args.output}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch_size}")
    
    # 1. Load a pretrained YOLO Segmentation model
    print(f"\n📦 Loading pretrained {args.model}...")
    model = YOLO(args.model)

    # 2. Train the model
    print("\n🚀 Starting training...")
    results = model.train(
        data=str(yaml_path.absolute()),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=str(args.output),
        name="run",
        exist_ok=True,
        device=0,  # GPU 0
    )

    print("\n✅ Training complete!")
    trained_pt = args.output / "run" / "weights" / "best.pt"
    print(f"   Best PyTorch model saved to: {trained_pt}")

    # 3. Export to ONNX for fast CPU inference
    print("\n⚡ Exporting to ONNX for CPU inference...")
    # Requires reloading the best model
    best_model = YOLO(str(trained_pt))
    onnx_path = best_model.export(format="onnx", imgsz=args.img_size, dynamic=True)
    
    print("\n✅ Export complete!")
    print(f"   ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    main()
