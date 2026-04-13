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
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for training (e.g., '0', 'cpu'). Default is auto-detect.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning instead of training",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Iterations for hyperparameter tuning",
    )
    args = parser.parse_args()

    yaml_path = args.dataset / "dataset.yaml"
    if not yaml_path.exists():
        print(f"❌ Cannot find {yaml_path}. Did you run prepare_dataset.py?")
        return

    print("🏥 Polyp Detection — YOLO11-seg Pipeline")
    print("=" * 50)
    print(f"  Dataset: {yaml_path}")
    print(f"  Output:  {args.output}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch_size}")
    print(f"  Device:  {args.device if args.device else 'auto'}")
    
    # 1. Load a pretrained YOLO Segmentation model
    print(f"\n📦 Loading pretrained {args.model}...")
    model = YOLO(args.model)

    if args.tune:
        print(f"\n🚀 Starting hyperparameter tuning for {args.iterations} iterations...")
        model.tune(
            data=str(yaml_path.absolute()),
            epochs=args.epochs,
            iterations=args.iterations,
            batch=args.batch_size,
            workers=8,  # L4 GPU has plenty of CPU RAM and cores, 8 is optimal
            optimizer='AdamW',
            plots=False,
            save=False,
            val=False,
            device=args.device if args.device else None,
        )
        print("\n✅ Tuning complete! Best hyperparameters saved to runs/detect/tune/.")
        return

    # 2. Train the model
    print(f"\n🚀 Starting training...")
    results = model.train(
        data=str(yaml_path.absolute()),
        epochs=args.epochs,
        batch=args.batch_size,
        workers=8,  # Plenty of resources on L4!
        imgsz=args.img_size,
        project=str(args.output),
        name="run",
        exist_ok=True,
        device=args.device if args.device else None,
        # Robust Data Augmentations for Medical Imagery
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.2,
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
