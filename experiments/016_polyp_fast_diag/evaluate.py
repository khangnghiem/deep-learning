"""
Evaluation script — Cross-dataset evaluation for polyp detection model.

Evaluates the trained Mask R-CNN on each dataset individually
to measure generalization across different clinical sources.

Usage:
    python evaluate.py \
        --model ./output/model_final.pth \
        --dataset ./dataset
"""

import argparse
import json
import os
from pathlib import Path

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
NUM_CLASSES = 1


def build_eval_config(model_path: str) -> any:
    """Build Detectron2 config for evaluation."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_path
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 800
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return cfg


def evaluate_split(cfg, dataset_name: str, output_dir: str) -> dict:
    """Run COCO evaluation on a registered dataset."""
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
    loader = build_detection_test_loader(cfg, dataset_name)
    return inference_on_dataset(predictor.model, loader, evaluator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Mask R-CNN polyp detection model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (model_final.pth)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to prepared dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./evaluation_results"),
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for predictions (default: 0.5)",
    )
    args = parser.parse_args()

    print("📊 Polyp Detection — Model Evaluation")
    print("=" * 50)

    # Register datasets
    splits = ["train", "val", "test"]
    for split in splits:
        json_path = args.dataset / f"{split}.json"
        images_dir = args.dataset / split / "images"
        if json_path.exists() and images_dir.exists():
            register_coco_instances(
                f"polyp_{split}", {}, str(json_path), str(images_dir)
            )

    # Build config
    cfg = build_eval_config(args.model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold

    output_dir = str(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate all available splits
    all_results = {}
    for split in ["val", "test"]:
        json_path = args.dataset / f"{split}.json"
        if not json_path.exists():
            continue

        print(f"\n📋 Evaluating {split} split...")
        results = evaluate_split(cfg, f"polyp_{split}", output_dir)
        all_results[split] = results

        if "bbox" in results:
            ap = results["bbox"]
            print(f"  BBox  — AP: {ap.get('AP', 0):.1f} | AP50: {ap.get('AP50', 0):.1f} | AP75: {ap.get('AP75', 0):.1f}")
        if "segm" in results:
            ap = results["segm"]
            print(f"  Segm  — AP: {ap.get('AP', 0):.1f} | AP50: {ap.get('AP50', 0):.1f} | AP75: {ap.get('AP75', 0):.1f}")

    # Save summary
    summary_path = args.output / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ Results saved to {summary_path}")


if __name__ == "__main__":
    main()
