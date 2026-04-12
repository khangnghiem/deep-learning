"""
=============================================================================
018_polyp_mobilesam — MobileSAM Polyp Segmentation
=============================================================================

Two-stage pipeline:
  Stage 1: Fine-tune a lightweight YOLO detector on polyp bounding boxes.
  Stage 2: Use MobileSAM in zero-shot mode — feed detected boxes as prompts,
           then evaluate predicted masks against ground-truth polygon masks.

This evaluates whether a foundation segmentation model (SAM) can produce
higher-fidelity masks than end-to-end models like YOLO-Seg.

Usage (Colab):
    %run train.py
"""

import sys
import json
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np

# =============================================================================
# LOCAL SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

import yaml
import torch
import cv2

# Auto-release Colab GPU runtime
import atexit
try:
    from google.colab import runtime
    atexit.register(runtime.unassign)
except ImportError:
    pass

from src.config.paths import MLFLOW_TRACKING_URI, TRAINED, setup_mlflow


# =============================================================================
# CONFIGURATION
# =============================================================================
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    required = {
        "experiment": ["name"],
        "data": ["dataset_yaml", "imgsz"],
        "model": ["architecture"],
        "mlflow": ["experiment_name"],
    }
    for section, keys in required.items():
        if section not in config:
            raise ValueError(f"Missing config section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing config key: '{section}.{key}'")
    return config


def write_completion_marker(config, metrics, duration, success, error=None):
    marker = {
        "experiment": config["experiment"]["name"],
        "completed_at": datetime.now().isoformat(),
        "success": success,
        "duration_seconds": round(duration, 1),
        "metrics": metrics,
        "model": config["model"]["architecture"],
        "error": error,
    }
    Path("completed.json").write_text(json.dumps(marker, indent=2))
    logger.info("Completion marker written.")


# =============================================================================
# MASK UTILITIES
# =============================================================================
def yolo_polygon_to_mask(label_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """Convert YOLO polygon label file to binary mask."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not label_path.exists():
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            # Skip class id, rest are x,y pairs normalized
            coords = list(map(float, parts[1:]))
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection / (union + 1e-8))


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return float(2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-8))


# =============================================================================
# MAIN
# =============================================================================
def main():
    config = load_config()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Experiment: %s", config["experiment"]["name"])
    logger.info("Pipeline: YOLO Detector → MobileSAM Segmentation")
    logger.info("=" * 60)

    try:
        mlflow = setup_mlflow()
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)
        if torch.cuda.is_available():
            logger.info("GPU: %s", torch.cuda.get_device_name(0))

        dataset_yaml = config["data"]["dataset_yaml"]
        imgsz = config["data"]["imgsz"]
        save_dir = Path(config["paths"]["project"])
        save_dir.mkdir(parents=True, exist_ok=True)

        run_name = config["mlflow"].get("run_name")
        if not run_name or str(run_name).lower() in ("none", "null"):
            run_name = f"mobilesam_yolo_prompt_ep{config['training'].get('detector_epochs', 25)}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": "mobilesam",
                "box_detector": config["model"].get("box_detector", "yolo11n-seg"),
                "detector_epochs": config["training"].get("detector_epochs", 25),
                "imgsz": imgsz,
                "dataset": config["data"].get("dataset", "polyp"),
                "seed": config["training"].get("seed", 42),
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            })

            # ══════════════════════════════════════════════════════
            # STAGE 1: Fine-tune YOLO detector for box proposals
            # ══════════════════════════════════════════════════════
            logger.info("─" * 40)
            logger.info("STAGE 1: Fine-tuning YOLO box detector")
            logger.info("─" * 40)

            from ultralytics import YOLO

            box_model_name = config["model"].get("box_detector", "yolo11n-seg")
            detector = YOLO(f"{box_model_name}.pt")

            det_epochs = config["training"].get("detector_epochs", 25)
            det_results = detector.train(
                data=dataset_yaml,
                epochs=det_epochs,
                batch=config["data"].get("batch_size", 8),
                imgsz=imgsz,
                lr0=config["training"].get("learning_rate", 0.001),
                seed=config["training"].get("seed", 42),
                device=0 if torch.cuda.is_available() else "cpu",
                project=str(save_dir / "detector"),
                name="yolo_boxes",
                exist_ok=True,
                verbose=True,
            )

            # Log detector metrics
            det_metrics = {}
            if hasattr(det_results, "box"):
                det_metrics = {
                    "detector_mAP50": round(det_results.box.map50, 4),
                    "detector_mAP50-95": round(det_results.box.map, 4),
                    "detector_precision": round(det_results.box.mp, 4),
                    "detector_recall": round(det_results.box.mr, 4),
                }
            mlflow.log_metrics(det_metrics)
            logger.info("Detector metrics: %s", det_metrics)

            det_best = save_dir / "detector" / "yolo_boxes" / "weights" / "best.pt"

            # ══════════════════════════════════════════════════════
            # STAGE 2: MobileSAM zero-shot segmentation
            # ══════════════════════════════════════════════════════
            logger.info("─" * 40)
            logger.info("STAGE 2: MobileSAM zero-shot evaluation")
            logger.info("─" * 40)

            from ultralytics import SAM

            sam_model = SAM("mobile_sam.pt")
            logger.info("Loaded MobileSAM")

            # Load fine-tuned detector for box proposals
            if det_best.exists():
                detector = YOLO(str(det_best))
                logger.info("Using fine-tuned detector: %s", det_best)
            else:
                logger.warning("Fine-tuned detector not found, using pretrained")

            # Parse dataset.yaml for val images
            with open(dataset_yaml) as f:
                ds_config = yaml.safe_load(f)
            ds_root = Path(dataset_yaml).parent
            val_img_dir = ds_root / ds_config.get("val", "val/images")
            val_label_dir = val_img_dir.parent.parent / "labels"

            val_images = sorted(val_img_dir.glob("*.*"))
            logger.info("Evaluating on %d validation images", len(val_images))

            ious = []
            dices = []

            for img_path in val_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                # Get GT mask
                label_path = val_label_dir / (img_path.stem + ".txt")
                gt_mask = yolo_polygon_to_mask(label_path, img_h, img_w)

                if gt_mask.sum() == 0:
                    continue  # Skip images with no polyps

                # Detect boxes with YOLO
                det_preds = detector.predict(
                    str(img_path), conf=0.25, verbose=False, device=device
                )

                if len(det_preds) == 0 or len(det_preds[0].boxes) == 0:
                    ious.append(0.0)
                    dices.append(0.0)
                    continue

                # Use detected boxes as prompts for MobileSAM
                boxes = det_preds[0].boxes.xyxy.cpu().numpy()

                sam_results = sam_model.predict(
                    str(img_path), bboxes=boxes, verbose=False, device=device
                )

                # Merge all SAM masks into one binary mask
                pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                if len(sam_results) > 0 and sam_results[0].masks is not None:
                    for m in sam_results[0].masks.data:
                        resized = cv2.resize(
                            m.cpu().numpy().astype(np.uint8),
                            (img_w, img_h),
                            interpolation=cv2.INTER_NEAREST
                        )
                        pred_mask = np.maximum(pred_mask, resized)

                ious.append(compute_iou(pred_mask, gt_mask))
                dices.append(compute_dice(pred_mask, gt_mask))

            # ── Aggregate Metrics ─────────────────────────────────
            metrics = {
                "sam_mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
                "sam_mean_dice": round(float(np.mean(dices)), 4) if dices else 0.0,
                "sam_median_iou": round(float(np.median(ious)), 4) if ious else 0.0,
                "sam_num_evaluated": len(ious),
                **det_metrics,
            }
            mlflow.log_metrics(metrics)

            logger.info("=" * 40)
            logger.info("MobileSAM Results:")
            logger.info("  Mean IoU:  %.4f", metrics["sam_mean_iou"])
            logger.info("  Mean Dice: %.4f", metrics["sam_mean_dice"])
            logger.info("  Images:    %d", metrics["sam_num_evaluated"])
            logger.info("=" * 40)

            # Save detector weights to TRAINED
            trained_dir = TRAINED / config["experiment"]["name"]
            trained_dir.mkdir(parents=True, exist_ok=True)
            if det_best.exists():
                shutil.copy2(det_best, trained_dir / "detector_best.pt")

        duration = time.time() - start_time
        write_completion_marker(config, metrics, duration, success=True)
        logger.info("✅ Pipeline complete in %.1f minutes", duration / 60)

    except Exception as e:
        duration = time.time() - start_time
        write_completion_marker(config, {}, duration, success=False, error=str(e))
        logger.error("❌ Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()
