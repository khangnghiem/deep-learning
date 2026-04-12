"""
=============================================================================
017_polyp_rtdetr — RT-DETR Polyp Detection
=============================================================================

Uses the Ultralytics RT-DETR engine for NMS-free transformer-based
object detection on the existing YOLO-format polyp dataset.

Ultralytics handles the full training loop, data augmentation, and evaluation.
We wrap it with MLflow logging and Drive persistence.

Usage (Colab):
    %run train.py
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

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

# Auto-release Colab GPU runtime when script finishes
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
    # Validate required keys
    required = {
        "experiment": ["name"],
        "data": ["dataset_yaml", "batch_size", "imgsz"],
        "model": ["architecture"],
        "training": ["epochs", "learning_rate"],
        "mlflow": ["experiment_name"],
    }
    for section, keys in required.items():
        if section not in config:
            raise ValueError(f"Missing config section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing config key: '{section}.{key}'")
    return config


# =============================================================================
# COMPLETION MARKER
# =============================================================================
def write_completion_marker(config, metrics, duration, success, error=None):
    marker = {
        "experiment": config["experiment"]["name"],
        "completed_at": datetime.now().isoformat(),
        "success": success,
        "duration_seconds": round(duration, 1),
        "metrics": metrics,
        "model": config["model"]["architecture"],
        "epochs": config["training"]["epochs"],
        "error": error,
    }
    Path("completed.json").write_text(json.dumps(marker, indent=2))
    logger.info("Completion marker written.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    config = load_config()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Experiment: %s", config['experiment']['name'])
    logger.info("Model: RT-DETR (%s)", config['model']['architecture'])
    logger.info("=" * 60)

    try:
        # ── MLflow Setup ──────────────────────────────────────────
        mlflow = setup_mlflow()
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        # ── Device Info ───────────────────────────────────────────
        device = "0" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)
        if torch.cuda.is_available():
            logger.info("GPU: %s", torch.cuda.get_device_name(0))

        # ── Ultralytics RT-DETR ───────────────────────────────────
        from ultralytics import RTDETR, settings
        settings.update({'mlflow': False})

        model_variant = config["model"]["architecture"]  # e.g. "rtdetr-l"
        model = RTDETR(f"{model_variant}.pt")
        logger.info("Loaded pretrained RT-DETR: %s", model_variant)

        # ── Dataset ───────────────────────────────────────────────
        dataset_yaml = config["data"]["dataset_yaml"]
        logger.info("Dataset YAML: %s", dataset_yaml)

        # ── Training ──────────────────────────────────────────────
        save_dir = Path(config["paths"]["project"])
        save_dir.mkdir(parents=True, exist_ok=True)

        run_name = config["mlflow"].get("run_name")
        if not run_name or str(run_name).lower() in ("none", "null"):
            run_name = f"rtdetr_{model_variant}_ep{config['training']['epochs']}"

        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_params({
                "model": model_variant,
                "epochs": config["training"]["epochs"],
                "lr": config["training"]["learning_rate"],
                "optimizer": config["training"].get("optimizer", "AdamW"),
                "batch_size": config["data"]["batch_size"],
                "imgsz": config["data"]["imgsz"],
                "patience": config["training"].get("patience", 10),
                "seed": config["training"].get("seed", 42),
                "dataset": config["data"].get("dataset", "polyp"),
                "num_classes": config["model"].get("num_classes", 1),
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            })

            # Run Ultralytics training
            results = model.train(
                data=dataset_yaml,
                epochs=config["training"]["epochs"],
                batch=config["data"]["batch_size"],
                imgsz=config["data"]["imgsz"],
                lr0=config["training"]["learning_rate"],
                optimizer=config["training"].get("optimizer", "AdamW"),
                patience=config["training"].get("patience", 10),
                seed=config["training"].get("seed", 42),
                device=device,
                project=str(save_dir),
                name="train",
                exist_ok=True,
                verbose=True,
                workers=config["data"].get("num_workers", 4),
            )

            # ── Extract Metrics ───────────────────────────────────
            metrics = {}
            if hasattr(results, 'results_dict'):
                metrics = {k: round(v, 4) for k, v in results.results_dict.items()
                           if isinstance(v, (int, float))}
            elif hasattr(results, 'box'):
                metrics = {
                    "mAP50": round(results.box.map50, 4),
                    "mAP50-95": round(results.box.map, 4),
                    "precision": round(results.box.mp, 4),
                    "recall": round(results.box.mr, 4),
                }

            logger.info("Final metrics: %s", metrics)
            mlflow.log_metrics(metrics)

            # Log best weights as artifact
            best_weights = save_dir / "train" / "weights" / "best.pt"
            if best_weights.exists():
                mlflow.log_artifact(str(best_weights))
                logger.info("Logged best weights to MLflow: %s", best_weights)

            # Also copy best weights to the standard TRAINED directory
            trained_dir = TRAINED / config["experiment"]["name"]
            trained_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            if best_weights.exists():
                shutil.copy2(best_weights, trained_dir / "best.pt")
                logger.info("Copied best weights to: %s", trained_dir / "best.pt")

        duration = time.time() - start_time
        write_completion_marker(config, metrics, duration, success=True)
        logger.info("✅ Training complete in %.1f minutes", duration / 60)

    except Exception as e:
        duration = time.time() - start_time
        write_completion_marker(config, {}, duration, success=False, error=str(e))
        logger.error("❌ Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
