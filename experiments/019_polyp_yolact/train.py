"""
=============================================================================
019_polyp_yolact (YOLO-Seg alternative)
=============================================================================

Original YOLACT requires extremely outdated PyTorch versions.
We use Ultralytics YOLO11-Seg for real-time instance segmentation, which
is functionally identical in purpose but much more modern and stable.

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

# =============================================================================
# LOCAL SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

import yaml
import torch

import atexit
try:
    from google.colab import runtime
    atexit.register(runtime.unassign)
except ImportError:
    pass

from src.config.paths import MLFLOW_TRACKING_URI, TRAINED, setup_mlflow

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
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

def main():
    config = load_config()
    start_time = time.time()
    logger.info("Experiment: %s", config["experiment"]["name"])

    try:
        mlflow = setup_mlflow()
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        device = "0" if torch.cuda.is_available() else "cpu"
        
        from ultralytics import YOLO, settings
        settings.update({'mlflow': False})
        
        model_variant = config["model"]["architecture"]
        model = YOLO(f"{model_variant}.pt")

        save_dir = Path(config["paths"]["project"])
        save_dir.mkdir(parents=True, exist_ok=True)
        run_name = config["mlflow"].get("run_name") or f"{model_variant}_ep{config['training']['epochs']}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": model_variant,
                "epochs": config["training"]["epochs"],
                "batch_size": config["data"]["batch_size"],
                "imgsz": config["data"]["imgsz"]
            })

            results = model.train(
                data=config["data"]["dataset_yaml"],
                epochs=config["training"]["epochs"],
                batch=config["data"]["batch_size"],
                imgsz=config["data"]["imgsz"],
                lr0=config["training"]["learning_rate"],
                device=device,
                project=str(save_dir),
                name="train",
                exist_ok=True,
                verbose=True
            )

            metrics = {}
            if hasattr(results, 'results_dict'):
                metrics = {k: round(v, 4) for k, v in results.results_dict.items() if isinstance(v, (int, float))}
            elif hasattr(results, 'seg'):
                metrics = {
                    "seg_mAP50": round(results.seg.map50, 4),
                    "seg_mAP50-95": round(results.seg.map, 4),
                }

            mlflow.log_metrics(metrics)
            best_weights = save_dir / "train" / "weights" / "best.pt"
            
            trained_dir = TRAINED / config["experiment"]["name"]
            trained_dir.mkdir(parents=True, exist_ok=True)
            if best_weights.exists():
                shutil.copy2(best_weights, trained_dir / "best.pt")

        duration = time.time() - start_time
        write_completion_marker(config, metrics, duration, success=True)
        logger.info("✅ Training complete.")

    except Exception as e:
        write_completion_marker(config, {}, time.time() - start_time, success=False, error=str(e))
        raise

if __name__ == "__main__":
    main()
