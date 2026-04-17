import os
import argparse
import logging
from pathlib import Path
import shutil

# Assumes sys.path is setup correctly in Colab, or run with python -m experiments.{EXPERIMENT}.prepare_dataset
try:
    from src.config.paths import BRONZE, SILVER, GOLD
except ImportError:
    # Fallback for direct execution without project path setup
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config.paths import BRONZE, SILVER, GOLD

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def prepare_dataset(dataset_name: str, force: bool = False):
    """
    Reads structured data from SILVER (or falls back to BRONZE), performs 
    preprocessing/splitting, and writes ML-ready tensors/images to GOLD.
    """
    bronze_dir = BRONZE / dataset_name
    silver_dir = SILVER / dataset_name
    gold_dir = GOLD / dataset_name
    
    # Silver Fallback Logic
    if silver_dir.exists():
        source_dir = silver_dir
        logger.info(f"Using cleaned SILVER data from: {source_dir}")
    elif bronze_dir.exists():
        source_dir = bronze_dir
        logger.info(f"SILVER not found. Falling back to raw BRONZE data from: {source_dir}")
    else:
        logger.error(f"Neither SILVER nor BRONZE directory found for {dataset_name}.")
        return
        
    if gold_dir.exists() and not force:
        logger.info(f"Gold directory already exists at {gold_dir}. Use --force to overwrite.")
        return
        
    logger.info(f"Starting preparation for {dataset_name}...")
    logger.info(f"Reading from: {bronze_dir}")
    logger.info(f"Writing to: {gold_dir}")
    
    # 1. Create Gold Directories
    os.makedirs(gold_dir, exist_ok=True)
    os.makedirs(gold_dir / "train", exist_ok=True)
    os.makedirs(gold_dir / "val", exist_ok=True)
    os.makedirs(gold_dir / "test", exist_ok=True)
    
    # ---------------------------------------------------------
    # TODO: Implement dataset-specific preprocessing here!
    # Examples:
    # - Stratified Train/Val/Test splitting
    # - Resizing images to 224x224 and saving as .pt tensors or standard jpegs
    # - Merging COCO JSONs or converting DICOM to PNG
    # ---------------------------------------------------------
    
    logger.info(f"Dataset {dataset_name} successfully prepared in GOLD layer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BRONZE dataset to GOLD tensors.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset in BRONZE layer")
    parser.add_argument("--force", action="store_true", help="Overwrite existing GOLD directory")
    
    args = parser.parse_args()
    prepare_dataset(args.dataset, args.force)
