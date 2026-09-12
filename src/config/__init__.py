"""
Shared configuration for ML repos.

Public API re-exports for convenient imports:
    from src.config import BRONZE, DATASETS, setup_mlflow
"""

# Path configuration
from src.config.paths import IN_COLAB as IN_COLAB
from src.config.paths import DRIVE as DRIVE
from src.config.paths import DATA as DATA
from src.config.paths import DATA_LAKE as DATA_LAKE
from src.config.paths import LANDING as LANDING
from src.config.paths import BRONZE as BRONZE
from src.config.paths import BRONZE_AUDIO as BRONZE_AUDIO
from src.config.paths import BRONZE_MEDICAL as BRONZE_MEDICAL
from src.config.paths import BRONZE_TABULAR as BRONZE_TABULAR
from src.config.paths import BRONZE_TEXT as BRONZE_TEXT
from src.config.paths import BRONZE_TIMESERIES as BRONZE_TIMESERIES
from src.config.paths import BRONZE_VIDEO as BRONZE_VIDEO
from src.config.paths import BRONZE_VISION as BRONZE_VISION
# Legacy aliases (retired categories → new targets)
from src.config.paths import BRONZE_DETECTION as BRONZE_DETECTION   # → BRONZE_VISION
from src.config.paths import BRONZE_EDUCATION as BRONZE_EDUCATION   # → BRONZE_TABULAR
from src.config.paths import BRONZE_GENERATIVE as BRONZE_GENERATIVE  # → BRONZE_VISION
from src.config.paths import BRONZE_NLP as BRONZE_NLP         # → BRONZE_TEXT
from src.config.paths import SILVER as SILVER
from src.config.paths import GOLD as GOLD
from src.config.paths import FEATURES as FEATURES
from src.config.paths import FEATURE_STORE as FEATURE_STORE
from src.config.paths import OPS as OPS
from src.config.paths import MLFLOW_DIR as MLFLOW_DIR
from src.config.paths import MLFLOW_TRACKING_URI as MLFLOW_TRACKING_URI
from src.config.paths import MLFLOW_ARTIFACTS as MLFLOW_ARTIFACTS
from src.config.paths import MODELS as MODELS
from src.config.paths import PRETRAINED as PRETRAINED
from src.config.paths import CHECKPOINTS as CHECKPOINTS
from src.config.paths import TRAINED as TRAINED
from src.config.paths import REGISTRY as REGISTRY
from src.config.paths import REPOS as REPOS
from src.config.paths import get_bronze_path as get_bronze_path
from src.config.paths import get_all_bronze_paths as get_all_bronze_paths
from src.config.paths import get_drive_root as get_drive_root
from src.config.paths import setup_mlflow as setup_mlflow
from src.config.paths import get_env_info as get_env_info

# Dataset catalog
from src.config.catalog import DATASETS as DATASETS
from src.config.catalog import TOTAL_DATASETS as TOTAL_DATASETS
from src.config.catalog import download_dataset as download_dataset
from src.config.catalog import list_datasets as list_datasets
from src.config.catalog import _parse_size as _parse_size

# Manifest management
from src.config.manifest import load_manifest as load_manifest
from src.config.manifest import generate_manifest as generate_manifest
from src.config.manifest import update_manifest_entry as update_manifest_entry
from src.config.manifest import get_manifest_datasets as get_manifest_datasets
from src.config.manifest import StaleManifestError as StaleManifestError
