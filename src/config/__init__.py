"""
Shared configuration for ML repos.

Public API re-exports for convenient imports:
    from src.config import BRONZE as BRONZE, DATASETS as DATASETS, setup_mlflow
"""

# Path configuration
from src.config.paths import (
    IN_COLAB as IN_COLAB,
    DRIVE as DRIVE,
    DATA as DATA,
    DATA_LAKE as DATA_LAKE,
    LANDING as LANDING,
    BRONZE as BRONZE,
    BRONZE_AUDIO as BRONZE_AUDIO,
    BRONZE_MEDICAL as BRONZE_MEDICAL,
    BRONZE_TABULAR as BRONZE_TABULAR,
    BRONZE_TEXT as BRONZE_TEXT,
    BRONZE_TIMESERIES as BRONZE_TIMESERIES,
    BRONZE_VIDEO as BRONZE_VIDEO,
    BRONZE_VISION as BRONZE_VISION,
    # Legacy aliases (retired categories → new targets)
    BRONZE_DETECTION as BRONZE_DETECTION,  # → BRONZE_VISION
    BRONZE_EDUCATION as BRONZE_EDUCATION,  # → BRONZE_TABULAR
    BRONZE_GENERATIVE as BRONZE_GENERATIVE,  # → BRONZE_VISION
    BRONZE_NLP as BRONZE_NLP,  # → BRONZE_TEXT
    SILVER as SILVER,
    GOLD as GOLD,
    FEATURES as FEATURES,
    FEATURE_STORE as FEATURE_STORE,
    OPS as OPS,
    MLFLOW_DIR as MLFLOW_DIR,
    MLFLOW_TRACKING_URI as MLFLOW_TRACKING_URI,
    MLFLOW_ARTIFACTS as MLFLOW_ARTIFACTS,
    MODELS as MODELS,
    PRETRAINED as PRETRAINED,
    CHECKPOINTS as CHECKPOINTS,
    TRAINED as TRAINED,
    REGISTRY as REGISTRY,
    REPOS as REPOS,
    get_bronze_path as get_bronze_path,
    get_all_bronze_paths as get_all_bronze_paths,
    get_drive_root as get_drive_root,
    setup_mlflow as setup_mlflow,
    get_env_info as get_env_info,
)

# Dataset catalog
from src.config.catalog import (
    DATASETS as DATASETS,
    TOTAL_DATASETS as TOTAL_DATASETS,
    download_dataset as download_dataset,
    list_datasets as list_datasets,
    _parse_size as _parse_size,
)

# Manifest management
from src.config.manifest import (
    load_manifest as load_manifest,
    generate_manifest as generate_manifest,
    update_manifest_entry as update_manifest_entry,
    get_manifest_datasets as get_manifest_datasets,
    StaleManifestError as StaleManifestError,
)
