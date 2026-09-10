"""
Shared configuration for ML repos.

Public API re-exports for convenient imports:
    from src.config import BRONZE, DATASETS, setup_mlflow
"""

# Path configuration
from src.config.paths import (
    IN_COLAB,
    DRIVE,
    DATA,
    DATA_LAKE,
    LANDING,
    BRONZE,
    BRONZE_AUDIO,
    BRONZE_MEDICAL,
    BRONZE_TABULAR,
    BRONZE_TEXT,
    BRONZE_TIMESERIES,
    BRONZE_VIDEO,
    BRONZE_VISION,
    # Legacy aliases (retired categories → new targets)
    BRONZE_DETECTION,   # → BRONZE_VISION
    BRONZE_EDUCATION,   # → BRONZE_TABULAR
    BRONZE_GENERATIVE,  # → BRONZE_VISION
    BRONZE_NLP,         # → BRONZE_TEXT
    SILVER,
    GOLD,
    FEATURES,
    FEATURE_STORE,
    OPS,
    MLFLOW_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_ARTIFACTS,
    MODELS,
    PRETRAINED,
    CHECKPOINTS,
    TRAINED,
    REGISTRY,
    REPOS,
    get_bronze_path,
    get_all_bronze_paths,
    get_drive_root,
    setup_mlflow,
    get_env_info,
)

# Dataset catalog
from src.config.catalog import (
    DATASETS,
    TOTAL_DATASETS,
    download_dataset,
    list_datasets,
    _parse_size,
)

# Manifest management
from src.config.manifest import (
    load_manifest,
    generate_manifest,
    update_manifest_entry,
    get_manifest_datasets,
    StaleManifestError,
)
