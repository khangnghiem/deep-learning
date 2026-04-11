"""
Environment-aware path configuration for multi-environment ML workflows.

Supports:
- Google Colab (Web)
- Google Colab VSCode Extension  
- Local MacOS development

Usage:
    from src.config.paths import BRONZE, MLFLOW_TRACKING_URI
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
"""

import sys
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (2 levels up from src/config/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env", override=False)

# =============================================================================
# Environment Detection
# =============================================================================

def _is_colab() -> bool:
    """Detect if running in Google Colab."""
    # Check sys.modules (works after google.colab is imported)
    if "google.colab" in sys.modules:
        return True
    # Check for Colab-specific path (works even before imports)
    if Path("/content/drive").exists():
        return True
    return False

IN_COLAB = _is_colab()

def get_drive_root() -> Path:
    """Determine the Google Drive root based on execution environment."""
    # 1. Explicit override always wins
    if "DRIVE_ROOT" in os.environ:
        return Path(os.environ["DRIVE_ROOT"])
        
    # 2. Colab
    if IN_COLAB:
        colab_drive = Path("/content/drive/MyDrive")
        if colab_drive.exists():
            return colab_drive
        raise FileNotFoundError(
            "Drive not mounted. Run: from google.colab import drive; drive.mount('/content/drive')"
        )
        
    # 3. Windows default fallback
    win_path = Path("G:/My Drive")
    if win_path.exists():
        return win_path
        
    raise FileNotFoundError(
        "Google Drive not found at default locations. "
        "Set DRIVE_ROOT in .env or environment variable."
    )

# Allow override via environment variable
DRIVE = get_drive_root()

# =============================================================================
# Data Lake (Medallion Architecture)
# =============================================================================

data_lake_root = "data_lake"
try:
    config_path = _REPO_ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cf = yaml.safe_load(f)
            if cf and "data" in cf and "root" in cf["data"]:
                data_lake_root = cf["data"]["root"]
except Exception:
    pass

if "DATA_LAKE_DIR" in os.environ:
    data_lake_root = os.environ["DATA_LAKE_DIR"]

DATA_LAKE = DRIVE / data_lake_root
LANDING = DATA_LAKE / "00_landing"

# Bronze layer - organized by category
BRONZE_AUDIO = DATA_LAKE / "01_bronze_audio"
BRONZE_DETECTION = DATA_LAKE / "01_bronze_detection"
BRONZE_EDUCATION = DATA_LAKE / "01_bronze_education"
BRONZE_GENERATIVE = DATA_LAKE / "01_bronze_generative"
BRONZE_MEDICAL = DATA_LAKE / "01_bronze_medical"
BRONZE_NLP = DATA_LAKE / "01_bronze_nlp"
BRONZE_TABULAR = DATA_LAKE / "01_bronze_tabular"
BRONZE_TIMESERIES = DATA_LAKE / "01_bronze_timeseries"
BRONZE_VIDEO = DATA_LAKE / "01_bronze_video"
BRONZE_VISION = DATA_LAKE / "01_bronze_vision"

# Category to path mapping
_BRONZE_CATEGORY_PATHS = {
    "audio": BRONZE_AUDIO,
    "detection": BRONZE_DETECTION,
    "education": BRONZE_EDUCATION,
    "generative": BRONZE_GENERATIVE,
    "medical": BRONZE_MEDICAL,
    "nlp": BRONZE_NLP,
    "tabular": BRONZE_TABULAR,
    "timeseries": BRONZE_TIMESERIES,
    "video": BRONZE_VIDEO,
    "vision": BRONZE_VISION,
}

def get_bronze_path(category: str) -> Path:
    """Get the bronze layer path for a given category.
    
    Args:
        category: Dataset category (vision, audio, nlp, etc.)
        
    Returns:
        Path to the category-specific bronze folder.
        Defaults to BRONZE_VISION if category not recognized.
    """
    return _BRONZE_CATEGORY_PATHS.get(category.lower(), BRONZE_VISION)

def get_all_bronze_paths() -> list:
    """Return list of all bronze category paths."""
    return list(_BRONZE_CATEGORY_PATHS.values())

SILVER = DATA_LAKE / "02_silver"
GOLD = DATA_LAKE / "03_gold"

# =============================================================================
# ML Ops (Active Tooling)
# =============================================================================

OPS = DRIVE / "ops"

# =============================================================================
# Observability
# =============================================================================

OBSERVABILITY = OPS / "observability"

# =============================================================================
# MLflow (kept in ops, out of data_lake)
# =============================================================================

MLFLOW_DIR = OPS / "mlflow"
MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR / 'mlruns'}"
MLFLOW_ARTIFACTS = MLFLOW_DIR / "artifacts"

# =============================================================================
# Models
# =============================================================================

MODELS = DRIVE / "models"
PRETRAINED = MODELS / "pretrained"
TRAINED = MODELS / "trained"

# =============================================================================
# Repos
# =============================================================================

REPOS = DRIVE / "repos"

# =============================================================================
# Utility Functions
# =============================================================================

def setup_mlflow():
    """Configure MLflow with Drive-based tracking."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow

def get_env_info() -> dict:
    """Return current environment configuration."""
    return {
        "in_colab": IN_COLAB,
        "drive": str(DRIVE),
        "data_lake": str(DATA_LAKE),
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }

if __name__ == "__main__":
    print("Environment Configuration:")
    for k, v in get_env_info().items():
        print(f"  {k}: {v}")
