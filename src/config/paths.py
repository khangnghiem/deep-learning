"""
Environment-aware path configuration for multi-environment ML workflows.

Supports:
- Google Colab (Web)
- Google Colab VSCode Extension
- Local MacOS development

All path constants use dynamic resolution with legacy fallback:
- data/ (preferred) or data_lake/ (legacy)
- 0_landing, 1_bronze, 2_silver, 3_gold (preferred) or 00_landing, 01_bronze, 02_silver, 03_gold (legacy)
- models/checkpoints/ (preferred) or models/trained/ (legacy)

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
# Internal Resolution Helpers
# =============================================================================


def _resolve_dir(parent: Path, new_name: str, legacy_name: str) -> Path:
    """Resolve a directory preferring new_name, falling back to legacy_name."""
    new = parent / new_name
    if new.exists():
        return new
    legacy = parent / legacy_name
    if legacy.exists():
        return legacy
    return new  # Default for fresh installs


def _resolve_layer(root: Path, new_prefix: str, legacy_prefix: str, name: str) -> Path:
    """Resolve a Medallion layer directory with prefix fallback.

    Prefers '{new_prefix}_{name}' (e.g. '1_bronze'),
    falls back to '{legacy_prefix}_{name}' (e.g. '01_bronze').
    """
    new = root / f"{new_prefix}_{name}"
    if new.exists():
        return new
    legacy = root / f"{legacy_prefix}_{name}"
    if legacy.exists():
        return legacy
    return new  # Default for fresh installs


# =============================================================================
# Data Lake (Medallion Architecture)
# =============================================================================

# Resolve data root: prefer 'data/', fall back to 'data_lake/'
data_lake_root = None

# 1. Check config.yaml override
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

# 2. Check environment variable override
if "DATA_LAKE_DIR" in os.environ:
    data_lake_root = os.environ["DATA_LAKE_DIR"]

# 3. Dynamic resolution (prefer 'data/', fall back to 'data_lake/')
if data_lake_root:
    DATA = DRIVE / data_lake_root
else:
    DATA = _resolve_dir(DRIVE, "data", "data_lake")

DATA_LAKE = DATA  # Backward-compatible alias

# Resolve Medallion layers with prefix fallback
LANDING = _resolve_layer(DATA, "0", "00", "landing")
BRONZE = _resolve_layer(DATA, "1", "01", "bronze")
SILVER = _resolve_layer(DATA, "2", "02", "silver")
GOLD = _resolve_layer(DATA, "3", "03", "gold")

# Feature Store (Offline representations in Silver: CLIP, DINOv2, SAM embeddings & clinical features)
FEATURES = SILVER / "features"
FEATURE_STORE = FEATURES

# Known data categories (7 pure modality-anchored categories)
KNOWN_CATEGORIES = [
    "audio",
    "multimodal",
    "tabular",
    "text",
    "timeseries",
    "video",
    "vision",
]


def get_bronze_path(category: str) -> Path:
    """Get the bronze layer path for a given category.

    Checks hierarchical '{prefix}_bronze/<category>' first.
    Falls back to legacy flat 'data_lake/01_bronze_<category>' if it exists.
    Remaps retired and domain categories to their pure modality homes.
    """
    cat = category.lower()
    hierarchical = BRONZE / cat
    if hierarchical.exists():
        return hierarchical

    # Remap retired and domain categories to their pure modality targets
    _CATEGORY_ALIASES = {
        "detection": "vision",
        "generative": "vision",
        "education": "tabular",
        "nlp": "text",
        "medical": "vision",
    }
    remapped = _CATEGORY_ALIASES.get(cat, cat)
    aliased = BRONZE / remapped
    if aliased.exists():
        return aliased

    # Legacy flat: try both '01_bronze_<cat>' and '1_bronze_<cat>'
    for pfx in ["01", "1"]:
        for c in [cat, remapped]:
            legacy = DATA / f"{pfx}_bronze_{c}"
            if legacy.exists():
                return legacy
    return aliased


def get_silver_path(category: str = None) -> Path:
    """Get the silver layer path, optionally scoped to a domain category."""
    if not category:
        return SILVER
    cat = category.lower()
    hierarchical = SILVER / cat
    return hierarchical if hierarchical.exists() else SILVER


def get_gold_path(category: str = None) -> Path:
    """Get the gold layer path, optionally scoped to a domain category."""
    if not category:
        return GOLD
    cat = category.lower()
    hierarchical = GOLD / cat
    return hierarchical if hierarchical.exists() else GOLD


def get_all_bronze_paths() -> list:
    """Return list of all existing bronze category paths (hierarchical and legacy)."""
    paths = []
    # Hierarchical {prefix}_bronze/<category>
    if BRONZE.exists():
        for d in BRONZE.iterdir():
            if d.is_dir():
                paths.append(d)
    # Legacy flat 01_bronze_<category> and 1_bronze_<category>
    if DATA.exists():
        for d in DATA.iterdir():
            if d.is_dir() and (
                d.name.startswith("01_bronze_") or d.name.startswith("1_bronze_")
            ):
                if d not in paths:
                    paths.append(d)
    if not paths:
        paths = [BRONZE / cat for cat in KNOWN_CATEGORIES]
    return sorted(list(set(paths)))


# Backward compatibility — domain constants
BRONZE_AUDIO = get_bronze_path("audio")
BRONZE_MEDICAL = get_bronze_path("medical")
BRONZE_TABULAR = get_bronze_path("tabular")
BRONZE_TEXT = get_bronze_path("text")
BRONZE_TIMESERIES = get_bronze_path("timeseries")
BRONZE_VIDEO = get_bronze_path("video")
BRONZE_VISION = get_bronze_path("vision")

# Legacy aliases (retired categories → their new targets)
BRONZE_DETECTION = BRONZE_VISION  # detection merged into vision
BRONZE_EDUCATION = BRONZE_TABULAR  # education merged into tabular
BRONZE_GENERATIVE = BRONZE_VISION  # generative merged into vision
BRONZE_NLP = BRONZE_TEXT  # nlp renamed to text

_BRONZE_CATEGORY_PATHS = {cat: get_bronze_path(cat) for cat in KNOWN_CATEGORIES}

# Backward compatibility — legacy default
LEGACY_BRONZE = BRONZE_VISION

# =============================================================================
# ML Ops (Active Tooling)
# =============================================================================

OPS = DRIVE / "ops"

# =============================================================================
# MLflow (kept in ops, out of data_lake)
# =============================================================================

MLFLOW_DIR = OPS / "mlflow"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACTS = MLFLOW_DIR / "artifacts"

# =============================================================================
# Models
# =============================================================================

MODELS = DRIVE / "models"
PRETRAINED = MODELS / "pretrained"

# Resolve checkpoints: prefer 'checkpoints/', fall back to 'trained/'
CHECKPOINTS = _resolve_dir(MODELS, "checkpoints", "trained")
TRAINED = CHECKPOINTS  # Backward-compatible alias

REGISTRY = MODELS / "registry"

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

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ.setdefault("MLFLOW_DEFAULT_ARTIFACT_ROOT", str(MLFLOW_ARTIFACTS))
    return mlflow


def get_env_info() -> dict:
    """Return current environment configuration."""
    return {
        "in_colab": IN_COLAB,
        "drive": str(DRIVE),
        "data": str(DATA),
        "data_lake": str(DATA_LAKE),
        "bronze": str(BRONZE),
        "silver": str(SILVER),
        "gold": str(GOLD),
        "checkpoints": str(CHECKPOINTS),
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


if __name__ == "__main__":
    print("Environment Configuration:")
    for k, v in get_env_info().items():
        print(f"  {k}: {v}")
