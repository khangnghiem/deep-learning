import logging

logger = logging.getLogger(__name__)
"""
Data Lake MANIFEST.json management.

Enforces MANIFEST.json as the single source of truth for data lake contents.
The manifest must be kept fresh — scanning code reads from it instead of
crawling 220K+ files on disk.

Usage:
    from src.config.manifest import load_manifest, generate_manifest

    # Load (raises StaleManifestError if too old)
    manifest = load_manifest(max_age_hours=24)

    # Regenerate after ingestion
    manifest = generate_manifest()
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config.paths import DATA_LAKE


# =============================================================================
# Constants
# =============================================================================

MANIFEST_PATH = DATA_LAKE / "MANIFEST.json"


# =============================================================================
# Exceptions
# =============================================================================

class StaleManifestError(Exception):
    """Raised when MANIFEST.json is older than the allowed threshold."""
    pass


# =============================================================================
# Core Functions
# =============================================================================

def load_manifest(max_age_hours: int = 24) -> dict:
    """Load MANIFEST.json, enforcing freshness.

    Args:
        max_age_hours: Maximum age in hours before raising StaleManifestError.
            Set to 0 to skip the staleness check.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: If MANIFEST.json does not exist.
        StaleManifestError: If MANIFEST.json is older than max_age_hours.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"MANIFEST.json not found at {MANIFEST_PATH}. "
            "Run generate_manifest() from Colab to create it."
        )

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Staleness check
    if max_age_hours > 0:
        generated_at = manifest.get("generated_at", "")
        if generated_at:
            try:
                gen_time = datetime.fromisoformat(generated_at)
                # Treat naive timestamps as UTC
                if gen_time.tzinfo is None:
                    gen_time = gen_time.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                age_hours = (now - gen_time).total_seconds() / 3600
                if age_hours > max_age_hours:
                    raise StaleManifestError(
                        f"MANIFEST.json is {age_hours:.1f}h old "
                        f"(max: {max_age_hours}h). "
                        "Run generate_manifest() to refresh."
                    )
            except (ValueError, TypeError):
                logger.debug("Can't parse timestamp — skip check")

    return manifest


def generate_manifest() -> dict:
    """Scan all bronze directories and write a fresh MANIFEST.json.

    This replaces the inline logic from colab_cleanup.ipynb Cell 5.
    Should be run from Colab where filesystem access is fast.

    Returns:
        The generated manifest dict.
    """
    logger.info("=== Generating MANIFEST.json ===")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "data_lake_path": str(DATA_LAKE),
        "layers": {},
        "datasets": {},
        "summary": {},
    }

    bronze_categories = sorted([
        d for d in DATA_LAKE.iterdir()
        if d.is_dir() and d.name.startswith("01_bronze")
    ])

    total_datasets = 0
    total_files = 0
    total_size_bytes = 0

    for bronze_dir in bronze_categories:
        category = bronze_dir.name.replace("01_bronze_", "").replace("01_bronze", "legacy")
        category_datasets = []

        for ds_dir in sorted(bronze_dir.iterdir()):
            if ds_dir.is_dir():
                files = list(ds_dir.rglob("*"))
                file_list = [f for f in files if f.is_file()]
                file_count = len(file_list)
                dir_count = sum(1 for f in files if f.is_dir())

                try:
                    size_bytes = sum(f.stat().st_size for f in file_list)
                except OSError:
                    size_bytes = 0

                size_mb = size_bytes / (1024 * 1024)
                is_empty = file_count == 0

                # Sample extensions from first 100 files
                extensions = set()
                for f in file_list[:100]:
                    if f.suffix:
                        extensions.add(f.suffix.lower())

                ds_info = {
                    "category": category,
                    "path": str(ds_dir.relative_to(DATA_LAKE)),
                    "file_count": file_count,
                    "dir_count": dir_count,
                    "size_mb": round(size_mb, 2),
                    "empty": is_empty,
                    "extensions": sorted(list(extensions)),
                }

                manifest["datasets"][ds_dir.name] = ds_info
                category_datasets.append(ds_dir.name)

                total_datasets += 1
                total_files += file_count
                total_size_bytes += size_bytes

        manifest["layers"][bronze_dir.name] = {
            "dataset_count": len(category_datasets),
            "datasets": category_datasets,
        }
        logger.info(f"  {bronze_dir.name}: {len(category_datasets)} datasets")

    manifest["summary"] = {
        "total_datasets": total_datasets,
        "total_files": total_files,
        "total_size_gb": round(total_size_bytes / (1024**3), 2),
        "categories": len(bronze_categories),
        "empty_datasets": sum(1 for d in manifest["datasets"].values() if d["empty"]),
    }

    # Write
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n✅ MANIFEST.json written to {MANIFEST_PATH}")
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Total datasets: {total_datasets}")
    logger.info(f"   Total files: {total_files:,}")
    logger.info(f"   Total size: {total_size_bytes / (1024**3):.2f} GB")
    logger.info(f"   Empty datasets: {manifest['summary']['empty_datasets']}")

    return manifest


def update_manifest_entry(dataset_name: str, category: str, bronze_dir: Path) -> None:
    """Update a single dataset entry in MANIFEST.json after download.

    Reads the existing manifest, updates/adds the entry for the given dataset,
    refreshes the layer summary, and writes back. Much faster than a full
    generate_manifest() call.

    Args:
        dataset_name: Name of the dataset (e.g. "cifar10").
        category: Dataset category (e.g. "vision").
        bronze_dir: Path to the dataset's bronze directory.
    """
    # Load existing or start fresh
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "data_lake_path": str(DATA_LAKE),
            "layers": {},
            "datasets": {},
            "summary": {},
        }

    # Scan the single dataset directory
    if bronze_dir.exists():
        files = list(bronze_dir.rglob("*"))
        file_list = [f for f in files if f.is_file()]
        file_count = len(file_list)
        dir_count = sum(1 for f in files if f.is_dir())

        try:
            size_bytes = sum(f.stat().st_size for f in file_list)
        except OSError:
            size_bytes = 0

        extensions = set()
        for f in file_list[:100]:
            if f.suffix:
                extensions.add(f.suffix.lower())

        # Map category to bronze folder name
        if category == "legacy":
            layer_name = "01_bronze"
        else:
            layer_name = f"01_bronze_{category}"

        ds_info = {
            "category": category,
            "path": str(bronze_dir.relative_to(DATA_LAKE)),
            "file_count": file_count,
            "dir_count": dir_count,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "empty": file_count == 0,
            "extensions": sorted(list(extensions)),
        }

        manifest["datasets"][dataset_name] = ds_info

        # Update layer summary
        if layer_name not in manifest.get("layers", {}):
            manifest["layers"][layer_name] = {"dataset_count": 0, "datasets": []}

        layer = manifest["layers"][layer_name]
        if dataset_name not in layer["datasets"]:
            layer["datasets"].append(dataset_name)
            layer["datasets"].sort()
            layer["dataset_count"] = len(layer["datasets"])

        # Update timestamp
        manifest["generated_at"] = datetime.now().isoformat()

        # Recalculate summary
        all_datasets = manifest.get("datasets", {})
        manifest["summary"] = {
            "total_datasets": len(all_datasets),
            "total_files": sum(d.get("file_count", 0) for d in all_datasets.values()),
            "total_size_gb": round(
                sum(d.get("size_mb", 0) for d in all_datasets.values()) / 1024, 2
            ),
            "categories": len(manifest.get("layers", {})),
            "empty_datasets": sum(1 for d in all_datasets.values() if d.get("empty")),
        }

        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"📋 MANIFEST.json updated: {dataset_name} ({ds_info['file_count']} files, {ds_info['size_mb']:.1f} MB)")


def get_manifest_datasets() -> set:
    """Return set of all dataset names from MANIFEST.json.

    Uses load_manifest() which enforces freshness. This is the preferred
    alternative to scanning bronze directories via iterdir().

    Returns:
        Set of dataset names.

    Raises:
        FileNotFoundError: If MANIFEST.json does not exist.
        StaleManifestError: If MANIFEST.json is too old.
    """
    manifest = load_manifest()
    return set(manifest.get("datasets", {}).keys())
