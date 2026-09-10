"""
MLflow Dataset Lineage Tracker for Medallion Architecture
=========================================================

Enables seamless dataset lineage tracking in MLflow without uploading large
binary archives into MLflow artifact storage.

Key features:
- Streaming SHA-256 checksum calculation for large tarballs.
- Standardized manifest generation for Gold archives (<dataset>_<version>.manifest.json).
- Native MLflow 2.x/3.x dataset logging using `mlflow.data.MetaDataset`.
- Structured Medallion run tags for filtering, auditing, and reproduction.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import mlflow
    import mlflow.data
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.data.meta_dataset import MetaDataset
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


def compute_file_sha256(filepath: Path | str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file in streaming chunks (memory-efficient).
    
    Args:
        filepath: Path to the target file.
        chunk_size: Byte chunk size (default 64 KB).
        
    Returns:
        Hex-encoded SHA-256 checksum string.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")
        
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_gold_manifest(
    archive_path: Path | str,
    dataset_name: str,
    category: str,
    version: str = "v1",
    split_strategy: str = "patient_isolated",
    splits: Optional[Dict[str, int]] = None,
    source_bronze: Optional[str] = None,
    curation_recipe: Optional[str] = None,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Generate a standardized JSON manifest for a Gold dataset archive.
    
    Args:
        archive_path: Path to the .tar.gz or split archive file.
        dataset_name: Standardized dataset identifier (e.g. 'kvasir_seg').
        category: Domain category (e.g. 'medical', 'vision', 'audio').
        version: Dataset release version (e.g. 'v1').
        split_strategy: Partitioning method (e.g. 'patient_isolated_70_15_15').
        splits: Sample counts per split {'train': 800, 'val': 100, 'test': 100}.
        source_bronze: Relative path or identifier of raw Bronze source.
        curation_recipe: Script or git commit that produced Silver/Gold.
        output_path: Destination manifest path. If None, saves beside archive.
        
    Returns:
        Path to the written manifest file.
    """
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Gold archive does not exist: {archive}")
        
    sha256 = compute_file_sha256(archive)
    manifest_data = {
        "dataset_name": dataset_name,
        "category": category.lower(),
        "version": version,
        "archive_filename": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_size_mb": round(archive.stat().st_size / (1024 * 1024), 2),
        "sha256": sha256,
        "split_strategy": split_strategy,
        "splits": splits or {},
        "source_bronze": source_bronze or "",
        "curation_recipe": curation_recipe or "",
        "created_at": datetime.now().isoformat(),
    }
    
    if output_path is None:
        target = archive.parent / f"{dataset_name}_{version}.manifest.json"
    else:
        target = Path(output_path)
        
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(manifest_data, f, indent=2)
        
    return target


def log_medallion_dataset(
    archive_path: Path | str,
    manifest_path: Optional[Path | str] = None,
    context: str = "training",
    dataset_name: Optional[str] = None,
    category: Optional[str] = None,
    version: str = "v1",
    extra_tags: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Log Gold dataset provenance and lineage to the active MLflow run.
    
    Args:
        archive_path: Path to the packaged .tar.gz gold archive.
        manifest_path: Optional path to companion .manifest.json.
        context: Context of the dataset: 'training', 'validation', 'evaluation', etc.
        dataset_name: Dataset name override if manifest not provided.
        category: Category override if manifest not provided.
        version: Version override if manifest not provided.
        extra_tags: Additional metadata tags to log.
        
    Returns:
        Dictionary of logged metadata, or None if MLflow is not active.
    """
    if not _MLFLOW_AVAILABLE:
        return None
        
    active_run = mlflow.active_run()
    if not active_run:
        return None
        
    archive = Path(archive_path)
    meta = {}
    
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path, "r") as f:
            meta = json.load(f)
            
    # Resolve metadata fields with fallbacks
    ds_name = meta.get("dataset_name") or dataset_name or archive.stem.replace(".tar", "")
    cat = meta.get("category") or category or "default"
    ver = meta.get("version") or version
    sha256 = meta.get("sha256")
    
    if not sha256 and archive.exists():
        sha256 = compute_file_sha256(archive)
    elif not sha256:
        sha256 = "unknown_sha256"
        
    split_strategy = meta.get("split_strategy", "standard")
    splits = meta.get("splits", {})
    
    # 1. Structured Medallion tags
    tags = {
        "medallion.tier": "3_gold",
        "medallion.category": cat,
        "medallion.dataset": ds_name,
        "medallion.version": ver,
        "medallion.archive": archive.name,
        "medallion.sha256": sha256,
        "medallion.split_strategy": split_strategy,
    }
    
    for split_key, count in splits.items():
        tags[f"data.samples_{split_key}"] = str(count)
        
    if extra_tags:
        tags.update(extra_tags)
        
    mlflow.set_tags(tags)
    
    # 2. Native MLflow Dataset Logging (mlflow.data.log_input)
    try:
        source_uri = str(archive)
        src = resolve_dataset_source(source_uri)
        ds = MetaDataset(
            source=src,
            name=f"{ds_name}_{ver}",
            digest=sha256[:16],
        )
        mlflow.log_input(ds, context=context)
    except Exception as e:
        # Graceful degradation if MLflow data registry cannot resolve
        mlflow.set_tag("medallion.log_input_warning", str(e))
        
    return tags
