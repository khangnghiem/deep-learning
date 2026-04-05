#!/usr/bin/env python3
"""
Data Lake Audit & Cleanup Script
=================================

Identifies and fixes orphaned folders in the data lake:
- Empty landing folders (failed downloads)
- Landing folders without corresponding bronze folders
- Empty bronze folders

Usage:
    python scripts/audit_data_lake.py              # Audit only (dry run)
    python scripts/audit_data_lake.py --fix        # Cleanup empty folders
    python scripts/audit_data_lake.py --redownload # Re-download failed datasets
"""

import sys
import argparse
from pathlib import Path

from shared_config.paths import LANDING, DATA_LAKE, get_bronze_path, get_all_bronze_paths
from shared_config.catalog import DATASETS, download_dataset


def get_all_bronze_datasets() -> set:
    """Get all dataset names across all bronze category folders."""
    # Prefer MANIFEST.json over filesystem scan
    try:
        from shared_config.manifest import get_manifest_datasets
        datasets = get_manifest_datasets()
        print("📋 Using MANIFEST.json (skipping filesystem scan)")
        return datasets
    except Exception as e:
        print(f"⚠️  MANIFEST unavailable ({e}), falling back to filesystem scan")

    datasets = set()
    for bronze_path in get_all_bronze_paths():
        if bronze_path.exists():
            for d in bronze_path.iterdir():
                if d.is_dir():
                    datasets.add(d.name)
    # Also check legacy 01_bronze
    legacy_bronze = DATA_LAKE / "01_bronze"
    if legacy_bronze.exists():
        for d in legacy_bronze.iterdir():
            if d.is_dir():
                datasets.add(d.name)
    return datasets


def get_landing_datasets() -> dict:
    """Get all datasets in landing zone, organized by source."""
    landing_data = {}
    if not LANDING.exists():
        return landing_data
    
    for source_dir in LANDING.iterdir():
        if source_dir.is_dir():
            source = source_dir.name
            landing_data[source] = {}
            for ds_dir in source_dir.iterdir():
                if ds_dir.is_dir():
                    # Check if empty
                    is_empty = not any(ds_dir.iterdir())
                    has_zip = any(f.suffix in ['.zip', '.tar', '.gz', '.tgz'] for f in ds_dir.glob('*'))
                    landing_data[source][ds_dir.name] = {
                        'path': ds_dir,
                        'empty': is_empty,
                        'has_zip': has_zip,
                    }
    return landing_data


def audit_data_lake():
    """Audit the data lake and return issues found."""
    print("🔍 Auditing Data Lake...")
    print("=" * 70)
    
    bronze_datasets = get_all_bronze_datasets()
    landing_data = get_landing_datasets()
    
    issues = {
        'empty_landing': [],
        'orphaned_landing': [],  # In landing but not in bronze
        'empty_bronze': [],
    }
    
    # Check landing folders
    for source, datasets in landing_data.items():
        for name, info in datasets.items():
            if info['empty']:
                issues['empty_landing'].append((source, name, info['path']))
            elif name not in bronze_datasets:
                # Has files but no bronze folder - extraction might have failed
                issues['orphaned_landing'].append((source, name, info['path']))
    
    # Check for empty bronze folders
    for bronze_path in get_all_bronze_paths():
        if bronze_path.exists():
            for d in bronze_path.iterdir():
                if d.is_dir() and not any(d.iterdir()):
                    issues['empty_bronze'].append(d)
    
    # Report
    print(f"\n📊 AUDIT RESULTS")
    print("-" * 50)
    print(f"  Bronze datasets:     {len(bronze_datasets)}")
    print(f"  Landing sources:     {len(landing_data)}")
    
    total_landing = sum(len(ds) for ds in landing_data.values())
    print(f"  Landing datasets:    {total_landing}")
    
    print(f"\n⚠️  ISSUES FOUND")
    print("-" * 50)
    
    if issues['empty_landing']:
        print(f"\n❌ Empty landing folders ({len(issues['empty_landing'])}):")
        print("   (Download started but failed - no files downloaded)")
        for source, name, path in issues['empty_landing'][:10]:
            reason = _get_failure_reason(name)
            print(f"   - [{source}] {name} {reason}")
        if len(issues['empty_landing']) > 10:
            print(f"   ... and {len(issues['empty_landing']) - 10} more")
    
    if issues['orphaned_landing']:
        print(f"\n⚠️  Orphaned landing folders ({len(issues['orphaned_landing'])}):")
        print("   (Has zip files but no bronze folder - extraction may have failed)")
        for source, name, path in issues['orphaned_landing'][:10]:
            print(f"   - [{source}] {name}")
        if len(issues['orphaned_landing']) > 10:
            print(f"   ... and {len(issues['orphaned_landing']) - 10} more")
    
    if issues['empty_bronze']:
        print(f"\n❌ Empty bronze folders ({len(issues['empty_bronze'])}):")
        for path in issues['empty_bronze'][:10]:
            print(f"   - {path.name}")
    
    if not any(issues.values()):
        print("\n✅ No issues found! Data lake is clean.")
    
    return issues


def _get_failure_reason(dataset_name: str) -> str:
    """Try to determine why a dataset failed to download."""
    if dataset_name not in DATASETS:
        return "(unknown dataset)"
    
    info = DATASETS[dataset_name]
    kaggle_id = info.get('kaggle_id', '')
    
    if kaggle_id.startswith('competitions/'):
        return "⚠️ (requires Kaggle competition rule acceptance)"
    if info.get('auth'):
        return "🔑 (requires authentication)"
    if info.get('size', '').endswith('GB'):
        size_gb = float(info['size'].replace('GB', ''))
        if size_gb > 10:
            return f"📦 (large: {info['size']} - may have timed out)"
    return ""


def fix_empty_folders(issues: dict, dry_run: bool = True):
    """Remove empty folders from landing and bronze."""
    action = "Would remove" if dry_run else "Removing"
    
    print(f"\n🧹 {'DRY RUN - ' if dry_run else ''}CLEANING UP EMPTY FOLDERS")
    print("=" * 70)
    
    removed = 0
    
    # Remove empty landing folders
    for source, name, path in issues['empty_landing']:
        print(f"  {action}: {path}")
        if not dry_run:
            try:
                path.rmdir()
                removed += 1
            except OSError as e:
                print(f"    ❌ Failed: {e}")
    
    # Remove empty bronze folders
    for path in issues['empty_bronze']:
        print(f"  {action}: {path}")
        if not dry_run:
            try:
                path.rmdir()
                removed += 1
            except OSError as e:
                print(f"    ❌ Failed: {e}")
    
    if dry_run:
        total = len(issues['empty_landing']) + len(issues['empty_bronze'])
        print(f"\n💡 Run with --fix to remove {total} empty folders")
    else:
        print(f"\n✅ Removed {removed} empty folders")
    
    return removed


def redownload_failed(issues: dict, max_size_mb: int = 2000):
    """Re-download datasets that failed (excluding large ones and competitions)."""
    print(f"\n🔄 RE-DOWNLOADING FAILED DATASETS (max {max_size_mb}MB)")
    print("=" * 70)
    
    to_download = []
    skipped = []
    
    for source, name, path in issues['empty_landing']:
        if name not in DATASETS:
            skipped.append((name, "not in catalog"))
            continue
        
        info = DATASETS[name]
        kaggle_id = info.get('kaggle_id', '')
        
        # Skip competitions (require manual rule acceptance)
        if kaggle_id.startswith('competitions/'):
            skipped.append((name, "competition - accept rules first"))
            continue
        
        # Skip authenticated datasets
        if info.get('auth'):
            skipped.append((name, "requires authentication"))
            continue
        
        # Check size
        size_str = info.get('size', '0')
        size_mb = _parse_size_mb(size_str)
        if size_mb > max_size_mb:
            skipped.append((name, f"too large ({size_str})"))
            continue
        
        to_download.append((name, info))
    
    # Show what will be skipped
    if skipped:
        print(f"\n⏭️  Skipping {len(skipped)} datasets:")
        for name, reason in skipped[:10]:
            print(f"   - {name}: {reason}")
        if len(skipped) > 10:
            print(f"   ... and {len(skipped) - 10} more")
    
    # Download
    if to_download:
        print(f"\n📥 Downloading {len(to_download)} datasets...")
        for name, info in to_download:
            print(f"\n{'='*50}")
            print(f"📥 {name} ({info.get('size', '?')})")
            print('='*50)
            download_dataset(name)
    else:
        print("\n✅ No datasets to re-download")
    
    return len(to_download)


def _parse_size_mb(size_str: str) -> float:
    """Parse size string to MB."""
    size_str = size_str.upper().replace(" ", "")
    if "GB" in size_str:
        return float(size_str.replace("GB", "")) * 1024
    elif "MB" in size_str:
        return float(size_str.replace("MB", ""))
    elif "KB" in size_str:
        return float(size_str.replace("KB", "")) / 1024
    elif "TB" in size_str:
        return float(size_str.replace("TB", "")) * 1024 * 1024
    return 0


def main():
    parser = argparse.ArgumentParser(description="Audit and cleanup data lake")
    parser.add_argument('--fix', action='store_true', 
                        help='Remove empty folders (default: dry run)')
    parser.add_argument('--redownload', action='store_true',
                        help='Re-download failed datasets')
    parser.add_argument('--max-size', type=int, default=2000,
                        help='Max size in MB for re-download (default: 2000)')
    parser.add_argument('--refresh-manifest', action='store_true',
                        help='Regenerate MANIFEST.json before auditing')
    args = parser.parse_args()
    
    if args.refresh_manifest:
        from shared_config.manifest import generate_manifest
        generate_manifest()
    
    issues = audit_data_lake()
    
    if args.fix:
        fix_empty_folders(issues, dry_run=False)
    elif issues['empty_landing'] or issues['empty_bronze']:
        fix_empty_folders(issues, dry_run=True)
    
    if args.redownload:
        redownload_failed(issues, max_size_mb=args.max_size)


if __name__ == "__main__":
    main()
