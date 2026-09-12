#!/usr/bin/env python3
"""
Data Lake Audit & Cleanup Script
=================================

Identifies and fixes orphaned or failed folders in the Medallion Data Lake:
- Empty landing folders (download started but failed)
- Orphaned landing folders (archive downloaded but bronze folder missing)
- Empty bronze folders

Usage:
    python scripts/audit_data_lake.py              # Dry-run audit report
    python scripts/audit_data_lake.py --fix        # Delete empty folders
    python scripts/audit_data_lake.py --redownload # Re-download failed datasets
"""

import sys
import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config.paths import LANDING, DATA_LAKE, get_all_bronze_paths  # noqa: E402
from src.config.catalog import DATASETS, download_dataset, _parse_size  # noqa: E402


def get_all_bronze_datasets() -> set:
    """Return all dataset names present across bronze folders."""
    try:
        from src.config.manifest import get_manifest_datasets
        return get_manifest_datasets()
    except Exception:
        pass

    datasets = set()
    for bronze_path in get_all_bronze_paths():
        if bronze_path.exists():
            datasets.update(d.name for d in bronze_path.iterdir() if d.is_dir())
    
    # Check legacy 01_bronze
    legacy = DATA_LAKE / "01_bronze"
    if legacy.exists():
        datasets.update(d.name for d in legacy.iterdir() if d.is_dir())
    return datasets


def get_landing_datasets() -> dict:
    """Scan landing zone and return dict of {source: {name: info}}."""
    landing_data = {}
    if not LANDING.exists():
        return landing_data

    archive_exts = {".zip", ".tar", ".gz", ".tgz"}
    for source_dir in LANDING.iterdir():
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        landing_data[source] = {}
        for ds_dir in source_dir.iterdir():
            if ds_dir.is_dir():
                is_empty = not any(ds_dir.iterdir())
                has_zip = any(f.suffix in archive_exts for f in ds_dir.glob("*"))
                landing_data[source][ds_dir.name] = {
                    "path": ds_dir,
                    "empty": is_empty,
                    "has_zip": has_zip,
                }
    return landing_data


def audit_data_lake() -> dict:
    """Audit the data lake and print structured report."""
    print("🔍 Auditing Medallion Data Lake...")
    print("=" * 70)

    bronze_datasets = get_all_bronze_datasets()
    landing_data = get_landing_datasets()

    issues = {
        "empty_landing": [],     # Download started but zero files created
        "orphaned_landing": [],  # Archive exists but bronze layer missing
        "empty_bronze": [],      # Empty directory in bronze
    }

    for source, datasets in landing_data.items():
        for name, info in datasets.items():
            if info["empty"]:
                issues["empty_landing"].append((source, name, info["path"]))
            elif name not in bronze_datasets:
                issues["orphaned_landing"].append((source, name, info["path"]))

    for bronze_path in get_all_bronze_paths():
        if bronze_path.exists():
            for d in bronze_path.iterdir():
                if d.is_dir() and not any(d.iterdir()):
                    issues["empty_bronze"].append(d)

    total_landing = sum(len(d) for d in landing_data.values())
    print(f"\n📊 Summary: {len(bronze_datasets)} bronze datasets, {total_landing} landing folders across {len(landing_data)} sources.")

    if issues["empty_landing"]:
        print(f"\n❌ Empty landing folders ({len(issues['empty_landing'])}):")
        for source, name, path in issues["empty_landing"][:10]:
            print(f"   - [{source}] {name}")
        if len(issues["empty_landing"]) > 10:
            print(f"   ... and {len(issues['empty_landing']) - 10} more")

    if issues["orphaned_landing"]:
        print(f"\n⚠️  Orphaned landing archives ({len(issues['orphaned_landing'])}):")
        for source, name, path in issues["orphaned_landing"][:10]:
            print(f"   - [{source}] {name} (unpack needed)")
        if len(issues["orphaned_landing"]) > 10:
            print(f"   ... and {len(issues['orphaned_landing']) - 10} more")

    if issues["empty_bronze"]:
        print(f"\n❌ Empty bronze folders ({len(issues['empty_bronze'])}):")
        for path in issues["empty_bronze"][:10]:
            print(f"   - {path.name}")

    if not any(issues.values()):
        print("\n✅ All clean! No orphaned or empty folders detected.")

    return issues


def fix_empty_folders(issues: dict, dry_run=True):
    """Delete empty folders from landing and bronze."""
    action = "Would remove" if dry_run else "Removing"
    print(f"\n🧹 {'[DRY RUN] ' if dry_run else ''}Cleaning empty folders...")
    
    removed = 0
    targets = [p for _, _, p in issues["empty_landing"]] + issues["empty_bronze"]
    for path in targets:
        print(f"  {action}: {path}")
        if not dry_run:
            try:
                path.rmdir()
                removed += 1
            except OSError as e:
                print(f"    ❌ Failed: {e}")

    if dry_run:
        print(f"\n💡 Run with --fix to remove {len(targets)} empty folder(s).")
    else:
        print(f"\n✅ Removed {removed} empty folder(s).")


def redownload_failed(issues: dict, max_size_mb=2000):
    """Re-download failed landing datasets that don't require manual auth/rules."""
    print(f"\n🔄 Re-downloading failed datasets (<= {max_size_mb}MB)...")
    to_download = []
    
    for _, name, _ in issues["empty_landing"]:
        if name not in DATASETS:
            continue
        info = DATASETS[name]
        if info.get("kaggle_id", "").startswith("competitions/"):
            continue  # Requires rule acceptance
        if info.get("auth"):
            continue  # Requires external authentication
        if _parse_size(info.get("size", "0")) > max_size_mb:
            continue
        to_download.append((name, info))

    if not to_download:
        print("✅ No eligible failed datasets to re-download.")
        return

    print(f"Found {len(to_download)} dataset(s) to re-download.")
    for name, info in to_download:
        print(f"\n📥 Downloading {name} ({info.get('size', '?')})...")
        download_dataset(name)


def main():
    parser = argparse.ArgumentParser(description="Audit and health-check Medallion Data Lake")
    parser.add_argument("--fix", action="store_true", help="Remove empty folders (default: dry run)")
    parser.add_argument("--redownload", action="store_true", help="Re-download failed datasets")
    parser.add_argument("--max-size", type=int, default=2000, help="Max size in MB for re-download (default: 2000)")
    parser.add_argument("--refresh-manifest", action="store_true", help="Regenerate MANIFEST.json before auditing")
    args = parser.parse_args()

    if args.refresh_manifest:
        try:
            from src.config.manifest import generate_manifest
            generate_manifest()
        except Exception as e:
            print(f"⚠️  Manifest generation failed: {e}")

    issues = audit_data_lake()

    if args.fix:
        fix_empty_folders(issues, dry_run=False)
    elif issues["empty_landing"] or issues["empty_bronze"]:
        fix_empty_folders(issues, dry_run=True)

    if args.redownload:
        redownload_failed(issues, max_size_mb=args.max_size)


if __name__ == "__main__":
    main()
