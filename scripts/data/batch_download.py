"""
Batch download multiple datasets from the catalog.

Usage:
    python batch_download.py                   # Show summary
    python batch_download.py --all             # ALL datasets (skips existing)
    python batch_download.py --priority        # Top priority datasets
    python batch_download.py --category vision # Download by category
    python batch_download.py --source kaggle   # Download by source
    python batch_download.py --medical         # All medical imaging datasets
    python batch_download.py --education        # All education datasets
    python batch_download.py --ultrasound      # Ultrasound datasets only
    python batch_download.py --xray            # X-ray datasets only
    python batch_download.py --ct              # CT scan datasets only
    python batch_download.py --mri             # MRI datasets only
    python batch_download.py --resume          # Retry failed downloads
    python batch_download.py --parallel 4      # 4 concurrent downloads
"""

import sys
import os
from pathlib import Path

# Add repo root to sys.path to allow importing from src
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import time
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.config.catalog import DATASETS, download_dataset, _parse_size
from src.config.paths import BRONZE

# Try to import tqdm, fallback to simple progress
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Cache directory for tracking downloads
CACHE_DIR = Path.home() / ".cache" / "dl_downloads"
FAILED_FILE = CACHE_DIR / "failed.json"
STATS_FILE = CACHE_DIR / "stats.json"

# Priority datasets for deep learning practice
PRIORITY_DATASETS = [
    # Vision classics
    "mnist", "fashion-mnist", "cifar10", "cifar100", "svhn", "stl10",
    # Kaggle popular
    "intel-image", "flowers", "dogs-vs-cats", "fruits360", "eurosat",
    "digit-recognizer", "natural-images", "sign-language",
    # Medical (by modality: ultrasound, xray, ct, mri)
    "chest-xray", "skin-cancer", "brain-tumor", "malaria", "covid-ct", "retinal-oct",
    "breast-ultrasound", "mura-xray", "lung-ct", "alzheimer-mri", "knee-mri",
    # NLP
    "imdb", "ag-news", "sst2", "spam", "emotion", "rotten-tomatoes",
    # Audio
    "gtzan", "heartbeat", "esc50",
    # Tabular (expanded with classics)
    "titanic", "credit-fraud", "iris", "wine", "adult", "heart-disease",
    # Time Series (NEW)
    "stock-market", "energy-consumption", "covid-time-series", "store-sales",
    # Sklearn classics
    "sklearn-digits", "sklearn-california",
    # URL datasets
    "tiny-imagenet", "omniglot", "lfw",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_disk_space_mb():
    """Get available disk space in MB.
    
    Note: Google Drive mounted via Colab may not report accurate free space.
    The disk_usage check may return Colab's local disk space instead.
    """
    try:
        # Check the actual data directory
        check_path = BRONZE if BRONZE.exists() else BRONZE.parent
        total, used, free = shutil.disk_usage(check_path)
        return free / (1024 * 1024)
    except Exception:
        return float('inf')  # Assume unlimited if can't check


def is_google_drive_path():
    """Check if BRONZE is on Google Drive (Colab mount)."""
    return "/content/drive" in str(BRONZE) or "Google Drive" in str(BRONZE)


def estimate_download_size_mb(datasets):
    """Estimate total download size in MB."""
    return sum(_parse_size(DATASETS[name].get("size", "0")) for name in datasets if name in DATASETS)


def check_disk_space(datasets, skip_check=False):
    """Check if enough disk space for downloads. Returns True if OK."""
    if skip_check:
        return True
    
    # Skip check for Google Drive - it has its own quota management
    if is_google_drive_path():
        estimated_mb = estimate_download_size_mb(datasets)
        print(f"📦 Estimated download: {estimated_mb/1024:.1f}GB")
        print(f"ℹ️  Saving to Google Drive - space check skipped (uses Drive quota)")
        return True
    
    estimated_mb = estimate_download_size_mb(datasets)
    available_mb = get_available_disk_space_mb()
    
    # Need 20% buffer
    required_mb = estimated_mb * 1.2
    
    if available_mb < required_mb:
        print(f"\n⚠️  DISK SPACE WARNING")
        print(f"   Estimated download: {estimated_mb/1024:.1f}GB")
        print(f"   Available space:    {available_mb/1024:.1f}GB")
        print(f"   Recommended:        {required_mb/1024:.1f}GB (with 20% buffer)")
        
        # Handle non-interactive environments (Colab notebooks)
        try:
            import sys
            if not sys.stdin.isatty():
                print("\n⚠️  Non-interactive mode detected. Use --no-space-check to proceed.")
                return False
            response = input("\nContinue anyway? [y/N]: ").strip().lower()
            return response == 'y'
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Cancelled.")
            return False
    return True


def load_failed_downloads():
    """Load list of previously failed downloads."""
    if FAILED_FILE.exists():
        return json.loads(FAILED_FILE.read_text())
    return {}


def save_failed_downloads(failed: dict):
    """Save failed downloads to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_FILE.write_text(json.dumps(failed, indent=2))


def load_download_stats():
    """Load download statistics."""
    if STATS_FILE.exists():
        return json.loads(STATS_FILE.read_text())
    return {"downloads": []}


def save_download_stats(stats: dict):
    """Save download statistics."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, indent=2))


def format_speed(mb_per_sec):
    """Format download speed nicely."""
    if mb_per_sec >= 1:
        return f"{mb_per_sec:.1f} MB/s"
    else:
        return f"{mb_per_sec*1024:.0f} KB/s"


def format_duration(seconds):
    """Format duration nicely."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_single(name, show_progress=True):
    """Download a single dataset with timing and error handling."""
    if name not in DATASETS:
        return {"name": name, "status": "error", "error": "Unknown dataset"}
    
    info = DATASETS[name]
    size_mb = _parse_size(info.get("size", "0"))
    
    start_time = time.time()
    try:
        result = download_dataset(name)
        elapsed = time.time() - start_time
        
        if result == "skipped":
            return {"name": name, "status": "skipped", "duration": elapsed}
        elif result:
            speed = size_mb / elapsed if elapsed > 0 else 0
            return {
                "name": name, 
                "status": "success", 
                "duration": elapsed,
                "size_mb": size_mb,
                "speed_mbps": speed
            }
        else:
            return {"name": name, "status": "failed", "duration": elapsed, "error": "download returned False"}
    except Exception as e:
        elapsed = time.time() - start_time
        return {"name": name, "status": "failed", "duration": elapsed, "error": str(e)[:200]}


def download_batch(datasets, parallel=1, check_space=True):
    """Download a batch of datasets with progress tracking."""
    dataset_names = [d[0] if isinstance(d, tuple) else d for d in datasets]
    
    # Disk space check
    if check_space and not check_disk_space(dataset_names):
        print("Download cancelled.")
        return
    
    total = len(dataset_names)
    print(f"\n📦 Downloading {total} datasets...")
    if parallel > 1:
        print(f"   Using {parallel} parallel workers\n")
    else:
        print()
    
    results = {"succeeded": [], "failed": [], "skipped": []}
    stats = load_download_stats()
    failed_tracking = load_failed_downloads()
    
    start_time = time.time()
    
    if parallel > 1:
        # Parallel downloads
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(download_single, name, False): name for name in dataset_names}
            
            if HAS_TQDM:
                pbar = tqdm(as_completed(futures), total=total, desc="Downloading", unit="dataset")
            else:
                pbar = as_completed(futures)
                print(f"[0/{total}] Starting...")
            
            completed = 0
            for future in pbar:
                result = future.result()
                completed += 1
                _process_result(result, results, stats, failed_tracking)
                
                if HAS_TQDM:
                    pbar.set_postfix({"last": result["name"][:15], "status": result["status"]})
                else:
                    print(f"[{completed}/{total}] {result['name']}: {result['status']}")
    else:
        # Sequential downloads with progress
        if HAS_TQDM:
            pbar = tqdm(dataset_names, desc="Downloading", unit="dataset")
        else:
            pbar = dataset_names
        
        for i, name in enumerate(pbar):
            if HAS_TQDM:
                pbar.set_description(f"[{i+1}/{total}] {name[:20]}")
            else:
                size = DATASETS.get(name, {}).get("size", "?")
                print(f"\n[{i+1}/{total}] {name} ({size})")
            
            result = download_single(name)
            _process_result(result, results, stats, failed_tracking)
    
    total_time = time.time() - start_time
    
    # Save tracking data
    save_download_stats(stats)
    save_failed_downloads(failed_tracking)
    
    # Print summary
    _print_summary(results, total_time, stats)


def _process_result(result, results, stats, failed_tracking):
    """Process a download result."""
    name = result["name"]
    
    if result["status"] == "success":
        results["succeeded"].append(name)
        # Remove from failed tracking if it was there
        failed_tracking.pop(name, None)
        # Add to stats
        stats["downloads"].append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "duration": result.get("duration", 0),
            "size_mb": result.get("size_mb", 0),
            "speed_mbps": result.get("speed_mbps", 0)
        })
    elif result["status"] == "skipped":
        results["skipped"].append(name)
        failed_tracking.pop(name, None)
    else:  # failed
        results["failed"].append((name, result.get("error", "Unknown error")))
        failed_tracking[name] = {
            "error": result.get("error", "Unknown error"),
            "timestamp": datetime.now().isoformat()
        }


def _print_summary(results, total_time, stats):
    """Print download summary with speed stats."""
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Succeeded: {len(results['succeeded'])}")
    print(f"⏭️  Skipped:   {len(results['skipped'])}")
    print(f"❌ Failed:    {len(results['failed'])}")
    print(f"⏱️  Total time: {format_duration(total_time)}")
    
    # Speed stats from this session
    session_downloads = [d for d in stats.get("downloads", []) 
                        if d["name"] in results["succeeded"]]
    if session_downloads:
        speeds = [d["speed_mbps"] for d in session_downloads if d["speed_mbps"] > 0]
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            max_speed = max(speeds)
            fastest = max(session_downloads, key=lambda x: x["speed_mbps"])
            print(f"\n📈 Speed Stats:")
            print(f"   Average: {format_speed(avg_speed)}")
            print(f"   Fastest: {format_speed(max_speed)} ({fastest['name']})")
    
    if results["failed"]:
        print(f"\n❌ Failed datasets (use --resume to retry):")
        for name, error in results["failed"][:10]:  # Show first 10
            print(f"   - {name}: {error[:50]}")
        if len(results["failed"]) > 10:
            print(f"   ... and {len(results['failed']) - 10} more")


# =============================================================================
# COMMAND HANDLERS
# =============================================================================

def show_summary():
    """Show download summary by category and size."""
    print("\n📊 Dataset Summary\n")
    
    # By category
    categories = {}
    for name, info in DATASETS.items():
        cat = info.get("category", "other")
        if cat not in categories:
            categories[cat] = {"count": 0, "total_mb": 0}
        categories[cat]["count"] += 1
        categories[cat]["total_mb"] += _parse_size(info.get("size", "0"))
    
    print("By Category:")
    for cat in sorted(categories.keys()):
        c = categories[cat]
        size_str = f"{c['total_mb']/1024:.1f}GB" if c['total_mb'] > 1024 else f"{c['total_mb']:.0f}MB"
        print(f"  {cat:12} {c['count']:3d} datasets  ({size_str})")
    
    # By size tier
    print("\nBy Size:")
    tiers = [
        ("< 100MB", 0, 100),
        ("100MB-500MB", 100, 500),
        ("500MB-2GB", 500, 2048),
        ("> 2GB", 2048, 999999),
    ]
    for tier_name, min_mb, max_mb in tiers:
        count = sum(1 for i in DATASETS.values() 
                   if min_mb <= _parse_size(i.get("size", "0")) < max_mb)
        print(f"  {tier_name:15} {count:3d} datasets")
    
    print(f"\nTotal: {len(DATASETS)} datasets")
    
    # Show failed count if any
    failed = load_failed_downloads()
    if failed:
        print(f"\n⚠️  {len(failed)} datasets previously failed (use --resume to retry)")


def download_all(parallel=1):
    """Download ALL datasets."""
    datasets = sorted(DATASETS.keys(), key=lambda x: _parse_size(DATASETS[x].get("size", "0")))
    download_batch(datasets, parallel=parallel)


def download_by_size(max_mb, parallel=1):
    """Download datasets under specified size."""
    datasets = [
        name for name, info in DATASETS.items()
        if _parse_size(info.get("size", "999GB")) <= max_mb
    ]
    datasets.sort(key=lambda x: _parse_size(DATASETS[x].get("size", "0")))
    print(f"Found {len(datasets)} datasets under {max_mb}MB")
    download_batch(datasets, parallel=parallel)


def download_by_category(category, parallel=1):
    """Download datasets in a category."""
    datasets = [
        name for name, info in DATASETS.items()
        if info.get("category") == category
    ]
    if not datasets:
        print(f"❌ No datasets found in category: {category}")
        print("Available categories:", ", ".join(sorted(set(i.get("category", "other") for i in DATASETS.values()))))
        return
    datasets.sort(key=lambda x: _parse_size(DATASETS[x].get("size", "0")))
    print(f"Found {len(datasets)} {category} datasets")
    download_batch(datasets, parallel=parallel)


def download_by_source(source, parallel=1):
    """Download datasets from a specific source."""
    datasets = [
        name for name, info in DATASETS.items()
        if info.get("source") == source
    ]
    if not datasets:
        print(f"❌ No datasets found for source: {source}")
        print("Available sources:", ", ".join(sorted(set(i.get("source", "?") for i in DATASETS.values()))))
        return
    datasets.sort(key=lambda x: _parse_size(DATASETS[x].get("size", "0")))
    print(f"Found {len(datasets)} {source} datasets")
    download_batch(datasets, parallel=parallel)


def download_by_modality(modality, parallel=1):
    """Download medical datasets by imaging modality (ultrasound, xray, ct, mri)."""
    datasets = [
        name for name, info in DATASETS.items()
        if info.get("category") == "medical" and info.get("modality") == modality
    ]
    if not datasets:
        print(f"❌ No datasets found for modality: {modality}")
        # Show available modalities
        modalities = set(
            i.get("modality") for i in DATASETS.values() 
            if i.get("category") == "medical" and i.get("modality")
        )
        print("Available modalities:", ", ".join(sorted(modalities)))
        return
    
    datasets.sort(key=lambda x: _parse_size(DATASETS[x].get("size", "0")))
    total_size = sum(_parse_size(DATASETS[d].get("size", "0")) for d in datasets)
    
    icon = {"ultrasound": "🔊", "xray": "☢️", "ct": "🔬", "mri": "🧲"}.get(modality, "🏥")
    print(f"{icon} Found {len(datasets)} {modality.upper()} datasets ({total_size/1024:.1f}GB total)")
    for d in datasets:
        size = DATASETS[d].get("size", "?")
        print(f"   - {d}: {size}")
    print()
    download_batch(datasets, parallel=parallel)


def download_priority(parallel=1):
    """Download priority datasets."""
    valid = [d for d in PRIORITY_DATASETS if d in DATASETS]
    print(f"Downloading {len(valid)} priority datasets")
    download_batch(valid, parallel=parallel)


def download_resume(parallel=1):
    """Re-download previously failed datasets."""
    failed = load_failed_downloads()
    if not failed:
        print("✅ No failed downloads to retry!")
        return
    
    print(f"🔄 Retrying {len(failed)} failed datasets...")
    for name, info in failed.items():
        print(f"   - {name}: {info.get('error', '?')[:50]}")
    
    download_batch(list(failed.keys()), parallel=parallel, check_space=False)


def clear_failed():
    """Clear the failed downloads tracking."""
    if FAILED_FILE.exists():
        FAILED_FILE.unlink()
        print("✅ Cleared failed downloads tracking")
    else:
        print("No failed downloads to clear")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch download datasets from the catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_download.py                     # Show summary
  python batch_download.py --all               # Download all datasets
  python batch_download.py --all --parallel 4  # Download with 4 workers
  python batch_download.py --priority          # Top datasets for DL
  python batch_download.py --category vision   # Vision category
  python batch_download.py --source kaggle     # Kaggle datasets only
  python batch_download.py --medical           # All medical datasets
  python batch_download.py --ultrasound        # Ultrasound only (20+ datasets)
  python batch_download.py --xray              # X-ray only
  python batch_download.py --ct                # CT scan only
  python batch_download.py --mri               # MRI only
  python batch_download.py --size 500          # Datasets < 500MB
  python batch_download.py --resume            # Retry failed downloads
        """
    )
    
    # Download selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Download all datasets")
    group.add_argument("--priority", action="store_true", help="Download priority datasets for DL practice")
    group.add_argument("--category", type=str, metavar="CAT", help="Download by category (vision, medical, nlp, etc.)")
    group.add_argument("--source", type=str, metavar="SRC", help="Download by source (kaggle, huggingface, etc.)")
    group.add_argument("--size", type=int, metavar="MB", help="Download datasets smaller than MB")
    group.add_argument("--resume", action="store_true", help="Retry previously failed downloads")
    group.add_argument("--clear-failed", action="store_true", help="Clear failed downloads tracking")
    
    # Download options
    parser.add_argument("--parallel", "-p", type=int, default=1, metavar="N",
                       help="Number of parallel downloads (default: 1)")
    parser.add_argument("--no-space-check", action="store_true",
                       help="Skip disk space check")
    
    # Legacy shortcuts
    parser.add_argument("--tiny", action="store_true", help="Shortcut for --size 100")
    parser.add_argument("--small", action="store_true", help="Shortcut for --size 500")
    parser.add_argument("--vision", action="store_true", help="Shortcut for --category vision")
    parser.add_argument("--medical", action="store_true", help="Shortcut for --category medical")
    parser.add_argument("--nlp", action="store_true", help="Shortcut for --category nlp")
    parser.add_argument("--tabular", action="store_true", help="Shortcut for --category tabular")
    parser.add_argument("--timeseries", action="store_true", help="Shortcut for --category timeseries")
    parser.add_argument("--audio", action="store_true", help="Shortcut for --category audio")
    parser.add_argument("--education", action="store_true", help="Shortcut for --category education")
    parser.add_argument("--uci", action="store_true", help="Shortcut for --source uci")
    parser.add_argument("--sklearn", action="store_true", help="Shortcut for --source sklearn")
    parser.add_argument("--openml", action="store_true", help="Shortcut for --source openml")
    parser.add_argument("--tfds", action="store_true", help="Shortcut for --source tfds")
    parser.add_argument("--huggingface", "--hf", action="store_true", help="Shortcut for --source huggingface")
    
    # Medical modality shortcuts
    parser.add_argument("--ultrasound", action="store_true", help="Download ultrasound datasets (20+)")
    parser.add_argument("--xray", action="store_true", help="Download X-ray datasets")
    parser.add_argument("--ct", action="store_true", help="Download CT scan datasets")
    parser.add_argument("--mri", action="store_true", help="Download MRI datasets")
    parser.add_argument("--polyp", "--endoscopy", action="store_true", help="Download polyp/endoscopy datasets (Fast-Diag)")
    
    args = parser.parse_args()
    parallel = args.parallel
    
    # Handle shortcuts
    if args.tiny:
        download_by_size(100, parallel)
    elif args.small:
        download_by_size(500, parallel)
    elif args.vision:
        download_by_category("vision", parallel)
    elif args.medical:
        download_by_category("medical", parallel)
    elif args.nlp:
        download_by_category("nlp", parallel)
    elif args.tabular:
        download_by_category("tabular", parallel)
    elif args.timeseries:
        download_by_category("timeseries", parallel)
    elif args.audio:
        download_by_category("audio", parallel)
    elif args.education:
        download_by_category("education", parallel)
    elif args.uci:
        download_by_source("uci", parallel)
    elif args.sklearn:
        download_by_source("sklearn", parallel)
    elif args.openml:
        download_by_source("openml", parallel)
    elif args.tfds:
        download_by_source("tfds", parallel)
    elif args.huggingface:
        download_by_source("huggingface", parallel)
    # Medical modality shortcuts
    elif args.ultrasound:
        download_by_modality("ultrasound", parallel)
    elif args.xray:
        download_by_modality("xray", parallel)
    elif args.ct:
        download_by_modality("ct", parallel)
    elif args.mri:
        download_by_modality("mri", parallel)
    elif args.polyp:
        download_by_modality("endoscopy", parallel)
    # Handle main options
    elif args.all:
        download_all(parallel)
    elif args.priority:
        download_priority(parallel)
    elif args.category:
        download_by_category(args.category, parallel)
    elif args.source:
        download_by_source(args.source, parallel)
    elif args.size:
        download_by_size(args.size, parallel)
    elif args.resume:
        download_resume(parallel)
    elif args.clear_failed:
        clear_failed()
    else:
        show_summary()
        parser.print_help()


if __name__ == "__main__":
    main()
