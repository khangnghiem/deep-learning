"""
Batch download multiple datasets from the catalog into the Medallion Data Lake.

Usage:
    python batch_download.py                   # Show summary
    python batch_download.py --list            # List catalog datasets
    python batch_download.py --list vision     # List vision datasets
    python batch_download.py cifar10           # Download specific dataset
    python batch_download.py --all             # Download all datasets
    python batch_download.py --priority        # Download priority practice datasets
    python batch_download.py --category vision # Download by category
    python batch_download.py --source kaggle   # Download by source
    python batch_download.py --resume          # Retry failed downloads
    python batch_download.py --parallel 4      # 4 concurrent downloads
"""

import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config.catalog import DATASETS, TOTAL_DATASETS, download_dataset, list_datasets, _parse_size  # noqa: E402
from src.config.paths import BRONZE  # noqa: E402

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

CACHE_DIR = Path.home() / ".cache" / "dl_downloads"
FAILED_FILE = CACHE_DIR / "failed.json"
STATS_FILE = CACHE_DIR / "stats.json"

PRIORITY_DATASETS = [
    "mnist", "fashion-mnist", "cifar10", "cifar100", "svhn", "stl10",
    "intel-image", "flowers", "dogs-vs-cats", "fruits360", "eurosat",
    "digit-recognizer", "natural-images", "sign-language",
    "chest-xray", "skin-cancer", "brain-tumor", "malaria", "covid-ct", "retinal-oct",
    "breast-ultrasound", "mura-xray", "lung-ct", "alzheimer-mri", "knee-mri",
    "imdb", "ag-news", "sst2", "spam", "emotion", "rotten-tomatoes",
    "gtzan", "heartbeat", "esc50",
    "titanic", "credit-fraud", "iris", "wine", "adult", "heart-disease",
    "stock-market", "energy-consumption", "covid-time-series", "store-sales",
    "sklearn-digits", "sklearn-california", "tiny-imagenet", "omniglot", "lfw",
]

SHORTCUT_FILTERS = {
    "tiny": {"max_mb": 100},
    "small": {"max_mb": 500},
    "vision": {"category": "vision"},
    "medical": {"category": "medical"},
    "nlp": {"category": "nlp"},
    "tabular": {"category": "tabular"},
    "timeseries": {"category": "timeseries"},
    "audio": {"category": "audio"},
    "education": {"category": "education"},
    "uci": {"source": "uci"},
    "sklearn": {"source": "sklearn"},
    "openml": {"source": "openml"},
    "tfds": {"source": "tfds"},
    "huggingface": {"source": "huggingface"},
    "ultrasound": {"modality": "ultrasound"},
    "xray": {"modality": "xray"},
    "ct": {"modality": "ct"},
    "mri": {"modality": "mri"},
    "polyp": {"modality": "endoscopy"},
}


# =============================================================================
# HELPERS
# =============================================================================

def get_filtered_datasets(category=None, source=None, modality=None, max_mb=None) -> list:
    """Filter catalog datasets matching criteria, sorted by size ascending."""
    results = []
    for name, info in DATASETS.items():
        if category and info.get("category") != category:
            continue
        if source and info.get("source") != source:
            continue
        if modality and info.get("modality") != modality:
            continue
        if max_mb and _parse_size(info.get("size", "0")) > max_mb:
            continue
        results.append(name)
    results.sort(key=lambda x: _parse_size(DATASETS[x].get("size", "0")))
    return results


def check_disk_space(dataset_names: list, skip_check=False) -> bool:
    """Check available disk space unless saving to Google Drive or skipped."""
    if skip_check or "/content/drive" in str(BRONZE) or "Google Drive" in str(BRONZE):
        return True
    
    total_est_mb = sum(_parse_size(DATASETS[d].get("size", "0")) for d in dataset_names if d in DATASETS)
    try:
        check_path = BRONZE if BRONZE.exists() else BRONZE.parent
        free_mb = shutil.disk_usage(check_path).free / (1024 * 1024)
    except Exception:
        return True
    
    if free_mb < (total_est_mb * 1.2):
        print(f"\n⚠️  LOW DISK SPACE: {free_mb/1024:.1f}GB available, ~{total_est_mb/1024:.1f}GB needed.")
        if not sys.stdin.isatty():
            print("Non-interactive mode. Use --no-space-check to force.")
            return False
        try:
            return input("Continue anyway? [y/N]: ").strip().lower() == "y"
        except (EOFError, KeyboardInterrupt):
            return False
    return True


def load_failed() -> dict:
    return json.loads(FAILED_FILE.read_text()) if FAILED_FILE.exists() else {}


def save_failed(data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_FILE.write_text(json.dumps(data, indent=2))


# =============================================================================
# CORE DOWNLOAD LOGIC
# =============================================================================

def download_single(name: str) -> dict:
    """Download a single dataset and record timing/status."""
    if name not in DATASETS:
        return {"name": name, "status": "failed", "error": "Unknown dataset"}
    
    size_mb = _parse_size(DATASETS[name].get("size", "0"))
    start = time.time()
    try:
        res = download_dataset(name)
        elapsed = time.time() - start
        if res == "skipped":
            return {"name": name, "status": "skipped", "duration": elapsed}
        elif res:
            speed = (size_mb / elapsed) if elapsed > 0 else 0
            return {"name": name, "status": "success", "duration": elapsed, "size_mb": size_mb, "speed_mbps": speed}
        return {"name": name, "status": "failed", "duration": elapsed, "error": "Download returned False"}
    except Exception as e:
        return {"name": name, "status": "failed", "duration": time.time() - start, "error": str(e)[:200]}


def download_batch(dataset_names: list, parallel=1, check_space=True):
    """Download a sequence of datasets sequentially or concurrently."""
    if not dataset_names:
        print("No datasets to download.")
        return
    
    if check_space and not check_disk_space(dataset_names):
        print("Download cancelled.")
        return
    
    total = len(dataset_names)
    print(f"\n📦 Downloading {total} dataset(s) (workers: {parallel})...\n")
    
    results = {"succeeded": [], "failed": [], "skipped": []}
    failed_tracker = load_failed()
    start_time = time.time()
    
    def handle_result(r):
        name = r["name"]
        if r["status"] == "success":
            results["succeeded"].append(name)
            failed_tracker.pop(name, None)
        elif r["status"] == "skipped":
            results["skipped"].append(name)
            failed_tracker.pop(name, None)
        else:
            results["failed"].append((name, r.get("error", "Unknown error")))
            failed_tracker[name] = {"error": r.get("error", "Unknown error"), "timestamp": datetime.now().isoformat()}
    
    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(download_single, name): name for name in dataset_names}
            iterator = tqdm(as_completed(futures), total=total, desc="Progress") if HAS_TQDM else as_completed(futures)
            for i, fut in enumerate(iterator, 1):
                res = fut.result()
                handle_result(res)
                if not HAS_TQDM:
                    print(f"[{i}/{total}] {res['name']}: {res['status']}")
    else:
        for i, name in enumerate(dataset_names, 1):
            size = DATASETS.get(name, {}).get("size", "?")
            print(f"[{i}/{total}] {name} ({size})...")
            res = download_single(name)
            handle_result(res)
            print(f"       Status: {res['status']}")
    
    save_failed(failed_tracker)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Succeeded: {len(results['succeeded'])}")
    print(f"⏭️  Skipped:   {len(results['skipped'])}")
    print(f"❌ Failed:    {len(results['failed'])}")
    print(f"⏱️  Duration:  {total_time/60:.1f}m")
    
    if results["failed"]:
        print("\nFailed datasets (use --resume to retry):")
        for name, err in results["failed"][:10]:
            print(f"  - {name}: {err}")


def show_summary():
    """Print catalog overview grouped by category and size."""
    print("\n📊 Dataset Catalog Summary\n")
    cats = {}
    for info in DATASETS.values():
        c = info.get("category", "other")
        cats.setdefault(c, {"count": 0, "mb": 0})
        cats[c]["count"] += 1
        cats[c]["mb"] += _parse_size(info.get("size", "0"))
    
    for c, data in sorted(cats.items()):
        size_str = f"{data['mb']/1024:.1f}GB" if data['mb'] > 1024 else f"{data['mb']:.0f}MB"
        print(f"  {c:<14} {data['count']:>3} datasets ({size_str})")
    
    failed = load_failed()
    if failed:
        print(f"\n⚠️  {len(failed)} failed downloads recorded (run with --resume to retry)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch download datasets into Medallion Data Lake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_download.py                     # Catalog summary
  python batch_download.py --list              # List all datasets
  python batch_download.py --list vision       # List vision datasets
  python batch_download.py cifar10             # Download single dataset
  python batch_download.py mnist cifar10       # Download multiple datasets
  python batch_download.py --priority          # Download priority DL datasets
  python batch_download.py --category vision   # Download entire category
  python batch_download.py --size 500          # Datasets < 500MB
  python batch_download.py --resume            # Retry failed downloads
        """
    )
    
    parser.add_argument("datasets", nargs="*", default=[], help="Specific dataset name(s) to download")
    parser.add_argument("--list", nargs="?", const="all", metavar="CAT", help="List datasets in catalog")
    parser.add_argument("--all", action="store_true", help="Download all datasets in catalog")
    parser.add_argument("--priority", action="store_true", help="Download priority practice datasets")
    parser.add_argument("--category", type=str, metavar="CAT", help="Download by category")
    parser.add_argument("--source", type=str, metavar="SRC", help="Download by source (kaggle, hf, etc.)")
    parser.add_argument("--size", type=int, metavar="MB", help="Download datasets smaller than MB")
    parser.add_argument("--resume", action="store_true", help="Retry previously failed downloads")
    parser.add_argument("--clear-failed", action="store_true", help="Clear failed downloads tracking")
    parser.add_argument("--parallel", "-p", type=int, default=1, metavar="N", help="Parallel worker threads")
    parser.add_argument("--no-space-check", action="store_true", help="Skip disk space validation")
    
    # Register shortcuts cleanly
    for flag, target in SHORTCUT_FILTERS.items():
        desc = f"Shortcut for {next(iter(target.keys()))}={next(iter(target.values()))}"
        parser.add_argument(f"--{flag}", action="store_true", help=desc)
    
    args = parser.parse_args()
    parallel = args.parallel
    check_space = not args.no_space_check
    
    if args.list:
        cat = None if args.list == "all" else args.list
        list_datasets(cat)
        print(f"\nTotal: {TOTAL_DATASETS} datasets in catalog")
    elif args.clear_failed:
        if FAILED_FILE.exists():
            FAILED_FILE.unlink()
        print("✅ Cleared failed downloads tracker.")
    elif args.resume:
        failed = load_failed()
        if not failed:
            print("✅ No failed downloads to retry.")
        else:
            download_batch(list(failed.keys()), parallel=parallel, check_space=False)
    elif args.datasets:
        download_batch(args.datasets, parallel=parallel, check_space=check_space)
    elif args.all:
        all_ds = sorted(DATASETS.keys(), key=lambda x: _parse_size(DATASETS[x].get("size", "0")))
        download_batch(all_ds, parallel=parallel, check_space=check_space)
    elif args.priority:
        valid = [d for d in PRIORITY_DATASETS if d in DATASETS]
        download_batch(valid, parallel=parallel, check_space=check_space)
    elif args.category:
        targets = get_filtered_datasets(category=args.category)
        download_batch(targets, parallel=parallel, check_space=check_space)
    elif args.source:
        targets = get_filtered_datasets(source=args.source)
        download_batch(targets, parallel=parallel, check_space=check_space)
    elif args.size:
        targets = get_filtered_datasets(max_mb=args.size)
        download_batch(targets, parallel=parallel, check_space=check_space)
    else:
        # Check if any shortcut was provided
        for flag, kwargs in SHORTCUT_FILTERS.items():
            if getattr(args, flag, False):
                targets = get_filtered_datasets(**kwargs)
                download_batch(targets, parallel=parallel, check_space=check_space)
                return
        show_summary()
        parser.print_help()


if __name__ == "__main__":
    main()
