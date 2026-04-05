"""
Education Dataset Downloader
=============================

Thin wrapper around batch_download.py --education.

All education datasets are defined in shared_config.catalog.DATASETS.
This script simply delegates to batch_download for consistent behavior
(skip-existing, progress tracking, parallel support, resume).

Usage (in Colab):
    !python scripts/download_education.py               # Download all education datasets
    !python scripts/download_education.py --parallel 4   # With parallel downloads
    !python scripts/download_education.py --resume       # Retry failed only
"""

import sys
import subprocess
from pathlib import Path


def main():
    # Build the batch_download command
    script = Path(__file__).parent / "batch_download.py"
    cmd = [sys.executable, str(script), "--education"]

    # Forward any extra arguments (e.g., --parallel 4, --resume)
    cmd.extend(sys.argv[1:])

    print("🎓 Education Dataset Downloader")
    print(f"   Delegating to: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
