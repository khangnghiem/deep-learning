"""
Data Ingestion CLI
==================

Thin wrapper around the shared_config catalog for CLI usage.

Usage:
    python ingest.py --list              # List all datasets
    python ingest.py --list vision       # List vision datasets
    python ingest.py cifar10             # Download specific dataset
    python ingest.py --all-small         # Download all <500MB datasets
"""

import sys
from shared_config.catalog import (
    DATASETS, TOTAL_DATASETS, list_datasets, download_dataset, download_small, _parse_size
)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        category = sys.argv[2] if len(sys.argv) > 2 else None
        list_datasets(category)
        print(f"\nTotal: {TOTAL_DATASETS} datasets")
    elif sys.argv[1] == "--all-small":
        download_small()
    else:
        download_dataset(sys.argv[1])
