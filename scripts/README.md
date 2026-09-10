# Scripts

Utility and workflow automation scripts for the deep-learning repository.

## Directory Structure

```
scripts/
├── README.md                  # Documentation for scripts
├── create_experiment.py       # Scaffold new experiments from _template/
├── batch_train.py             # Batch run experiments with GPU memory tracking
├── batch_download.py          # Canonical CLI dataset downloader (Kaggle, HF, URLs, UCI)
└── audit_data_lake.py         # Health-check and audit Data Lake landing and bronze tiers
```

---

## 1. Experiment Scaffolding

### `create_experiment.py`
Generates a new experiment directory by copying `experiments/_template/` and configuring `config.yaml` with dataset information, smart hyperparameter defaults, and paired exploration notebooks.

```bash
# Scaffold an experiment for a cataloged dataset
python scripts/create_experiment.py intel-image

# Force a specific experiment number
python scripts/create_experiment.py intel-image --number 12

# List all available datasets from catalog
python scripts/create_experiment.py --list

# List datasets that don't have an experiment yet
python scripts/create_experiment.py --list-pending
```

---

## 2. Batch Training Execution

### `batch_train.py`
Executes experiments sequentially on Google Colab or remote machines, tracking GPU VRAM usage, managing garbage collection, handling crash recovery, and recording results to `completed.json`.

```bash
# List experiments and completion status
python scripts/batch_train.py --list

# Run specific experiments sequentially
python scripts/batch_train.py 007 008 009

# Run a sequential range of experiments
python scripts/batch_train.py --range 007 020

# Run all pending (unfinished) experiments
python scripts/batch_train.py --pending
```

---

## 3. Data Lake Ingestion & Auditing

### `batch_download.py`
The single unified CLI for ingesting datasets into the Medallion Data Lake (`data/1_bronze/<modality>/<dataset>/`). Integrates with Kaggle, HuggingFace, UCI, and direct URLs, and supports parallel transfers and resuming interrupted jobs.

```bash
# List datasets available in catalog (optionally by category)
python scripts/batch_download.py --list
python scripts/batch_download.py --list vision

# Download a specific dataset
python scripts/batch_download.py cifar10
python scripts/batch_download.py mnist cifar10

# Download curated priority datasets for DL practice
python scripts/batch_download.py --priority

# Download by category or source
python scripts/batch_download.py --category vision
python scripts/batch_download.py --source kaggle

# Filter by size with parallel downloads
python scripts/batch_download.py --size 500 --parallel 4

# Resume previously failed downloads
python scripts/batch_download.py --resume
```

### `audit_data_lake.py`
Audits the Medallion Data Lake for orphaned landing folders, empty bronze stubs, and inconsistencies against `MANIFEST.json`.

```bash
# Dry-run audit
python scripts/audit_data_lake.py

# Automatically remove empty stub folders
python scripts/audit_data_lake.py --fix

# Re-download failed or missing datasets
python scripts/audit_data_lake.py --redownload
```

---

## Architectural Guidelines

- **Annotation Converters:** Reusable dataset format converters (COCO, YOLO, segmentation masks) live in [`src/data/converters/`](../src/data/converters/) as part of the core Python package.
- **Exploration & EDA:** Interactive `.ipynb` notebooks belong in [`notebooks/`](../notebooks/), not in `scripts/`.
- **Reproducible Experiments:** Experiment execution scripts and launcher notebooks belong in [`experiments/NNN_<dataset>_<model>/`](../experiments/).
