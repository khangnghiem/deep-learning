# Medallion Data Lake Architecture & Feature Store

> The authoritative specification for the 4-tier Medallion data architecture, offline Feature Store, and model-ready bundle packaging.

---

## 1. Overview & Core Philosophy

The Data Lake is hosted in Google Drive under `data/` and structured as a **4-tier Medallion architecture** designed specifically for high-throughput deep learning on Google Colab Pro+:

```text
Google Drive: data/
├── MANIFEST.json                              # Single source of truth catalog (SHA-256, paths, counts)
│
├── 0_landing/                                 # Ephemeral Staging (Raw vendor archives before validation)
│   ├── kaggle/
│   │   ├── titanic/titanic.zip
│   │   └── credit_fraud/creditcardfraud.zip
│   ├── physionet/
│   │   └── ptb_xl/records100.tar.gz
│   ├── torchvision/
│   └── zenodo/
│       └── polypgen/polypgen2021_data.zip
│
├── 1_bronze/                                  # Immutable Raw Sources (Hierarchical by Modality)
│   ├── audio/
│   │   └── esc50/
│   │       ├── SOURCE.yaml                    # Provenance: URL, license, date, original checksum
│   │       └── raw/audio/*.wav
│   ├── tabular/
│   │   ├── titanic/
│   │   │   ├── SOURCE.yaml
│   │   │   └── raw/train.csv
│   │   └── credit_fraud/
│   │       ├── SOURCE.yaml
│   │       └── raw/creditcard.csv
│   ├── text/
│   │   └── imdb/
│   │       ├── SOURCE.yaml
│   │       └── raw/train.csv
│   ├── timeseries/
│   │   └── ecg_ptb_xl/
│   │       ├── SOURCE.yaml
│   │       └── raw/records100/
│   ├── video/
│   │   └── kinetics400/
│   │       ├── SOURCE.yaml
│   │       └── raw/clips/...
│   └── vision/
│       ├── cifar10/
│       │   ├── SOURCE.yaml
│       │   └── raw/batches.meta
│       ├── coco/
│       │   ├── SOURCE.yaml
│       │   └── raw/train2017/...
│       ├── kvasir_seg/                        # Medical vision (domain: medical in SOURCE.yaml)
│       │   ├── SOURCE.yaml
│       │   └── raw/
│       │       ├── images/cju0qkwjx3ki30801xkit505g.jpg
│       │       └── masks/cju0qkwjx3ki30801xkit505g.png
│       └── polypgen/
│           ├── SOURCE.yaml
│           └── raw/data/...
│
├── 2_silver/                                  # Curated Annotations, Masks & Feature Store
│   ├── features/                              # Global Multimodal Feature Store (.parquet)
│   │   ├── multimodal_biomarkers.parquet      # Standardized tabular biomarkers/features
│   │   └── ecg_timeseries_windows.parquet     # Extracted rolling stats & FFT features
│   ├── tabular/
│   │   └── titanic/
│   │       └── clean_features.parquet         # Conformed types, imputed missing values
│   └── vision/
│       └── kvasir_seg/
│           ├── annotations_coco.json          # Conformed labels (COCO bounding boxes & polygons)
│           ├── masks/                         # Standardized 1-channel binary masks (.png)
│           │   └── cju0qkwjx3ki30801xkit505g.png
│           ├── embeddings/                    # Precomputed Foundation Model Representations
│           │   ├── dinov2_vitb14.parquet      # 768-dim patch vectors per image_id
│           │   └── clip_vit_b32.parquet       # 512-dim visual embeddings
│           └── curation_report.json           # Data quality audit (dropped corrupt hashes)
│
└── 3_gold/                                    # Model-Ready Training Packages
    ├── tabular/
    │   └── titanic/
    │       ├── train.parquet                  # Direct columnar train split (no tar needed)
    │       ├── val.parquet                    # Direct columnar validation split
    │       └── titanic_v1.manifest.json       # SHA-256 digests and row split counts
    └── vision/
        └── kvasir_seg/
            ├── kvasir_seg_v1.tar.gz           # Leak-free train/val/test bundle for Colab NVMe
            └── kvasir_seg_v1.manifest.json    # SHA-256 digest, group-aware split stats
```

---

## 2. Layer Specifications

### 0_landing: Ephemeral Staging
- **Purpose**: Temporary staging buffer for vendor downloads (Kaggle API, PhysioNet, Zenodo).
- **Retention**: Ephemeral. Files are moved or deleted after ingestion into Bronze.
- **Zero-Duplication**: Move files into Bronze rather than copying. For trusted APIs (TorchVision, Hugging Face), download directly to Bronze.
- **Rule**: Never referenced by training scripts or feature jobs.

### 1_bronze: Immutable Raw Sources
- **Purpose**: Ground-truth raw data preserved exactly as delivered by upstream sources.
- **Hierarchy**: Partitioned strictly by 7 pure modalities: `audio/`, `multimodal/`, `tabular/`, `text/`, `timeseries/`, `video/`, `vision/`.
- **Provenance (`SOURCE.yaml`)**: Every dataset folder requires a `SOURCE.yaml` recording metadata:
  ```yaml
  name: kvasir_seg
  category: vision
  task: semantic_segmentation
  domain: medical
  url: "https://datasets.simula.no/kvasir-seg/"
  license: "CC BY 4.0"
  download_date: "2026-09-07"
  original_checksum: "sha256:abc123..."
  sample_count: 1000
  ```
- **Immutability**: Raw files must never be modified or normalized in-place.

### 2_silver: Curated Annotations, Masks & Feature Store
- **Purpose**: Standardized annotations, conformed masks, and offline precomputed representations.
- **No Image Duplication**: Raw images remain referenced from Bronze; Silver only stores conformed annotations and embeddings.
- **Mask Standardization (`masks/*.png`)**: Inconsistent vendor masks (RGB, boolean, polygons) are converted to standardized 1-channel grayscale PNGs (0 = background, 255 = target).
- **Conformed Annotations (`annotations_coco.json`)**: Bounding boxes and segmentation polygons converted to standard COCO JSON format.
- **Offline Feature Store**:
  - Resides in `2_silver/features/` (global tables) or `2_silver/<modality>/<dataset>/embeddings/` (dataset representations).
  - Compressed Apache Parquet format (`.parquet` with Snappy/ZSTD).
  - Stores high-dimensional vectors (CLIP, DINOv2 patch tokens, SAM features). Eliminates redundant backbone forward passes.

### 3_gold: Model-Ready Training Bundles
- **Purpose**: Verified, leak-free training packages optimized for Colab Pro+ GPU execution.
- **Bundle Packaging (`.tar.gz`) for Loose Files**: Multi-file datasets (images, masks, audio) are archived as a single compressed `.tar.gz` containing frozen `train/`, `val/`, and `test/` splits. Prevents FUSE streaming bottlenecks by unpacking to local NVMe SSD before training.
- **Direct Parquet Splits for Tabular & Features**: Saved directly as `train.parquet` and `val.parquet` without tar archives.
- **Companion Manifest (`.manifest.json`)**:
  - SHA-256 digest of the bundle/splits.
  - Group-aware splitting confirmation (zero cross-split leakage across subjects, scenes, or sessions).
  - Split counts and class distributions.

---

## 3. Dataset Catalog & Lineage Tracking

### MANIFEST.json Single Source of Truth
- Located at `data/MANIFEST.json`.
- Managed by `src/config/manifest.py`.
- Enforces instant catalog lookups (`load_manifest()`) without recursive cloud filesystem scanning.
- Automated update after ingestion: `update_manifest_entry(dataset_name, category, bronze_dir)`.

### MLflow Lineage Logging
Every experiment run automatically logs its data lineage:
```python
from src.data.mlflow_tracker import log_medallion_dataset

log_medallion_dataset(
    dataset_name="kvasir_seg",
    category="vision",
    gold_dir=gold_path,
    source_bronze="1_bronze/vision/kvasir_seg",
    split_counts={"train": 800, "val": 100, "test": 100},
)
```
This logs the Gold bundle digest, Bronze origin, and split seeds as immutable MLflow run tags.
