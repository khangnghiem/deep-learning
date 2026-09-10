# Google Drive Cloud Storage & 3-Pillar Architecture

> Authoritative specification for Google Drive directory organization, 3-Pillar topology, centralized archive hygiene, and FUSE mount behavior.

---

## 1. Top-Level Root Organization

Google Drive (`khangnghiem@gmail.com`) enforces a strict root-level policy with **zero loose files** and strictly **7 purposeful top-level directories**:

```text
My Drive/
├── data/                  # PILLAR 1 — Medallion Data Lake (Strict 4 Tiers)
├── models/                # PILLAR 2 — Model Assets (checkpoints, pretrained, registry)
├── ops/                   # PILLAR 3 — MLOps & Tracking (mlflow)
├── repos/                 # Cloned Git repositories for Colab execution
├── courses/               # Learning materials & coursework
├── Colab Notebooks/       # Colab auto-save directory
└── archive/               # CENTRALIZED ARCHIVE (all legacy & inactive assets)
```

Hidden system directories (`.agents/`, `.secrets/`, `.vscode/`) are permitted for tooling configuration.

---

## 2. Pillar Decomposition (With Concrete Nested Files)

### Pillar 1: `data/` — Medallion Data Lake
The authoritative data layer hosting all raw datasets, features, and model-ready bundles:

```text
My Drive/data/
├── MANIFEST.json                              # Single source of truth catalog (SHA-256, paths, counts)
│
├── 0_landing/                                 # Ephemeral staging buffers (raw vendor archives)
│   ├── kaggle/
│   │   ├── titanic/titanic.zip
│   │   └── credit_fraud/creditcardfraud.zip
│   ├── physionet/
│   │   └── ptb_xl/records100.tar.gz
│   └── zenodo/
│       └── polypgen/polypgen2021_data.zip
│
├── 1_bronze/                                  # Immutable Raw Sources (7 Pure Modality Categories)
│   ├── audio/
│   │   └── esc50/
│   │       ├── SOURCE.yaml                    # URL, license, date, original checksum
│   │       └── raw/audio/*.wav
│   ├── tabular/
│   │   ├── titanic/
│   │   │   ├── SOURCE.yaml
│   │   │   └── raw/train.csv
│   │   └── student_performance/               # (Merged from retired 'education' category)
│   │       ├── SOURCE.yaml
│   │       └── raw/student-por.csv
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
│   └── vision/                                # Computer Vision (general + medical vision datasets)
│       ├── cifar10/
│       │   ├── SOURCE.yaml
│       │   └── raw/batches.meta
│       ├── coco/
│       │   ├── SOURCE.yaml
│       │   └── raw/train2017/...
│       ├── kvasir_seg/                        # (domain: medical in SOURCE.yaml)
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
│   │   ├── multimodal_biomarkers.parquet      # Standardized lab metrics & tabular features
│   │   └── ecg_timeseries_windows.parquet     # Extracted rolling stats & FFT features
│   ├── tabular/
│   │   └── titanic/
│   │       └── clean_features.parquet         # Conformed types, imputed missing values
│   └── vision/
│       └── kvasir_seg/
│           ├── annotations_coco.json          # Conformed COCO bounding boxes & polygons
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

### Pillar 2: `models/` — Model Lifecycle Assets

Organized strictly by lifecycle phase (**no domain nesting**; application domain is tracked in MLflow metadata):

- `checkpoints/`: 1:1 mapping with `experiments/{NNN}_{dataset}_{model}/`.
- `pretrained/`: Shared foundation backbones (DINOv2, SAM2, ResNet) reused across domains.
- `registry/`: Versioned production inference artifacts (TorchScript, ONNX, TensorRT).

```text
My Drive/models/
├── checkpoints/                               # Active Training Checkpoints by Experiment (1:1 with experiments/)
│   ├── 001_cifar10_cnn/
│   │   ├── best.pt                            # Best validation metric checkpoint
│   │   ├── last.pt                            # Resumption checkpoint for Colab preemption
│   │   ├── config.yaml                        # Frozen hyperparameters
│   │   └── metrics.json                       # Final epoch metrics summary
│   └── 011_polyp_segmentation/
│       ├── best.pt                            # High-performing U-Net / SAM2 adapter weights
│       ├── last.pt
│       └── config.yaml
│
├── pretrained/                                # Foundation Weights Cache (Cross-Session & Cross-Domain)
│   ├── dinov2/
│   │   └── dinov2_vitb14_pretrain.pth         # 350MB DINOv2 vision backbone
│   ├── resnet/
│   │   └── resnet50-11ad3456.pth              # PyTorch official ImageNet weights
│   ├── sam2/
│   │   └── sam2_hiera_large.pt                # SAM2 foundation checkpoint (800MB)
│   └── yolo/
│       └── yolov8x-seg.pt                     # Pretrained YOLO segmentation weights
│
└── registry/                                  # Production-Ready Exported Models
    ├── cifar10_cnn_v1.torchscript.pt          # JIT-compiled TorchScript graph
    ├── kvasir_unet_v1.onnx                    # Exported ONNX graph with dynamic axes
    └── polyp_sam2_trt.engine                  # TensorRT optimized inference engine
```

---

### Pillar 3: `ops/` — MLOps & Tracking
Authoritative storage for persistent experiment tracking, parameters, and metrics:

```text
My Drive/ops/
└── mlflow/                                    # Persistent MLflow Central Store
    ├── mlflow.db                              # SQLite tracking database (runs, params, tags)
    └── mlruns/                                # MLflow artifacts storage
        ├── 0/                                 # Default experiment ID
        │   └── meta.yaml
        └── 1/                                 # Dedicated project experiment ID
            ├── <run_id_hash>/
            │   ├── artifacts/                 # Saved confusion matrices, ROC curves, configs
            │   ├── metrics/                   # Epoch loss, Dice, mIoU logs
            │   ├── params/                    # Learning rate, batch size, optimizer
            │   └── tags/                      # Medallion SHA-256 lineage tags
            └── meta.yaml
```

---

## 3. Centralized `archive/` Organization

All inactive, legacy, and non-production materials are consolidated under a single `archive/` root directory:

```text
My Drive/archive/
├── academic/                                  # University materials and coursework
│   └── hcmus/
│       ├── assignments/
│       └── thesis_drafts/
├── docs/                                      # Deprecated project specs and certificates
│   ├── fast_diag_specs_2025/
│   └── certificates/
├── ops/                                       # Deprecated tracking dumps & legacy runs
│   └── mlruns_legacy_2025/
├── personal_office/                           # Standalone Google Workspace documents
│   ├── gdocs/                                 # Old .gdoc narrative reports
│   ├── gsheet/                                # Old .gsheet budget and inventory tables
│   └── gslides/                               # Old presentation slide decks
├── 01_silver/                                 # Retired Silver datasets from pre-v1 era
├── schemas/                                   # Deprecated SQL/NoSQL schemas
└── scripts/                                   # Deprecated standalone ingestion scripts
```

**Retention Rule**: When retiring experiments, datasets, or documentation, never leave orphan folders in root or in `data/`. Move them into the appropriate `archive/` category.

---

## 4. Environment Path Resolution Matrix

All repository code accesses these Google Drive paths dynamically via [`src.config.paths`](file:///Users/khangnghiem/deep-learning/src/config/paths.py):

| Logical Variable | Local macOS Path (`khangnghiem`) | Google Colab Pro+ Headless Path |
| :--- | :--- | :--- |
| `DRIVE` | `~/Library/CloudStorage/GoogleDrive-khangnghiem@gmail.com/My Drive` | `/content/drive/MyDrive` |
| `DATA` (`DATA_LAKE`) | `.../My Drive/data` | `/content/drive/MyDrive/data` |
| `LANDING` | `.../My Drive/data/0_landing` | `/content/drive/MyDrive/data/0_landing` |
| `BRONZE` | `.../My Drive/data/1_bronze` | `/content/drive/MyDrive/data/1_bronze` |
| `SILVER` | `.../My Drive/data/2_silver` | `/content/drive/MyDrive/data/2_silver` |
| `FEATURE_STORE` | `.../My Drive/data/2_silver/features` | `/content/drive/MyDrive/data/2_silver/features` |
| `GOLD` | `.../My Drive/data/3_gold` | `/content/drive/MyDrive/data/3_gold` |
| `CHECKPOINTS` | `.../My Drive/models/checkpoints` | `/content/drive/MyDrive/models/checkpoints` |
| `PRETRAINED` | `.../My Drive/models/pretrained` | `/content/drive/MyDrive/models/pretrained` |
| `MLFLOW_TRACKING_URI` | `sqlite:///.../My Drive/ops/mlflow/mlflow.db` | `sqlite:////content/drive/MyDrive/ops/mlflow/mlflow.db` |

---

## 5. Google Drive FUSE Mount Rules & Gotchas

On local macOS machines (`~/Library/CloudStorage/GoogleDrive-...`) and Google Colab (`/content/drive/MyDrive`):

1. **Never Perform Recursive Directory Scans on FUSE**:
   Calling `os.walk()` or `Path.rglob()` on large Drive folders (e.g. Bronze or Checkpoints with >50,000 files) causes FUSE to hang indefinitely due to high API round-trip latency. Always read from `data/MANIFEST.json` or query specific known dataset paths.
2. **FileProvider Asynchronous Reparenting**:
   On macOS FileProvider, directory renames executed via low-level POSIX commands (`os.rename()`) on cloud-only folders can be delayed in syncing upstream. For cloud-wide structural reorganizations, use the Google Drive Web UI or Drive API.
3. **Multi-Account Verification**:
   Ensure browser profiles and local mounts are authenticated to the primary production account (`khangnghiem@gmail.com`), not secondary test accounts.
