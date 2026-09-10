# System & Data Architecture

> Architectural specifications for the Medallion Data Lake, Google Drive 3-Pillar topology, and MLOps platform.

This directory houses the living, authoritative specifications describing how data, models, and operations are structured and maintained.

---

## Architectural Specifications

| Specification | Document | Focus |
|---|---|---|
| **🏛️ Medallion Data Lake** | [**`data_lake.md`**](data_lake.md) | 4-Tier Medallion hierarchy (`0_landing`, `1_bronze`, `2_silver`, `3_gold`), Feature Store parquet tables, Gold `.tar.gz` bundle packaging, and `MANIFEST.json` single source of truth. |
| **💾 Google Drive Storage** | [**`drive_storage.md`**](drive_storage.md) | The 3-Pillars (`data/`, `models/`, `ops/`), root folder rules, centralized `archive/` organization, and Google Drive FUSE mount behavior. |
| **⚙️ MLOps Platform** | [**`mlops_platform.md`**](mlops_platform.md) | Colab Pro+ ephemeral execution, MLflow SQLite backend, Ray Tune hyperparameter studies, and Model Registry. |

---

## Architectural Invariants (Must Never Violate)

1. **Zero Direct FUSE Streaming**: Training scripts on Google Colab must **never** stream raw image files over Google Drive FUSE. Gold datasets must be archived as single `.tar.gz` packages, copied to Colab's `/content/` NVMe SSD, and unpacked locally before DataLoader execution.
2. **Strict 4-Tier Medallion Layout**: Data lake paths under `data/` must adhere strictly to `0_landing/`, `1_bronze/<modality>/`, `2_silver/<modality>/`, and `3_gold/<modality>/`. Domains are metadata attributes, not directory levels.
3. **Lifecycle-First Models Topology**: `models/` is organized strictly by lifecycle phase (`checkpoints/`, `pretrained/`, `registry/`) without application domain subfolders. Checkpoint folders map 1:1 to `experiments/` runs.
4. **Strict Lowercase Snake Case**: All directory and file names across Google Drive and the codebase must strictly use lowercase `snake_case` (no spaces, no camelCase, no kebab-case).
5. **Single Source of Truth Catalog**: Ingestion scripts must regenerate or update `data/MANIFEST.json` to prevent crawling 200K+ files on cloud storage.
