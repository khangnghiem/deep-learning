# Deep Learning Documentation

> Architectural blueprints, operational runbooks, decision logs, and quality standards for model training, dataset engineering, and experiment tracking across Google Colab Pro+ and Google Drive.

---

## Navigation

| Pillar | Focus | What's Inside |
|---|---|---|
| **🏛️ [Architecture](architecture/README.md)** | **System & Data Specifications** | [Medallion Data Lake](architecture/data_lake.md), [Google Drive Storage](architecture/drive_storage.md), [MLOps Platform](architecture/mlops_platform.md) |
| **📜 [Decisions (ADRs)](decisions/README.md)** | **Architecture Decision Records** | [ADR-0001 Medallion Lake](decisions/ADR-0001-medallion-data-lake-architecture.md), [ADR-0002 3-Pillar Drive Layout](decisions/ADR-0002-google-drive-three-pillar-layout.md) |
| **🛠️ [Runbooks](runbooks/README.md)** | **Operational Procedures** | [Colab Pro+ Training](runbooks/colab_training.md), [Dataset Ingestion](runbooks/dataset_ingestion.md) |
| **🛡️ [Quality & Testing](quality.md)** | **TDD, Test Pyramid & Quality Gates** | [Testing Strategy & TDD Guide](quality.md) |

---

## How Docs Are Organized

```text
docs/
├── README.md                      ← Documentation portal (this file)
│
├── architecture/                  ← 🏛️ System & Data Specifications
│   ├── README.md                  ← Architecture overview & invariants
│   ├── data_lake.md               ← 4-Tier Medallion Data Lake, Feature Store, & Gold packaging
│   ├── drive_storage.md           ← Google Drive 3-Pillar structure (data, models, ops) & archive
│   └── mlops_platform.md          ← Colab Pro+ execution, MLflow tracking & lineage, Ray Tune
│
├── decisions/                     ← 📜 Architecture Decision Records (MADRs)
│   ├── README.md                  ← ADR index & status register
│   ├── ADR-0001-medallion-data-lake-architecture.md
│   └── ADR-0002-google-drive-three-pillar-layout.md
│
├── runbooks/                      ← 🛠️ Operational Runbooks
│   ├── README.md                  ← Runbooks index
│   ├── colab_training.md          ← Headless Colab execution, NVMe unpacking, auto-unassign
│   └── dataset_ingestion.md       ← Ingestion into 0_landing/1_bronze and manifest generation
│
├── quality.md                     ← 🛡️ Testing pyramid, TDD workflow, & quality gates
│
├── _template/                     ← 📝 Canonical Design Templates
│   ├── E-design-template.md       ← Epic Design Template (major research initiatives)
│   └── M-design-template.md       ← Model & Experiment Design Template
│
└── _archive/                      ← 📦 Deprecated documents (read-only)
```

> **Dataset catalogs**: The authoritative dataset catalog is [`src/config/catalog.py`](../src/config/catalog.py) (200+ datasets). Run `python -m src.config.catalog --list` to browse.

---

## Quick Links by Role

| Task / Role | Start Here | Key Reference |
|---|---|---|
| **Starting a new experiment** | Copy `experiments/_template/` or run `scripts/create_experiment.py` | [M-Design Template](_template/M-design-template.md) |
| **Launching on Colab Pro+** | Open `launch.ipynb` or thin wrapper notebook | [Colab Training Runbook](runbooks/colab_training.md) |
| **Adding a new dataset** | Stage in `0_landing/`, register in `1_bronze/<modality>/` | [Data Lake Architecture](architecture/data_lake.md) |
| **Tracking metrics & lineage** | Use `src.data.log_medallion_dataset()` and MLflow | [MLOps Platform §2](architecture/mlops_platform.md) |
| **Reviewing architecture decisions** | Review point-in-time design justifications | [Decisions Index](decisions/README.md) |
| **Writing unit & shape tests** | Run fast local smoke tests with `pytest` | [Quality & Testing](quality.md) |
