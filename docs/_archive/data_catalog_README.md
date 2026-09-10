# Dataset Catalogs & Domain Registries

> Centralized catalog of supported datasets, medical imaging benchmarks, and public data sources.

---

## Domain Registries

| Catalog | Focus | Highlights |
|---|---|---|
| [**`medical_datasets.md`**](medical_datasets.md) | **Medical Imaging & Clinical Datasets** | Kvasir-SEG, CVC-ClinicDB, BKAI-IGH, ISIC, Chest X-ray, BraTS, PolypGen. Details modalities, resolutions, and licensing. |
| [**`public_data_sources.md`**](public_data_sources.md) | **Public Vision, NLP & Audio Benchmarks** | CIFAR-10, CIFAR-100, ImageNet, STL-10, Fashion-MNIST, PhysioNet, Kaggle competitions. |

---

## Ingestion & Catalog Standards

All datasets in these registries follow the **4-Tier Medallion lifecycle**:
1. Staged in `data/0_landing/`.
2. Stored raw in `data/1_bronze/<category>/<dataset>/` with `SOURCE.yaml`.
3. Registered in `data/MANIFEST.json`.
4. Curated into `data/2_silver/<category>/<dataset>/` (with Feature Store embeddings in `features/`).
5. Packaged into `data/3_gold/<category>/<dataset>_v<X>.tar.gz` for Colab NVMe training.

For the step-by-step ingestion runbook, see [**Medallion Dataset Ingestion Runbook**](../runbooks/dataset_ingestion.md).
