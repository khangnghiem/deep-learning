# Operational Runbooks & Protocols

> Repeatable step-by-step procedures for model training, experiment tracking, and data ingestion.

---

## Runbook Directory

| Runbook | Protocol | Purpose |
|---|---|---|
| [**`colab_training.md`**](colab_training.md) | **Google Colab Pro+ Training Runbook** | Mounting Drive, unpacking Gold `.tar.gz` bundles to NVMe, running headless `train.py`, and releasing compute units. |
| [**`dataset_ingestion.md`**](dataset_ingestion.md) | **Medallion Data Lake Ingestion Runbook** | Downloading to `0_landing/`, registering raw in `1_bronze/`, updating `MANIFEST.json`, and packing to `3_gold/`. |

> **MLflow tracking** procedures are documented in [`architecture/mlops_platform.md`](../architecture/mlops_platform.md) §2.
