---
id: ADR-0001
status: accepted
date: 2026-09-07
decision-makers: [khangnghiem]
---

# 0001 — Adopt 4-Tier Medallion Architecture with Hierarchical Bronze & Offline Feature Store

## Context

Raw datasets in Google Drive previously suffered from:
1. **Namespace Clutter**: 13+ unorganized top-level folders breaking abstraction.
2. **FUSE Bottlenecks**: Streaming 50,000+ loose images over Google Drive FUSE throttled I/O (<10 img/s).
3. **Data Leakage Risk**: Lack of frozen splits risked grouping leakage across subjects/sessions.
4. **Redundant Computation**: Vision backbones (SAM, CLIP, DINOv2) repeatedly extracted features on every run.

## Decision

Adopt a **4-tier Medallion Data Lake** under `data/` in Google Drive:

1. **4 Pure Layers**:
   - `0_landing/`: Ephemeral staging for vendor downloads; cleared after validation.
   - `1_bronze/`: Immutable raw data partitioned strictly by 7 pure modalities (`audio`, `multimodal`, `tabular`, `text`, `timeseries`, `video`, `vision`). Domains (`medical`, `finance`) are tracked as metadata tags in `SOURCE.yaml` and `catalog.py`.
   - `2_silver/`: Conformed annotations (`annotations_coco.json`), 1-channel masks (`masks/*.png`), and the **Feature Store** (`features/`, `embeddings/*.parquet`).
   - `3_gold/`: Model-ready training packages (`.tar.gz` bundles for multi-file datasets, direct `.parquet` for tabular, plus `.manifest.json` checksums).
2. **Zero FUSE Streaming**: Loose files are packaged into Gold `.tar.gz`, copied to Colab `/content/` NVMe, and extracted before training (>2,000 img/s).
3. **Automated Lineage**: Experiments log Gold SHA-256 and Bronze origin to MLflow via `src.data.log_medallion_dataset()`.

## Consequences

- **High Throughput**: Colab NVMe extraction saturates GPU compute.
- **Reproducibility**: Frozen Gold packages enforce group-aware, leakage-free splits.
- **Cost Savings**: Feature Store enables training heads on precomputed embeddings in seconds.
- **Trade-off**: Requires one-time packaging of Silver datasets into Gold bundles before training.
