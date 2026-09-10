# Medallion Dataset Ingestion Runbook

> Step-by-step protocol for downloading, validating, curating, and packaging datasets into the Medallion Data Lake.

---

## 1. Lifecycle Overview

```text
[External Source] (Kaggle / PhysioNet / Zenodo)
       │
       ▼ (Download via API / CLI)
[0_landing/] (Ephemeral download buffer)
       │
       ▼ (Extract & Validate Checksum)
[1_bronze/<category>/<dataset>/] (Immutable Raw Source + SOURCE.yaml)
       │
       ▼ (Update Catalog)
[data/MANIFEST.json]
       │
       ▼ (Clean Annotations, Generate Masks & Feature Store Embeddings)
[2_silver/<category>/<dataset>/]
       │
       ▼ (Create Leak-Free Splits & Bundle)
[3_gold/<category>/<dataset>_v1.tar.gz] + [.manifest.json]
```

---

## 2. Ingestion Steps

### Step 1: Download to Staging (`0_landing/`)
Using dataset ingestion utilities in `scripts/batch_download.py`:
```bash
# Example: Kaggle dataset download
kaggle datasets download -d <owner>/<dataset> -p "/content/drive/MyDrive/data/0_landing/kaggle/"
```

### Step 2: Register in Immutable Raw Bronze (`1_bronze/`)
Extract unmodified source files to the appropriate modality namespace:
```bash
mkdir -p "/content/drive/MyDrive/data/1_bronze/vision/kvasir_seg/raw"
unzip "/content/drive/MyDrive/data/0_landing/kaggle/kvasir-seg.zip" \
  -d "/content/drive/MyDrive/data/1_bronze/vision/kvasir_seg/raw"
```

Create `SOURCE.yaml` in the dataset root:
```yaml
name: kvasir_seg
category: vision
domain: medical
source_url: https://datasets.simula.no/kvasir-seg/
license: CC BY 4.0
access_date: "2026-09-07"
format: png
modality: endoscopy
original_checksum_sha256: 3a4b...
```

Clean up the staging archive from `0_landing/`.

### Step 3: Update MANIFEST.json Catalog
Update the catalog manifest without scanning the whole drive:
```python
from pathlib import Path
from src.config.manifest import update_manifest_entry

bronze_dir = Path("/content/drive/MyDrive/data/1_bronze/vision/kvasir_seg")
update_manifest_entry(
    dataset_name="kvasir_seg",
    category="vision",
    bronze_dir=bronze_dir,
)
```

### Step 4: Curate in Silver (`2_silver/`)
- Convert bounding boxes / polygons to conformed COCO/YOLO JSON (`annotations_coco.json`).
- Standardize segmentation mask values (e.g. 0 = background, 1 = polyp).
- Extract offline high-dimensional representations to `2_silver/features/` or `embeddings/`.

### Step 5: Package Model-Ready Gold Bundle (`3_gold/`)
1. Partition dataset strictly by group / subject / scene ID to prevent cross-split data leakage.
2. Structure directory with `train/`, `val/`, `test/`.
3. Create single compressed archive:
   ```bash
   tar -czf "/content/drive/MyDrive/data/3_gold/vision/kvasir_seg_v1.tar.gz" -C /content/kvasir_seg_splits .
   ```
4. Generate companion manifest with SHA-256 digest:
   ```python
   from src.data.mlflow_tracker import create_gold_manifest

   create_gold_manifest(
       archive_path=Path("/content/drive/MyDrive/data/3_gold/vision/kvasir_seg_v1.tar.gz"),
       dataset_name="kvasir_seg",
       version="v1",
       splits={"train": 800, "val": 100, "test": 100},
       category="vision",
   )
   ```
