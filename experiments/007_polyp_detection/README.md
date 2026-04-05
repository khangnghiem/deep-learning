# 007 — Polyp Detection (YOLOv8)

Colonoscopy polyp detection using **YOLOv8m** fine-tuned on public datasets.

## Goal

- Detect all polyps with **high sensitivity** (recall ≥ 0.90)
- Draw bounding boxes around detected polyps
- Merge nearby polyps into a single bounding box

## Files

| File                 | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| `config.yaml`        | Hyperparameters and training config                               |
| `007_polyp_detection_prepare-data.ipynb` | ⬇️ Download Kvasir-SEG + CVC-ClinicDB, convert masks → YOLO boxes |
| `007_polyp_detection_train.ipynb`        | 🏋️ Fine-tune YOLOv8m, evaluate, run inference on custom images    |

## Quick Start

1. Open `007_polyp_detection_prepare-data.ipynb` in Colab → Run All (downloads ~500MB)
2. Open `007_polyp_detection_train.ipynb` in Colab (GPU runtime) → Run All (~1hr on T4)
3. Results saved to `models/trained/polyp_detection/` and `data_lake/06_inference/polyp_detection/`

## Data

- **Kvasir-SEG**: 1,000 colonoscopy images with segmentation masks
- **CVC-ClinicDB**: 612 images with segmentation masks
- **Your samples**: 17 images from `data_lake/00_landing/polyp/`
