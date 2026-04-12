# 003 — Polyp Detection (Mask R-CNN / Detectron2)

Polyp detection and instance segmentation for colonoscopy images using Mask R-CNN.

## Architecture

- **Model**: Mask R-CNN R50-FPN (pretrained on COCO, fine-tuned on polyp datasets)
- **Training format**: COCO JSON (native — matches labeling tool export)
- **Output**: Bounding boxes + segmentation masks + confidence scores
- **Deployment**: Modal.com serverless GPU (T4)

## Datasets (6 polyp datasets, unified)

| Dataset | Source | Images | Masks |
|---------|--------|--------|-------|
| kvasir-seg | Kaggle | ~1000 | ✅ Binary masks |
| cvc-clinicdb | Kaggle | ~612 | ✅ Binary masks |
| cvc-colondb | Kaggle | ~380 | ✅ Binary masks |
| etis-larib | Kaggle | ~196 | ✅ Binary masks |
| ldpolypvideo | Google Drive | ~33k frames | ✅ Binary masks |
| polypgen | Synapse | ~3446 | ✅ Binary masks |

## Quick Start

```bash
# 1. Prepare dataset (run locally or on Colab)
python prepare_dataset.py --data-root "G:/My Drive/data_lake/01_bronze_medical" --output ./dataset

# 2. Train (on Colab with T4 GPU)
python train.py --dataset ./dataset --output "G:/My Drive/models/trained/polyp_detection"

# 3. Deploy (to Modal.com)
cd deploy && modal deploy modal_app.py
```

## File Structure

```
003_polyp_detection/
├── prepare_dataset.py    # Unify 6 datasets → COCO JSON
├── train.py              # Detectron2 training script
├── train.ipynb           # Colab notebook
├── evaluate.py           # Cross-dataset evaluation
└── deploy/
    ├── modal_app.py      # Modal.com serverless deployment
    └── requirements.txt  # Deployment dependencies
```
