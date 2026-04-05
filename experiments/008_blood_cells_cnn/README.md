# 008 — Blood Cell Classification (CNN)

**Blood cell type classification** from microscopy images using pretrained ResNet18.

## Classes (4)

| Class      | Description                           |
| ---------- | ------------------------------------- |
| EOSINOPHIL | White blood cell with bilobed nucleus |
| LYMPHOCYTE | Small round white blood cell          |
| MONOCYTE   | Largest type of white blood cell      |
| NEUTROPHIL | Most common white blood cell          |

## Data

- **Source**: `data_lake/01_bronze_medical/blood-cells/` (Kaggle dataset)
- **Silver**: `data_lake/02_silver/blood-cells/` (cleaned, 70/15/15 split)
- **Issues fixed**: Missing NEUTROPHIL in original TRAIN split, nested folders, deduplication

## Model

- **Architecture**: ResNet18 (ImageNet pretrained)
- **Input**: 224×224 RGB
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing
- **Early Stopping**: patience=10

## Run (Colab)

Open `008_blood_cells_cnn_train.ipynb` in Google Colab → Runtime → Change to GPU → Run all.

The notebook handles: data cleaning → training → evaluation → MLflow logging.

## MLflow

Experiment name: `blood-cells-resnet`
