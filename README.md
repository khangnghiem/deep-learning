# Deep Learning Repository

PyTorch-based deep learning experiments with MLflow tracking.

## Quick Start

```bash
# Run an experiment
cd experiments/001_cifar10_cnn
python train.py

# View MLflow results
cd /path/to/My\ Drive/mlflow
mlflow ui --backend-store-uri file://./mlruns
```

## Structure

```
deep-learning/
├── config/           # Path configs, MLflow setup
├── experiments/      # Individual experiments (001_, 002_, etc.)
├── explorations/     # Quick prototypes, notebooks
├── src/              # Shared code (models, transforms, data)
├── scripts/          # Utility scripts
└── tests/            # Unit tests
```

## Data Lake (Medallion Architecture)

All data lives in `My Drive/data_lake/`:

| Layer   | Path          | Purpose                                 |
| ------- | ------------- | --------------------------------------- |
| Landing | `00_landing/` | Raw uploads, zips, untouched files      |
| Bronze  | `01_bronze/`  | Extracted raw data (CIFAR, MNIST, etc.) |
| Silver  | `02_silver/`  | Cleaned, validated data                 |
| Gold    | `03_gold/`    | Analysis-ready datasets                 |

See [`config/paths.py`](config/paths.py) for path constants.

## Experiments

| #   | Dataset       | Model     | Status |
| --- | ------------- | --------- | ------ |
| 001 | CIFAR-10      | SimpleCNN | ✅     |
| 002 | CIFAR-100     | SimpleCNN | ✅     |
| 003 | Fashion-MNIST | SimpleCNN | ✅     |
| 004 | MNIST         | SimpleCNN | ✅     |
| 005 | SVHN          | SimpleCNN | ✅     |
| 006 | STL10         | ResNet18  | ✅     |

## Environment

- **Local**: `python train.py`
- **Colab**: Open `train.ipynb` → Run all
