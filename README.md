# Deep Learning Repository

PyTorch-based deep learning experiments with MLflow tracking.

## Quick Start

```bash
# Run an experiment
cd experiments/001_cifar10_cnn
python train.py

# View MLflow results
cd /path/to/My\ Drive/ops/mlflow
mlflow ui --backend-store-uri file://./mlruns
```

## Structure

```
deep-learning/
├── config/           # Path configs, MLflow setup
├── experiments/      # Reproducible training runs (train.py + config.yaml)
├── explorations/     # Interactive notebooks (EDA, prototyping, visualization)
├── src/              # Shared library code (models, transforms, training utils)
├── scripts/          # Utility scripts (create_experiment, batch_train)
└── tests/            # Unit & integration tests
```

## Data Lake (Medallion Architecture)

All data lives in `My Drive/data_lake/`:

| Layer      | Path             | Purpose                                           |
| ---------- | ---------------- | ------------------------------------------------- |
| Landing    | `00_landing/`    | Raw uploads, zips, untouched files                |
| Bronze     | `01_bronze_*/`     | Extracted raw data (CIFAR, MNIST, etc.)           |
| Silver     | `02_silver/`     | Cleaned, validated data                           |
| Gold       | `03_gold/`       | Analysis-ready datasets                           |
| Telemetry  | `07_telemetry/`  | Exported DuckDB logs / system telemetry           |
| Monitoring | `08_monitoring/` | Data drift reports & model health snapshots       |

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

## Development Workflow

This repo uses a **three-tier workflow** that separates interactive R&D from reproducible execution:

### 1. Explore → `explorations/{exp_name}.ipynb`

**Every experiment must be explored first.** The `explorations/` folder maps 1:1 with `experiments/`.
Use your matched Jupyter notebook for EDA, data visualization, architecture search,
augmentation experiments, and quick prototyping. No MLflow logging, no checkpoints.
When code stabilizes, graduate reusable pieces to `src/`.

### 2. Formalize → `experiments/NNN_*/train.py`

Copy `_template/`, implement `get_model()` and `get_dataloaders()`, tune
`config.yaml`. The `.py` script is the **source of truth** — config-driven,
MLflow-tracked, git-diffable, and testable. A thin `.ipynb` launcher handles
Colab execution: mount Drive → `%run train.py` → disconnect.

### 3. Share → `src/`

Reusable models, transforms, and training utilities imported by both
`explorations/` and `experiments/`. Tested via `pytest tests/`.

## Environment

- **Local**: `python train.py` (dev/debug only — training runs go to Colab)
- **Colab**: Open `train.ipynb` → Run all → Close tab
