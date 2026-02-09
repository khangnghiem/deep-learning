# AGENTS.md

This file provides guidance when working with code in this repository.

## Common commands

### Environment setup

- Install shared config: `pip install -e ../shared-config`
- Install dependencies: `pip install -r requirements.txt`

### Running experiments

- CIFAR-10 baseline: `cd experiments/001_cifar10_cnn && python train.py`
- New experiments: use `python scripts/create_experiment.py <dataset>` or copy `experiments/_template/`.

### Tests

- Full suite: `pytest tests -v`
- Unit only: `pytest tests/unit/test_models.py -v`
- Integration: `pytest tests/integration/test_training.py -v`

### MLflow UI

- `cd "/path/to/My Drive/mlflow" && mlflow ui --backend-store-uri file://./mlruns --port 5000`

## High-level architecture

### Dependencies

- **`shared-config`** (sibling repo `../shared-config`): Provides `shared_config.paths` (environment-aware paths, data lake, MLflow config) and `shared_config.catalog` (100+ dataset catalog).
- Data ingestion scripts live in **`../data-ingestion`** (separate repo).

### Repository layout

- `src/data/` — Dataset loaders, transforms, augmentation. Imports paths from `shared_config.paths`.
- `src/models/` — `SimpleCNN`, `MLP`, `get_pretrained_resnet`, `get_pretrained_vit`, medical models, U-Net.
- `src/training/` — `Trainer` class, `EarlyStopping`, checkpoint utilities. Uses `shared_config.paths` for MLflow and model saving.
- `src/utils/` — Metrics, visualization.
- `experiments/` — Self-contained numbered experiments (e.g. `001_cifar10_cnn`). Each has `config.yaml`, `train.py`, `README.md`.
- `scripts/` — `create_experiment.py` (generates new experiments), `batch_train.py`.
- `tests/` — Unit tests for models/losses, integration tests for training loop.

## Agent usage notes

- Import paths/MLflow from `shared_config.paths`, never hardcode.
- Import `DATASETS` from `shared_config.catalog` when needed by `create_experiment.py`.
- **NEVER run training locally** — use Google Colab.
- When creating new experiments, follow patterns in `experiments/001_cifar10_cnn` and `_template/`.
