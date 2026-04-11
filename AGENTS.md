# AGENTS.md

This file provides guidance when working with code in this repository.

## Common commands

### Environment setup

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

- Data ingestion scripts live in **`../../data_lake/scripts`**.

### Repository layout

- `src/data/` — Dataset loaders, transforms, augmentation. Imports paths from `src.config.paths`.
- `src/models/` — `SimpleCNN`, `MLP`, `get_pretrained_resnet`, `get_pretrained_vit`, medical models, U-Net.
- `src/training/` — `Trainer` class, `EarlyStopping`, checkpoint utilities. Uses `src.config.paths` for MLflow and model saving.
- `src/utils/` — Metrics, visualization.
- `experiments/` — **Reproducible training runs**. Each has `config.yaml`, `train.py` (source of truth), thin `train.ipynb` (Colab launcher), `README.md`.
- `explorations/` — **Interactive notebooks** for EDA, prototyping, visualization, and architecture search. No MLflow logging or checkpoints.
- `scripts/` — `create_experiment.py` (generates new experiments), `batch_train.py`.
- `tests/` — Unit tests for models/losses, integration tests for training loop.

## Agent usage notes

- Import paths/MLflow from `src.config.paths`, never hardcode.
- Import `DATASETS` from `src.config.catalog` when needed by `create_experiment.py`.
- **NEVER run training locally** — use Google Colab.
- **Naming Convention:** All experiments, explorations, and folders MUST strictly follow the format `{NNN}_{dataset}_{model}` (e.g., `001_cifar10_cnn.ipynb` matching `experiments/001_cifar10_cnn/`).
  - `{NNN}`: 3-digit zero-padded sequential number.
  - `{dataset}`: The dataset or generic domain used.
  - `{model}`: The primary architecture or algorithm.
- When creating new experiments, follow patterns in `experiments/001_cifar10_cnn` and `_template/`.
- **`experiments/` = `.py` scripts** (train.py is the source of truth). Notebooks here are thin Colab launchers only.
- **`explorations/` = `.ipynb` notebooks** mapped 1:1 to `experiments/`. **Every experiment must be explored first here.** The exploration notebook MUST exist before the formal experiment directory is created. Put EDA, prototyping, and visualization work here. **Important:** Any processed or modified data generated during exploration MUST simply be held in memory or written to Colab's ephemeral `/content/` disk, NEVER directly to Bronze/Silver.
- When exploration work matures, graduate reusable code to `src/` and create a formal experiment via `create_experiment.py`.
