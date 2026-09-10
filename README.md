# Deep Learning Repository

PyTorch-based deep learning experiments with MLflow tracking, trained on **Google Colab Pro+**.

## Colab Quickstart

```python
# Cell 1 — Mount Drive & clone repo
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/<you>/deep-learning.git /content/deep-learning
%cd /content/deep-learning

# Cell 2 — Install package (editable, all src.* imports just work)
%pip install -q -e .

# Cell 3 — Run experiment
%cd experiments/001_cifar10_cnn
%run train.py
```

## Repository Structure

```
deep-learning/
├── pyproject.toml      # PEP 621 package metadata & dependencies
├── src/                # Shared library code (models, transforms, training utils, config)
│   ├── config/         #   paths.py (env-aware Drive resolver), catalog.py (100+ datasets)
│   ├── data/           #   Loaders, transforms, augmentation, converters
│   ├── models/         #   SimpleCNN, MLP, ResNet, ViT, UNet, medical models
│   ├── training/       #   Trainer, EarlyStopping, checkpoints, schedulers, losses
│   └── utils/          #   Metrics, visualization
├── experiments/        # Reproducible training runs (train.py + config.yaml + _train.ipynb)
├── notebooks/          # Interactive notebooks (EDA, prototyping, visualization)
├── scripts/            # Utility scripts (create_experiment, batch_download, data ingestion)
├── tests/              # Unit & integration tests (pytest)
├── deploy/             # Deployment configs (Modal.com SAM2 inference)
└── docs/               # 4-Pillar documentation portal (architecture, ADRs, runbooks, quality)
```

## Documentation

Full architectural specifications, operational runbooks, and decision logs are centralized in [**`docs/`**](docs/README.md):

- 🏛️ [**Architecture**](docs/architecture/README.md): [Medallion Data Lake](docs/architecture/data_lake.md), [Google Drive 3-Pillar Storage](docs/architecture/drive_storage.md), [MLOps Platform](docs/architecture/mlops_platform.md).
- 📜 [**Decisions (ADRs)**](docs/decisions/README.md): [ADR-0001 Medallion Lake](docs/decisions/ADR-0001-medallion-data-lake-architecture.md), [ADR-0002 3-Pillar Drive Layout](docs/decisions/ADR-0002-google-drive-three-pillar-layout.md).
- 🛠️ [**Runbooks**](docs/runbooks/README.md): [Colab Pro+ Training](docs/runbooks/colab_training.md), [Dataset Ingestion](docs/runbooks/dataset_ingestion.md).
- 🛡️ [**Quality & Testing**](docs/quality.md): Testing pyramid, TDD workflow, and quality gates.

## Data Lake (Medallion Architecture)

All data lives on Google Drive under the project root (`data/`):

| Layer      | Path                  | Purpose                                           |
| ---------- | --------------------- | ------------------------------------------------- |
| Landing    | `0_landing/`          | Raw uploads, zips, untouched files                |
| Bronze     | `1_bronze/<category>/`| Extracted raw data by category (medical, vision)  |
| Silver     | `2_silver/<category>/`| Cleaned annotations, masks & Feature Store        |
| Gold       | `3_gold/<category>/`  | Analysis-ready datasets (.tar.gz bundles)         |

See [`src/config/paths.py`](src/config/paths.py) for all path constants and [`docs/architecture/data_lake.md`](docs/architecture/data_lake.md) for full data lake specifications.

## Development Workflow

### 1. Explore → `notebooks/{exp_name}.ipynb`

**Every experiment must be explored first.** The `notebooks/` folder maps 1:1 with `experiments/`.
Use your matched Jupyter notebook for EDA, data visualization, architecture search,
augmentation experiments, and quick prototyping. No MLflow logging, no checkpoints.
When code stabilizes, graduate reusable pieces to `src/`.

### 2. Formalize → `experiments/NNN_*/train.py`

Run `python scripts/create_experiment.py <dataset>` to scaffold from the template.
Implement `get_model()` and `get_dataloaders()`, tune `config.yaml`.
The `.py` script is the **source of truth** — config-driven, MLflow-tracked, git-diffable, and testable.
A thin `.ipynb` launcher handles Colab execution: mount Drive → `pip install -e .` → `%run train.py`.

### 3. Share → `src/`

Reusable models, transforms, and training utilities imported by both
`notebooks/` and `experiments/`. Tested via `pytest tests/`.

## MLflow Tracking

```bash
# View experiment results (locally or on Colab)
mlflow ui --backend-store-uri sqlite:///path/to/My\ Drive/ops/mlflow/mlflow.db --port 5000
```

## Environment

- **Local**: `pip install -e ".[test]"` → `pytest tests -v` (dev/debug only — training runs go to Colab)
- **Colab**: Open `train.ipynb` → Run all → Close tab
