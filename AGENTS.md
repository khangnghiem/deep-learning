# AGENTS.md

This file provides guidance when working with code in this repository.

## Common commands

### Environment setup

- Install dependencies: `pip install -e ".[test]"`
- All dependencies are defined in `pyproject.toml` (single source of truth)

### Running experiments

- CIFAR-10 baseline: `cd experiments/001_cifar10_cnn && python train.py`
- New experiments: use `python scripts/create_experiment.py <dataset>` or copy `experiments/_template/`.

### Tests

- Full suite: `pytest tests -v`
- Unit only: `pytest tests/unit/test_models.py -v`
- Integration: `pytest tests/integration/test_training.py -v`

### MLflow UI

- `mlflow ui --backend-store-uri sqlite:///<path>/mlflow.db --port 5000`

## High-level architecture

### Dependencies

- Data ingestion scripts live in **`scripts/`** (`batch_download.py`) and dataset definitions in **`src/config/catalog.py`**.

### Repository layout

- `src/data/` — Dataset loaders, transforms, augmentation. Imports paths from `src.config.paths`.
- `src/models/` — `SimpleCNN`, `MLP`, `get_pretrained_resnet`, `get_pretrained_vit`, medical models, U-Net.
- `src/training/` — `Trainer` class, `EarlyStopping`, checkpoint utilities. Uses `src.config.paths` for MLflow and model saving.
- `src/utils/` — Metrics, visualization.
- `experiments/` — **Reproducible training runs**. Each has `config.yaml`, `train.py` (source of truth), thin `train.ipynb` (Colab launcher), `README.md`.
- `notebooks/` — **Interactive notebooks** for EDA, prototyping, visualization, and architecture search. No MLflow logging or checkpoints.
- `scripts/` — `create_experiment.py` (generates new experiments), `batch_train.py` (batch execution), `batch_download.py` (ingestion), `audit_data_lake.py` (data lake health check).
- `tests/` — Unit tests for models/losses, integration tests for training loop.
- `deploy/` — Deployment configs (Modal.com SAM2 inference).
- `docs/` — Design docs, data source references.

## Agent usage notes

- Import paths/MLflow from `src.config.paths`, never hardcode.
- Import `DATASETS` from `src.config.catalog` when needed by `create_experiment.py`.
- **NEVER run training locally** — use Google Colab. Run fast unit/shape tests and 1-batch smoke tests locally with `pytest`.
- **Design Documentation Standards:**
  - Master documentation portal is located at [`docs/README.md`](docs/README.md).
  - Major research programs/capabilities use the **Epic Design Template** in [`docs/_template/E-design-template.md`](docs/_template/E-design-template.md).
  - Individual model architectures and experiments use the **Model & Experiment Design Template** in [`docs/_template/M-design-template.md`](docs/_template/M-design-template.md).
  - **Results & Post-Mortem Protocol:** Every completed, stopped, or superseded experiment (`M-XX`) or epic (`E-XX`) MUST have its `Results & Post-Mortem` section filled out. Record actual metrics vs. baselines/targets, key findings, disproven hypotheses/surprises, and concrete follow-up actions. Never leave an experiment or epic marked "Completed" with empty post-mortem tables.
- **Google Drive & Medallion Data Lake Rules:**
  - Architecture specs: [`docs/architecture/data_lake.md`](docs/architecture/data_lake.md) and [`docs/architecture/drive_storage.md`](docs/architecture/drive_storage.md).
  - Structure uses a 4-tier layout: `0_landing/`, `1_bronze/<modality>/`, `2_silver/<modality>/`, and `3_gold/<modality>/`.
  - `1_bronze/` is partitioned strictly by 7 pure modalities (`audio`, `multimodal`, `tabular`, `text`, `timeseries`, `video`, `vision`); domains (`medical`, etc.) are tracked via metadata (`SOURCE.yaml`, `catalog.py`).
  - `models/` uses a lifecycle-first layout without domain nesting: `checkpoints/{NNN}_{dataset}_{model}/` (1:1 with `experiments/`), `pretrained/{model_family}/`, and `registry/`.
  - All folder and file names MUST strictly use lowercase `snake_case`.
  - **Feature Store:** Precomputed representations (CLIP, DINOv2, SAM embeddings) reside in Silver (`2_silver/features/` or `.../embeddings/`) as compressed `.parquet` files.
  - **Zero Direct Streaming:** Never stream loose image files over Google Drive FUSE. Gold packages in Drive (`data/3_gold/<modality>/<dataset>/`) are compressed `.tar.gz` bundles unpacked to Colab's local `/content/` NVMe before training.
  - **Dataset Lineage & MLflow Tracking:** Every experiment tracks data lineage via `src.data.log_medallion_dataset()` with companion `.manifest.json` recording SHA-256 digest, group-aware leakage-free protocol, and sample counts.
  - Persistent checkpoints and MLflow logs reside in Google Drive (`models/checkpoints/` and `ops/mlflow/`).
- **Google Colab Pro+ Execution:**
  - Runbook: [`docs/runbooks/colab_training.md`](docs/runbooks/colab_training.md).
  - Launchers (`launch.ipynb` or `train.ipynb`) must remain thin wrappers that clone the repo, mount Drive, unpack data, run headless `python train.py`, and immediately call `from google.colab import runtime; runtime.unassign()` to release compute units.
- **Naming Convention:** All experiments, notebooks, and folders MUST strictly follow the format `{NNN}_{dataset}_{model}` (e.g., `001_cifar10_cnn.ipynb` matching `experiments/001_cifar10_cnn/`).
  - `{NNN}`: 3-digit zero-padded sequential number.
  - `{dataset}`: The dataset or generic domain used.
  - `{model}`: The primary architecture or algorithm.
- When creating new experiments, follow patterns in `experiments/001_cifar10_cnn` and `_template/`.
- **`experiments/` = `.py` scripts** (train.py is the source of truth). Notebooks here are thin Colab launchers only.
- **`notebooks/` = `.ipynb` notebooks** mapped 1:1 to `experiments/`. **Every experiment must be explored first here.** The exploration notebook MUST exist before the formal experiment directory is created. Put EDA, prototyping, and visualization work here. **Important:** Any processed or modified data generated during exploration MUST simply be held in memory or written to Colab's ephemeral `/content/` disk, NEVER directly to Bronze/Silver.
- When exploration work matures, graduate reusable code to `src/` and create a formal experiment via `create_experiment.py`.
- **Post-Training Closure & Artifact Promotion:** Once Colab training finishes, pull evaluation metrics from MLflow (`ops/mlflow/`), verify artifacts in Google Drive (`models/checkpoints/{NNN}_{dataset}_{model}/`), fill out the `Results & Post-Mortem` in the design document, and promote production-ready models to `models/registry/`.


