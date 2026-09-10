# MLOps Platform & Distributed Runtime Architecture

> Authoritative specification for Colab Pro+ training execution, MLflow experiment tracking, hyperparameter tuning, and model deployment.

---

## 1. Google Colab Pro+ Execution Paradigm

### The Golden Rule: Local Tests, Cloud Training
- **Local Machine**: Strictly for development, editing, fast unit/shape tests (`pytest tests/unit`), and 1-batch smoke tests. **Never run full training epochs locally.**
- **Google Colab Pro+**: High-memory GPU runtime (A100, L4, V100) dedicated to training loops, validation sweeps, and feature extraction.

### Headless Launcher Pattern
To prevent lost compute units and ensure headless reproducibility, interactive Colab notebooks (`train.ipynb` / `launch.ipynb`) act strictly as thin wrappers:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Unpack Gold dataset from Drive to ephemeral NVMe
!mkdir -p /content/data
!tar -xzf "/content/drive/MyDrive/data/3_gold/vision/kvasir_seg_v1.tar.gz" -C /content/data/

# 3. Clone / Pull repository
!git clone https://github.com/khangnghiem/deep-learning.git /content/deep-learning
%cd /content/deep-learning
!pip install -e .

# 4. Run headless training script (source of truth)
!python experiments/011_polyp_segmentation/train.py

# 5. Automatically release GPU compute units upon completion
from google.colab import runtime
runtime.unassign()
```

---

## 2. MLflow Tracking & Lineage

### Architecture
- **Backend Store**: Persistent SQLite database in Google Drive: `sqlite:////content/drive/MyDrive/ops/mlflow/mlflow.db`.
- **Artifact Store**: `file:///content/drive/MyDrive/ops/mlflow/artifacts/`.

### Setup (Auto-Resolved via `src.config.paths`)
```python
from src.config.paths import MLFLOW_TRACKING_URI, setup_mlflow

# Sets tracking URI and experiment name automatically
setup_mlflow(experiment_name="011_polyp_segmentation")
```

### Dataset Lineage Logging Protocol
Every training run must record its exact Medallion data lineage at the start of the script:

```python
import mlflow
from src.data.mlflow_tracker import log_medallion_dataset

with mlflow.start_run(run_name="unet_resnet34_baseline"):
    # 1. Log data lake lineage
    log_medallion_dataset(
        dataset_name="kvasir_seg",
        category="vision",
        gold_dir=gold_path,
        source_bronze="1_bronze/vision/kvasir_seg",
        split_counts={"train": 800, "val": 100, "test": 100},
    )

    # 2. Log hyperparams & metrics
    mlflow.log_params({"lr": 1e-4, "batch_size": 16, "optimizer": "AdamW"})
    for epoch in range(epochs):
        train_loss = train_epoch(...)
        val_dice = validate(...)
        mlflow.log_metrics({"train_loss": train_loss, "val_dice": val_dice}, step=epoch)

    # 3. Log model artifacts
    mlflow.log_artifact(str(checkpoint_path), artifact_path="weights")
```

### What Every Run Tracks
1. **Parameters**: Batch size, learning rate, optimizer, scheduler, model backbone, image size.
2. **Metrics**: Train/val loss, IoU, Dice, precision, recall per epoch.
3. **Data Lineage**: SHA-256 hash of Gold training bundle, Bronze dataset path, split seeds via `src.data.log_medallion_dataset()`.
4. **Artifacts**: Best weights (`best.pt`), confusion matrices, validation segmentation grids.

### Launching Local MLflow UI
```bash
mlflow ui \
  --backend-store-uri "sqlite:///$HOME/Library/CloudStorage/GoogleDrive-khangnghiem@gmail.com/My Drive/ops/mlflow/mlflow.db" \
  --port 5000
```
Open `http://localhost:5000` in your browser.

---

## 3. Deployment & Ephemeral Observability

- **Unified Tracking**: All experiments, metrics, hyperparameter configs, and data lineage are centralized directly in MLflow (`ops/mlflow/mlflow.db`). Separate Drive databases for Optuna or TensorBoard are omitted to keep storage clean and single-sourced.
- **Ephemeral Profiling**: Real-time TensorBoard scalar logs or PyTorch profiler traces stay ephemeral on Colab's local `/content/` NVMe during active training sessions and are not synced to Google Drive.
- **Model Deployment**: Production-ready models are exported to TorchScript/ONNX in `models/registry/` and served serverless on Modal.com GPUs (`deploy/`).
