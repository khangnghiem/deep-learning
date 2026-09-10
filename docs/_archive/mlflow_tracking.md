# MLflow Experiment Tracking Runbook

> Step-by-step protocol for configuring MLflow tracking, logging dataset lineage, and inspecting metrics locally.

---

## 1. Architecture & URI Setup

All experiments track parameters, metrics, model artifacts, and data lineage to a single persistent SQLite database in Google Drive:

- **Database Location**: `ops/mlflow/mlflow.db`
- **Artifacts Location**: `ops/mlflow/artifacts/`
- **Tracking URI**: `sqlite:///<drive_root>/ops/mlflow/mlflow.db`

In code, the tracking URI is automatically resolved via `src.config.paths`:
```python
from src.config.paths import MLFLOW_TRACKING_URI, setup_mlflow

# Sets tracking URI and experiment name automatically
setup_mlflow(experiment_name="011_polyp_segmentation")
```

---

## 2. Dataset Lineage Tracking Protocol

Every training run must record its exact Medallion data lineage.
Call `log_medallion_dataset()` at the start of your training script:

```python
import mlflow
from src.data.mlflow_tracker import log_medallion_dataset

with mlflow.start_run(run_name="unet_resnet34_baseline"):
    # 1. Log data lake lineage
    log_medallion_dataset(
        dataset_name="kvasir_seg",
        category="medical",
        gold_dir=gold_path,
        source_bronze="1_bronze/medical/kvasir_seg",
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

---

## 3. Launching Local MLflow UI

To visualize training curves, compare experiments, and view model parameters locally on your Mac:

```bash
# Point MLflow UI to your mounted Google Drive SQLite DB:
mlflow ui \
  --backend-store-uri "sqlite:///$HOME/Library/CloudStorage/GoogleDrive-khangnghiem@gmail.com/My Drive/ops/mlflow/mlflow.db" \
  --port 5000
```

Open `http://localhost:5000` in your browser.
