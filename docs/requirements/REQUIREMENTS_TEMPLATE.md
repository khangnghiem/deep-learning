# Experiment Requirements: {EXPERIMENT_ID}

> **Experiment ID**: `{EXPERIMENT_ID}`  
> **Date**: YYYY-MM-DD  
> **Author**: Khang Nghiem  
> **Status**: Planned | Running on Colab | Completed | Superseded  

---

## 1. Hypothesis & Problem Statement

### Problem Statement
[What specific problem or task is this experiment addressing?]

### Scientific Hypothesis
> *If we* **[change independent variable X]** (e.g. switch from cross-entropy to focal loss, or add LoRA r=16 to decoder),  
> *then* **[metric Y]** will improve from **[baseline value]** to **[target value]**,  
> *because* **[theoretical or architectural reason]**.

- **Reference Baseline**: `experiments/{BASELINE_EXP_ID}` (Baseline Metric: `0.XX`)
- **Independent Variable**: [The ONE factor being altered in this experiment]
- **Controlled Variables**: [All hyperparameters held constant to ensure fair attribution]

---

## 2. Success Criteria & Evaluation Targets

| Metric | Baseline Score | Minimum Acceptable | Stretch Goal | Evaluation Split |
| :--- | :--- | :--- | :--- | :--- |
| **Dice Score / Accuracy** | 0.00 | 0.00 | 0.00 | Test Set |
| **AUC-ROC / mIoU** | 0.00 | 0.00 | 0.00 | Test Set |
| **F1-Score / mAP** | 0.00 | 0.00 | 0.00 | Test Set |
| **Inference Latency** | — | < XX ms | < XX ms | Batch Size = 1 |

---

## 3. Dataset & Data Lake Lineage

- **Dataset Name**: `{DATASET_NAME}`
- **Medallion Tier**: `03_gold`
- **Drive Path**: `src.config.paths.GOLD / "{DATASET_NAME}"`
- **Splits**: Train: XX% | Validation: XX% | Test: XX% (Random Seed: 42)
- **Total Samples**: XX train, XX val, XX test
- **Classes**: {NUM_CLASSES}

### Data Quality Checklist
- [ ] Gold dataset verified on Google Drive
- [ ] Class balance / distribution inspected in exploratory notebook (`notebooks/`)
- [ ] Image dimensions, color channels, and normalization parameters verified
- [ ] Augmentation pipeline defined without introducing corrupt labels

---

## 4. Model Architecture & Parameters

- **Architecture Family**: CNN / ResNet / ViT / UNet / SAM 2
- **Base Pretrained Weights**: [e.g., ImageNet / Meta SAM 2 Hiera Small / None]
- **Input Tensor Shape**: `[B, C, H, W]` (e.g. `[B, 3, 512, 512]`)
- **Output Tensor Shape**: `[B, num_classes, H, W]`
- **Total Parameters**: ~ XX M
- **Trainable Parameters**: ~ XX M (XX % trainable)

---

## 5. Training Configuration & Hyperparameters

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | `{OPTIMIZER}` | AdamW with weight decay |
| **Learning Rate** | `{LR}` | Base learning rate |
| **Batch Size** | `{BATCH_SIZE}` | Optimal for Colab VRAM allocation |
| **Epochs** | `{EPOCHS}` | Target epoch count before early stopping |
| **Scheduler** | CosineAnnealingLR / StepLR | Learning rate schedule |
| **Loss Function** | [e.g., BCEWithLogitsLoss / DiceLoss] | Loss objective |
| **Mixed Precision** | `fp16` / `bf16` | PyTorch AMP enabled |

---

## 6. Colab Pro+ Hardware & Compute Profile

- **Target Accelerator**: NVIDIA T4 (16GB) / L4 (24GB) / A100 (40GB)
- **High-RAM Mode**: Enabled / Standard
- **Estimated Epoch Duration**: ~ X minutes / epoch
- **Max Training Wall-Clock**: < X hours (within Colab session timeout)
- **Local Ephemeral Disk Strategy**: Copy dataset to `/content/` if Drive FUSE latency occurs

---

## 7. Artifacts & Experiment Tracking

- **MLflow Database**: SQLite at `DRIVE_ROOT / "mlflow.db"`
- **MLflow Experiment**: `{MLFLOW_EXP_NAME}`
- **MLflow Run Name**: `{EXPERIMENT_ID}`
- **Checkpoint Path**: `DRIVE_ROOT / "models" / "trained" / "{EXPERIMENT_ID}" / "best.pt"`
- **Tracked Metrics**: `train_loss`, `val_loss`, `val_metric`, `learning_rate`

---

## 8. Verification & TDD Plan

### Pre-Colab Local Tests
- [ ] Transforms produce correct tensor shape and dtype (`pytest tests/unit/test_transforms.py`)
- [ ] Model forward pass produces expected output shape (`pytest tests/unit/test_models.py`)
- [ ] Loss function executes and returns finite scalar (`pytest tests/unit/test_losses.py`)
- [ ] Local integration dry-run passes (`pytest tests/integration/test_training.py`)

### Colab Execution & Evaluation
- [ ] Epoch 1 executes cleanly on GPU without CUDA OOM
- [ ] Validation loss decreases over initial epochs
- [ ] Early stopping checkpoints best model to Google Drive
- [ ] Final evaluation run against held-out test split
- [ ] Confusion matrix or segmentation mask visualizations logged to MLflow
