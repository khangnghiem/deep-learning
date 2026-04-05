
## Goal
Learn the fundamentals of deep learning by training a CNN on CIFAR-10 with MLflow tracking.

## Dataset
- **Source**: torchvision.datasets.CIFAR10 (auto-downloads)
- **Size**: 60,000 images (50k train, 10k test)
- **Resolution**: 32×32 RGB
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Class balance**: Perfectly balanced — 5,000 train / 1,000 test per class

## Model

| Architecture | Parameters | Expected Accuracy |
|--------------|------------|-------------------|
| SimpleCNN | ~323K | 75-85% |
| ResNet18 (pretrained) | ~11M | 85-93% |

## How to Run

```bash
# From this directory
python train.py

# View results in MLflow
cd "/path/to/My Drive/mlflow"
mlflow ui --backend-store-uri file://./mlruns --port 5000
# Open http://localhost:5000
```

## Results

| Metric | SimpleCNN (v1) | SimpleCNN (v2) | ResNet18 |
|--------|----------------|----------------|----------|
| Val Accuracy | 85.03% | (pending) | (pending) |
| Training Time | ~14 min (CPU) | (pending) | (pending) |
| Best Epoch | 20/20 | (pending) | (pending) |
| Optimizer | Adam (constant LR) | Adam + Cosine | (pending) |

### Run History

| Date | Architecture | Optimizer | Scheduler | Epochs | Best Val Acc | Notes |
|------|-------------|-----------|-----------|--------|-------------|-------|
| 2026-03 | SimpleCNN | Adam | None (bug) | 20 | 85.03% | Config said "cosine" but code ignored it. No seed. No early stopping. |
| (next) | SimpleCNN | Adam | Cosine | 20 | (pending) | Fixed: seed=42, cosine scheduler, early stopping, full checkpoints |

## MLflow Run
- **Experiment**: cifar10-cnn
- **Run ID**: (auto-generated)

## What to Try Next
1. **Run with fixes**: The v2 script now has cosine scheduler, early stopping, and full evaluation — expect +2-4% accuracy
2. **Try ResNet18**: Edit `config.yaml` → `model.architecture: "resnet18"` (pretrained ImageNet → fine-tune)
3. **Adjust learning rate**: Try 0.0001, 0.001, 0.01 and compare side-by-side in MLflow
4. **Try AdamW**: Edit `config.yaml` → `training.optimizer: "adamw"` (better generalization with weight decay)
5. **Compare in MLflow**: Use the UI to visualize learning curves and confusion matrices

## Notes
- **What worked**:
  - CIFAR-specific transforms (RandomCrop with padding=4, HorizontalFlip) are critical for good accuracy
  - BatchNorm + Dropout (0.25 in conv, 0.5 in FC) prevented early overfitting
  - SimpleCNN reaches competitive 85% with only 323K params
- **What didn't**:
  - v1 had cosine scheduler in config but code never used it — loss was still decreasing at epoch 20
  - No seed meant runs were unreproducible
  - Only saved state_dict — no crash recovery possible
- **What changed in v2**:
  - Added seed (42), cosine scheduler, early stopping, full checkpoints
  - Post-training evaluation: confusion matrix + per-class F1 + sample predictions
  - Richer MLflow logging: LR per epoch, epoch time, environment info
  - Completion marker for batch monitoring
