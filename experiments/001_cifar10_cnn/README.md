# 001 CIFAR-10 CNN

## Goal
Learn the fundamentals of deep learning by training a CNN on CIFAR-10 with MLflow tracking.

## Dataset
- **Source**: torchvision.datasets.CIFAR10 (auto-downloads)
- **Size**: 60,000 images (50k train, 10k test)
- **Resolution**: 32×32 RGB
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Model
| Architecture | Parameters | Expected Accuracy |
|--------------|------------|-------------------|
| SimpleCNN | ~500K | 75-82% |
| ResNet18 (pretrained) | ~11M | 85-92% |

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
| Metric | SimpleCNN | ResNet18 |
|--------|-----------|----------|
| Val Accuracy | | |
| Training Time | | |

## MLflow Run
- **Experiment**: cifar10-cnn
- **Run ID**: (auto-generated)

## What to Try Next
1. **Change architecture**: Edit `config.yaml` → `model.architecture: "resnet18"`
2. **Adjust learning rate**: Try 0.0001, 0.001, 0.01 and compare
3. **Increase epochs**: Train longer for higher accuracy
4. **Compare in MLflow**: Use the UI to visualize learning curves

## Notes
- **What worked**:
- **What didn't**:
- **Next steps**:
