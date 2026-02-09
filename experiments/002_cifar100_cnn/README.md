# 002_cifar100_cnn

## Goal
CIFAR-100 image classification using CNN. This is a harder version of CIFAR-10 with 100 fine-grained classes.

## Dataset
- **CIFAR-100**: 60,000 32x32 color images
- **100 classes** organized into 20 superclasses (e.g., aquatic mammals, flowers, vehicles)
- 500 training + 100 test images per class

## Model
- SimpleCNN or ResNet18 (pretrained)

## Run

**Local:**
```bash
cd experiments/002_cifar100_cnn
python train.py
```

**Colab:**
Open `train.ipynb` → Run all

## Results
| Model | Best Val Acc | Epochs |
|-------|--------------|--------|
| | | |

## Notes
- CIFAR-100 is much harder than CIFAR-10 (100 vs 10 classes)
- Expect ~40-50% accuracy with SimpleCNN, ~60-70% with ResNet18
