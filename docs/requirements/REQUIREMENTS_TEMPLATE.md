# Experiment Requirements Template

> Copy this file to your experiment folder and fill in the details.

## Experiment Overview

**Experiment ID**: `NNN_name`  
**Date**: YYYY-MM-DD  
**Author**: Khang Nghiem  

## Problem Statement

[What problem is this experiment solving?]

## Objectives

1. Primary: 
2. Secondary: 

## Success Criteria

| Metric | Minimum Target | Stretch Goal |
|--------|----------------|--------------|
| Accuracy | | |
| AUC-ROC | | |
| F1-Score | | |
| Inference Time | | |

## Dataset

- **Name**: 
- **Source**: HuggingFace / Kaggle / Custom
- **Size**: 
- **Split**: Train % / Val % / Test %
- **Classes**: 

### Data Quality Checks
- [ ] No missing values handled
- [ ] Class distribution analyzed
- [ ] Data augmentation strategy defined

## Model Architecture

- **Type**: CNN / ResNet / ViT / MLP / UNet
- **Base Model**: (if transfer learning)
- **Input Shape**: 
- **Output Shape**: 
- **Parameters**: ~ M

### Architecture Diagram
See: `docs/designs/NNN_name_architecture.drawio`

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam / AdamW / SGD |
| Learning Rate | |
| Batch Size | |
| Epochs | |
| Scheduler | |
| Loss Function | |

## Constraints

- [ ] Memory: Max GPU memory
- [ ] Time: Max training time
- [ ] Interpretability: Required / Not required

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting | High | Dropout, augmentation |
| Class imbalance | Medium | FocalLoss, weighted sampling |

## Dependencies

- [ ] Data downloaded
- [ ] Model architecture tested
- [ ] Training pipeline verified

## Verification Plan

### Unit Tests
- [ ] Data transforms output correct shapes
- [ ] Model forward pass works
- [ ] Loss computes correctly

### Integration Tests
- [ ] One epoch trains successfully
- [ ] Checkpointing works
- [ ] MLflow logging works

### Evaluation
- [ ] Validate on held-out test set
- [ ] Compare to baseline
- [ ] Generate confusion matrix
