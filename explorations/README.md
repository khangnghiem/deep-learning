# Explorations

Interactive Jupyter notebooks for research and development work.

## Purpose

This folder is your **sketchpad** — use it for:

- **EDA**: Data exploration, distribution analysis, sample visualization
- **Prototyping**: Quick model architecture experiments, loss function comparisons
- **Visualization**: Grad-CAM, attention maps, augmentation previews
- **Learning**: Tutorials, library demos, concept tests

## What Does NOT Belong Here

- Full training loops with MLflow logging → use `experiments/`
- Production model checkpoints → use `models/trained/`
- Reusable utility code → graduate to `src/`

## Strict 1-to-1 Mapping Rule

> [!IMPORTANT]
> **Every experiment must be explored first.** 
> The exploratory notebook must exist BEFORE its formal experiment folder is created. The filename must exactly match.

```
experiments/001_cifar10_cnn/  <--->  explorations/001_cifar10_cnn.ipynb
experiments/007_polyp_unet/   <--->  explorations/007_polyp_unet.ipynb
```

When you run `python scripts/create_experiment.py <dataset>`, the script will automatically generate the formal experiment folder and its paired `.ipynb` template here.

## Base Templates

Standalone exploration templates are kept here for quick reuse. They are prefixed with an underscore so they don't get confused with formal experiments:
- `_template.ipynb` (Base EDA/Prototyping)
- `_template_architecture_search.ipynb`
- `_template_data_augmentation.ipynb`
