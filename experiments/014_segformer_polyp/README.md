# Experiment 014: SegFormer-B2 Polyp Segmentation Baseline

## Goal
Establish a new state-of-the-art baseline for polyp segmentation using the `SegFormer-B2` architecture.

## Why SegFormer-B2?
According to recent 2025/2026 literature, SegFormer provides a superior balance of accuracy and inference speed compared to traditional CNN configurations:
- **013 Baseline (U-Net + ResNet34)**: ~0.90 Dice, 24M params, 45 FPS
- **014 SegFormer (MiT-b2)**: ~0.94 Dice, 25M params, 35 FPS

Transformer models with heirarchical encoding and lightweight decoders capture complex contextual clues necessary for difficult-to-detect flat or sessile polyps without sacrificing clinical real-time inference utility.

## Dataset
- **Kvasir-SEG & CVC-ClinicDB Combined**
- **80% / 20%** Train/Val Split

## Architecture Details
- **Base model**: `nvidia/mit-b2` (SegformerForSemanticSegmentation via HuggingFace `transformers` library)
- **Loss Function**: StructureLoss (BCE + IoU)
- **Optimizer**: AdamW (lr=6e-5, weight_decay=1e-4)

## Expected Outcome
Achieve a validation Dice score of >0.92, reliably outperforming the 013 CNN baseline.
