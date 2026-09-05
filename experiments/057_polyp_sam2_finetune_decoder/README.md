# 057 — Polyp SAM2 Fine-tune Decoder (BCE)

Fine-tuning the SAM2 mask decoder using PEFT LoRA (rank 16) with BCE loss on polyp segmentation dataset.

## Setup
- **Base Model**: SAM 2 Hiera Small (`sam2_hiera_small.pt`)
- **Prompt Detector**: YOLO (`025_polyp_yolov12x_aug`)
- **LoRA Target**: Mask Decoder (`r=16, alpha=32`)
- **Loss**: BCE
