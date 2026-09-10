# 062 — Polyp SAM2 LoRA (r=64, Decoder)

Fine-tuning SAM2 mask decoder scaled to LoRA rank 64 with BCE + Dice loss.

## Setup
- **Base Model**: SAM 2 Hiera Small (`sam2_hiera_small.pt`)
- **Prompt Detector**: YOLO (`025_polyp_yolov12x_aug`)
- **LoRA Target**: Mask Decoder (`r=64, alpha=128`)
- **Loss**: BCE + Dice Loss
