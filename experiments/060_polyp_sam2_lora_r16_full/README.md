# 060 — Polyp SAM2 LoRA (r=16, Full Model)

Fine-tuning full SAM2 model using PEFT LoRA (rank 16) with BCE + Dice loss.

## Setup
- **Base Model**: SAM 2 Hiera Small (`sam2_hiera_small.pt`)
- **Prompt Detector**: YOLO (`025_polyp_yolov12x_aug`)
- **LoRA Target**: Full Model (`r=16, alpha=32`)
- **Loss**: BCE + Dice Loss
