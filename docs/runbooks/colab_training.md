# Colab Pro+ Training Execution Runbook

> Step-by-step protocol for training deep learning models on Google Colab Pro+ with zero FUSE latency and automatic compute unit release.

---

## 1. Prerequisites
- Active Google Colab Pro+ subscription.
- Google Drive mounted and authorized under the primary user account (`khangnghiem@gmail.com`).
- Repository cloned or pulled.

---

## 2. Headless Launcher Notebook Template

Create or open `experiments/{NNN}_{dataset}_{model}/train.ipynb`.
Every training launcher must execute the following sequential cells:

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Unpack Model-Ready Gold Dataset to Local NVMe
```bash
%%bash
# NEVER stream loose image files over Google Drive FUSE!
# Unpack compressed Gold bundle directly to Colab ephemeral NVMe SSD:
mkdir -p /content/data
tar -xzf "/content/drive/MyDrive/data/3_gold/vision/kvasir_seg_v1.tar.gz" -C /content/data/
```

### Cell 3: Setup Codebase & Dependencies
```bash
%%bash
# Pull latest repository code or clone if fresh session
if [ ! -d "/content/deep-learning" ]; then
  git clone https://github.com/khangnghiem/deep-learning.git /content/deep-learning
fi

cd /content/deep-learning
git pull origin main
pip install -q -e .
```

### Cell 4: Execute Headless Training Script
```bash
%%bash
cd /content/deep-learning
# train.py is the single source of truth for experiment execution
python experiments/011_polyp_segmentation/train.py \
  --data-dir /content/data/kvasir_seg \
  --epochs 100 \
  --batch-size 16
```

### Cell 5: Release GPU Compute Units Immediately
```python
# Terminate session immediately upon completion to avoid burning compute units:
from google.colab import runtime
runtime.unassign()
```

---

## 3. Troubleshooting & Failure Modes

| Issue | Cause | Fix |
|---|---|---|
| **DataLoader epoch time >5 mins** | Streaming loose files directly over `/content/drive/MyDrive/...` FUSE mount | Check that dataset was unpacked to local `/content/data/` NVMe. |
| **Out of Memory (OOM) on GPU** | Batch size too large for VRAM allocation | Decrease `batch-size` in `config.yaml` or enable mixed precision (`amp: true`). |
| **Colab disconnected overnight** | Notebook stopped due to inactivity without headless execution | Use headless CLI invocation (`python train.py`) and verify unassign cell is present. |
