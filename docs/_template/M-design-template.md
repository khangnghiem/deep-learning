<!--
  Model & Experiment Low-Level Design (LLD) Template for Deep Learning.
  Filename: docs/epics/E-<epic-id>-<slug>/M-<nn>-<experiment-slug>.md
-->

# Model & Experiment M-XX: <Experiment Title>

> **One-line summary:** <Concise description of the hypothesis, model architecture, and target task (e.g., Fine-tuning SAM2 Vision Encoder with LoRA rank 32 for zero-shot polyp segmentation generalization).>

---

## 1. Experiment Metadata & Tracing

| Field | Value |
| :--- | :--- |
| **Experiment ID** | `EXP-{NNN}` (Mapped to `experiments/{NNN}_{dataset}_{model}/`) |
| **Parent Epic** | [Epic E-XX](../README.md) |
| **Status** | Draft / In Progress / Completed / Superseded |
| **Author / Engineer** | <Name or Agent ID> |
| **Target Hardware** | Google Colab Pro+ (NVIDIA L4 24GB or A100 40GB) |
| **MLflow Experiment** | `<epic_slug> / <experiment_slug>` |

---

## 2. Hypothesis & Mechanistic Rationale

> **Hypothesis:**  
> If we **[introduce change X — e.g., Replace U-Net CNN backbone with SegFormer-B2 and apply Focal Loss ($\gamma=2$)]**,  
> then **[metric Y — e.g., Validation Mean Dice]** will improve from **[baseline value — e.g., 0.812]** to **[target value — e.g., $\ge 0.860$]**,  
> because **[mechanistic reason — e.g., the self-attention mechanism captures global multi-scale context of subtle polyps while focal loss penalizes hard background mucosa misclassifications]**.

---

## 3. Data Contract & Pipeline Specification

### Data Source & Partitioning
- **Dataset Path:** `GOLD / "<category>" / "<dataset_name>"` (e.g. `GOLD / "vision" / "kvasir_seg"`)
- **Archive on Drive:** `data/3_gold/<category>/<dataset_name>/<dataset_name>_v<X>.tar.gz`
- **Integrity Manifest:** `data/3_gold/<category>/<dataset_name>/<dataset_name>_v<X>.manifest.json` (SHA-256 hash, split seeds)
- **Feature Store Inputs (if applicable):** Precomputed representations from `2_silver/<category>/<dataset_name>/embeddings/*.parquet`
- **Splits & Isolation Protocol:**
  - **Train:** <N> images / volumes
  - **Validation:** <N> images / volumes
  - **Test (Held-out):** <N> images / volumes
  - **Leakage Prevention:** Group split strictly partitioned on `patient_id` / `center_id`. No overlapping cases between splits.

### Tensor Contract
| Property | Specification |
| :--- | :--- |
| **Input Shape** | `[BatchSize, 3, 512, 512]` (RGB) or `[BatchSize, 1, H, W, D]` (3D) |
| **Input Dtype & Range** | `torch.float32`, normalized: $[0.0, 1.0]$ or ImageNet standard (Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`) |
| **Target Shape & Dtype** | `[BatchSize, 1, 512, 512]`, `torch.float32` (Binary Mask) or `torch.int64` (Multi-class) |

### Augmentation Pipeline (`Albumentations`)
```python
train_transforms = A.Compose([
    A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.2)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

---

## 4. Model Architecture Specification

### Backbone & Modules
- **Architecture Family:** <e.g., Encoder-Decoder Segmentation / Object Detector / Transformer>
- **Encoder / Backbone:** `<e.g., mit-b2 (pretrained on ImageNet-1k) / resnet50 / sam2_hiera_large>`
- **Decoder / Head:** `<e.g., All-MLP Decoder / FPN Head / Mask Decoder>`
- **Transfer Learning Setup:**
  - **Frozen Layers:** `<e.g., Stages 1 & 2 of backbone>`
  - **Trainable Layers:** `<e.g., Stages 3 & 4 + Decoder head>`
  - **LoRA Config (if applicable):** Rank $r=32$, $\alpha=64$, Dropout $=0.05$, Target modules: `["q_proj", "v_proj"]`

### Parameter Budget
| Layer / Submodule | Parameters | Trainable? |
| :--- | :--- | :--- |
| Backbone / Encoder | ~<NN> M | Frozen / Trainable / LoRA |
| Neck / Attention | ~<NN> M | Trainable |
| Head / Decoder | ~<NN> M | Trainable |
| **Total** | **~<NN> M** | **~<NN> M (~XX%)** |

---

## 5. Optimization & Training Dynamics

### Loss Formulation
Composite loss function balances segmentation overlap and edge confidence:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{dice}} \mathcal{L}_{\text{Dice}} + \lambda_{\text{bce}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{focal}} \mathcal{L}_{\text{Focal}}$$
- Weights: $\lambda_{\text{dice}} = 1.0$, $\lambda_{\text{bce}} = 0.5$, $\lambda_{\text{focal}} = 0.5$ ($\gamma=2.0$, $\alpha=0.25$).

### Hyperparameters
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | `AdamW` | Decoupled weight decay provides superior generalization |
| **Base Learning Rate** | `1e-4` | Tuned for fine-tuning transformer backbones |
| **Weight Decay** | `1e-2` | L2 regularization to prevent overfitting on small clinical cohort |
| **LR Scheduler** | `CosineAnnealingLR` | Smooth decay to `1e-6` |
| **Warmup** | 5 epochs | Linear warmup stabilizes initial transformer gradients |
| **Batch Size** | 8 per GPU (effective 16) | Gradient accumulation steps = 2 |
| **Mixed Precision** | `torch.cuda.amp.autocast(dtype=torch.float16)` | $2\times$ throughput speedup and 40% VRAM reduction |
| **Gradient Clipping** | `max_norm = 1.0` | Prevents gradient explosion in attention layers |
| **Early Stopping** | Patience = 10 epochs on Val Dice | Halts training when validation stops improving |

---

## 6. Execution & Colab Pro+ Contract

> Follow the [Colab Training Runbook](../runbooks/colab_training.md) for NVMe extraction, checkpointing, and auto-unassign protocols.

- **Data I/O:** Gold `.tar.gz` archive copied to `/content/` NVMe before training (never stream over FUSE).
- **Training launch:** Headless `python train.py` (source of truth).
- **Checkpoints saved to:** `${DRIVE_ROOT}/models/checkpoints/{NNN}_{dataset}_{model}/`
- **Auto-unassign:** Session teardown calls `google.colab.runtime.unassign()` to free GPU units.

---

## 7. Evaluation & Acceptance Gates

### Metrics & Targets
| Metric | Baseline | Target Threshold | Minimum Acceptance Floor |
| :--- | :--- | :--- | :--- |
| **Validation Mean Dice** | 0.812 | $\ge 0.860$ | 0.840 |
| **Validation Mean IoU** | 0.745 | $\ge 0.800$ | 0.770 |
| **Recall / Sensitivity** | 0.830 | $\ge 0.890$ | 0.850 |
| **Inference Latency** | 42 ms | $\le 25\text{ ms}$ | $\le 33\text{ ms}$ (30 FPS) |

### Binary Acceptance Criteria
- [ ] **AC-1 (Convergence):** Training loss decreases steadily without gradient NaN/Inf spikes across all epochs.
- [ ] **AC-2 (Performance Gate):** Validation Mean Dice on the held-out Gold split strictly reaches $\ge 0.840$.
- [ ] **AC-3 (Generalization):** Performance gap between Train Dice and Val Dice does not exceed 0.08 (overfitting guard).
- [ ] **AC-4 (Artifact Persistence):** `best_model.pt`, `completed.json`, and MLflow metrics are successfully written to Google Drive.

---

## 8. Failure Modes & Mitigations

| Failure Mode | Root Cause | Detection Signal | Automated / Manual Mitigation |
| :--- | :--- | :--- | :--- |
| **CUDA Out of Memory (OOM)** | Batch size exceeds 24GB VRAM | `torch.cuda.OutOfMemoryError` | Reduce batch size from 8 to 4; increase `grad_accum_steps` from 2 to 4 |
| **Loss Divergence / NaN** | High learning rate or unclipped gradients | Loss outputs `nan` or `inf` | Clamp loss; enforce `torch.nn.utils.clip_grad_norm_`; reduce LR by $5\times$ |
| **Colab Timeout / Disconnect** | Session timeout after 12h | Execution stops midway | `Trainer` loads latest `checkpoint.pt` from Drive and resumes epoch $E$ |
| **Drive FUSE Throttling** | Direct small-file I/O over FUSE mount | GPU utilization drops < 25% | Ensure dataset is unpacked to `/content/` NVMe before DataLoader starts |

---

## 9. Deliverables & MLflow Tracking

 - **MLflow Run Name:** `EXP-{NNN}_{dataset}_{model}`
 - **Source Script:** [`experiments/{NNN}_{dataset}_{model}/train.py`](train.py)
 - **Config:** [`experiments/{NNN}_{dataset}_{model}/config.yaml`](config.yaml)
 - **Colab Launcher:** [`experiments/{NNN}_{dataset}_{model}/launch.ipynb`](launch.ipynb)
 - **Dataset Lineage Logging (`src.data.log_medallion_dataset`):**
   - `medallion.tier`: `3_gold`
   - `medallion.category`: `<category>`
   - `medallion.dataset`: `<dataset_name>`
   - `medallion.version`: `v<X>`
   - `medallion.sha256`: `<sha256_hash>`
   - Native MLflow input entity logged via `mlflow.data.MetaDataset`.
 - **Artifacts in Google Drive:**
   - Checkpoint: `models/checkpoints/{NNN}_{dataset}_{model}/best_model.pt`
   - Completion Card: `models/checkpoints/{NNN}_{dataset}_{model}/completed.json`
   - Export: `models/checkpoints/{NNN}_{dataset}_{model}/model.onnx`

---

## 10. Results & Post-Mortem

### Actual Metrics
| Metric | Baseline | Target | Actual | Δ vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Validation Mean Dice** | 0.812 | $\ge 0.860$ | **<fill>** | **<fill>** |
| **Validation Mean IoU** | 0.745 | $\ge 0.800$ | **<fill>** | **<fill>** |
| **Inference Latency** | 42 ms | $\le 25\text{ ms}$ | **<fill>** | **<fill>** |

### Key Findings
- <What worked? What didn't? What was surprising?>

### Follow-Up Actions
- [ ] <Next experiment to run based on findings>
- [ ] <Architecture changes to investigate>
- [ ] <Data quality issues discovered>
