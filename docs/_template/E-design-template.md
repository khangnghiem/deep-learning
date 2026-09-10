<!--
  Epic High-Level Design (HLD) Template for Deep Learning & Computer Vision.
  Filename: docs/epics/E-<epic-id>-<slug>/README.md
-->

# Epic E-XX: <Research Capability / Program Title>

> **One-line summary:** <Concise description of the deep learning capability delivered (e.g., Real-time semantic segmentation and tracking of colorectal polyps from colonoscopy video).>

| Field | Value |
| :--- | :--- |
| **Status** | Draft / Active / Completed / Superseded |
| **Created** | YYYY-MM-DD |
| **Last Updated** | YYYY-MM-DD |
| **Primary Metric** | <e.g., Mean Dice, Top-1 Accuracy, FID, BLEU> |

---

## 1. Executive Summary & Problem Formulation

- **Problem Statement:** <What bottleneck, scientific question, or engineering problem does this capability address? What are current limitations of existing algorithmic or manual approaches?>
- **Target Value & Objectives:** <Quantifiable impact and goals — e.g., Increase detection rate by 15%, reduce diagnostic/processing latency under 25ms, or demonstrate generalization to out-of-distribution cohorts.>
- **Key Milestones:**
  1. Benchmark baseline open-source models on unified Gold dataset.
  2. Develop & fine-tune specialized domain architecture / adaptation strategy.
  3. Validate generalization across multi-center / out-of-distribution cohorts.
  4. Export production-ready artifacts (ONNX / TensorRT) for downstream deployment.

---

## 2. Scope & Constraints

- **In Scope:**
  - <Specific task: e.g., Semantic segmentation on multi-center benchmark datasets>
  - <Target metrics: e.g., Mean Dice $\ge 0.88$, mIoU $\ge 0.82$, Inference latency $\le 25\text{ms}$ on NVIDIA L4>
  - <Datasets: e.g., Kvasir-SEG, CVC-ClinicDB, BKAI-IGH, PolypGen>
- **Out of Scope:**
  - <e.g., Multi-modal captioning, end-to-end user application UI>
  - <e.g., Direct external database synchronization (handled in downstream service)>
- **Dependencies & Assumptions:**
  - Gold datasets pre-packaged as single tarballs in Google Drive (`data/3_gold/<modality>/<dataset>/`).
  - Google Colab Pro+ compute available with L4/A100 GPU tiers.
  - MLflow SQLite tracking hosted in persistent Google Drive (`ops/mlflow/mlflow.db`).

---

## 3. Benchmark Baselines & SOTA Targets

| Metric | Current SOTA / Baseline | Epic Target | Acceptance Floor | Evaluation Split |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Metric** | 0.812 (Baseline U-Net) | $\ge 0.880$ | 0.850 | Gold Test Cohort (Held-out) |
| **Secondary Metric** | 0.745 (mIoU) | $\ge 0.820$ | 0.780 | Gold Test Cohort |
| **Inference Latency** | 45 ms / frame (T4) | $\le 25\text{ ms}$ (L4) | $\le 33\text{ ms}$ (30 FPS) | Batch size = 1, FP16 |
| **VRAM Footprint** | 8.2 GB | $\le 6.0\text{ GB}$ | $\le 8.0\text{ GB}$ | Inference mode |

---

## 4. Data Strategy & Lineage

> See [Medallion Data Lake Architecture](../architecture/data_lake.md) for the full 4-tier spec, Feature Store, and lineage protocol.

- **Datasets used in this Epic:** <list specific datasets and their Medallion tier paths, e.g., `data/3_gold/vision/kvasir_seg/`>
- **Ingestion & Curation Plan:** <how new data transitions from `0_landing/` to `1_bronze/` to `2_silver/` and finally packaged as `.tar.gz` in `3_gold/`>
- **Feature Store Inputs (if applicable):** <e.g., precomputed representations from `data/2_silver/<modality>/<dataset>/embeddings/`>
- **Data Leakage & Split Isolation Protocol:** <group-aware splitting strategy, e.g., subject/patient/center-level isolation with zero train-val-test overlap>

---

## 5. System Architecture & Core Modules

> See [MLOps Platform](../architecture/mlops_platform.md) for the full system architecture diagram (Data → Compute → Tracking).

- **New Model Architectures:** <list new models added to `src/models/`, e.g., `PolySegNet`, `SAM2LoRAAdapter`>
- **New Dataset Classes / Loaders:** <list new loaders added to `src/data/`, e.g., `PolypSegDataset`>
- **Core Training / Loss Modules:** <list `src/` modules or custom losses this Epic introduces or modifies>
- **Scripts & Tools:** <list any new `scripts/data/prepare_*.py` or utility scripts>

---

## 6. Experimentation Plan & Ablation Roadmap

| Exp ID | Folder / Slug | LLD Document | Primary Hypothesis | Architecture / Variation | Target Metric | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **001** | `{NNN}_{dataset}_baseline` | [M-01](M-01-baseline.md) | Benchmark vanilla baseline architecture | Simple U-Net / ResNet34 backbone | Primary Metric $\ge 0.81$ | Completed |
| **002** | `{NNN}_{dataset}_aug` | [M-02](M-02-aug.md) | Domain augmentations reduce overfitting | Baseline + AutoAugment / ElasticDeform | Primary Metric $\ge 0.84$ | Active |
| **003** | `{NNN}_{dataset}_backbone` | [M-03](M-03-backbone.md) | Vision transformer backbone improves multi-scale feature extraction | SegFormer-B2 / EfficientNet-B4 | Primary Metric $\ge 0.86$ | Draft |
| **004** | `{NNN}_{dataset}_adaptation` | [M-04](M-04-adaptation.md) | Foundation model with low-rank adaptation surpasses SOTA | SAM2 / DINOv2 + LoRA rank 32 | Primary Metric $\ge 0.88$ | Draft |
| **005** | `{NNN}_{dataset}_quantized` | [M-05](M-05-quantized.md) | FP16/INT8 export retains $\ge 99\%$ accuracy with $2\times$ speedup | TensorRT / ONNX Runtime engine | Latency $\le 20\text{ ms}$ | Draft |

---

## 7. Quality Gates & Risk Management

### Pipeline Quality Gates
1. **Pre-Commit (Local):** Linting (`ruff`), type checking (`mypy`), and tensor shape tests (`pytest tests/unit/`).
2. **Pre-Training (Smoke Test):** 1-batch dry-run locally verifying loss computation, zero NaN values, and memory footprint.
3. **Model Acceptance Gate:** Verification against held-out Gold test split. No model is graduated to production/registry without passing the minimum acceptance floor.

### Risks & Failure Modes

| Risk / Failure Mode | Likelihood | Impact | Detection Signal | Mitigation Protocol |
| :--- | :--- | :--- | :--- | :--- |
| **Data Distribution Shift** | High | High | Out-of-distribution drop in test metric | Multi-center validation; cross-dataset evaluation protocol |
| **Class Imbalance / Small Targets** | High | Medium | High false-negative rate on tiny lesions | Composite loss (BCE + Focal + Dice); patch-based sampling |
| **Colab Session Preemption** | Medium | Medium | Incomplete training run | Checkpointing state dicts to Drive after every epoch; auto-resume |
| **Drive FUSE I/O Bottleneck** | High | High | Low GPU utilization (< 20%) | Package datasets into single tarball; extract to local `/content/` NVMe |

---

## 8. Compute & Hardware Budget

> See [Colab Training Runbook](../runbooks/colab_training.md) for NVMe extraction, checkpointing, and auto-unassign protocols.

- **Target GPUs:** NVIDIA L4 (24GB VRAM) / A100 (40GB VRAM) via Colab Pro+.
- **Estimated Compute Hours:** ~<NN> GPU hours across planned experiments.
- **Drive Storage Allocation:** ~<NN> GB for datasets, checkpoints, and MLflow artifacts.

---

## 9. Results & Post-Mortem

> To be completed when the Epic is concluded or superseded. Summarizes cumulative program outcomes against original targets.

### Final Benchmark vs. Baselines
| Metric | Baseline | Epic Target | Achieved SOTA | Status (Pass/Fail) | Best Experiment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Metric** | 0.812 | $\ge 0.880$ | **<fill>** | Pass / Fail | `EXP-004` |
| **Secondary Metric** | 0.745 | $\ge 0.820$ | **<fill>** | Pass / Fail | `EXP-004` |
| **Inference Latency** | 45 ms | $\le 25\text{ ms}$ | **<fill>** | Pass / Fail | `EXP-005` |

### Key Findings & Retrospective
- **What worked:** <Architectural decisions, loss functions, augmentations, or training strategies that yielded significant gains>
- **What didn't:** <Hypotheses that were disproven, dead ends, or techniques that degraded performance>
- **Surprises & Technical Discoveries:** <Unexpected behaviors, data nuances, or scaling observations>

### Graduation & Next Steps
- [ ] **Artifacts Graduated:** <e.g., Best checkpoint promoted to `models/registry/` or exported ONNX runtime engine>
- [ ] **Code Graduated to `src/`:** <e.g., Novel model architecture or custom loss added to repository core>
- [ ] **Follow-Up Inquiries / Next Epic:** <New research questions or subsequent Epic, e.g., E-02 on real-time video tracking>

---

## 10. References & Artifacts

- **Primary Literature:** <Key research papers, DOIs, arXiv links>
- **Public Datasets:** <Source repositories, licenses, download scripts in `scripts/data/`>
- **MLflow Tracking URI:** `sqlite:///${DRIVE_ROOT}/ops/mlflow/mlflow.db`
- **Model Checkpoints:** `${DRIVE_ROOT}/models/checkpoints/`