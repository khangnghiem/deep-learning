---
id: ADR-0002
status: accepted
date: 2026-09-07
decision-makers: [khangnghiem]
---

# 0002 — Standardize Google Drive on 3-Pillars (`data`, `models`, `ops`) and Centralized `archive/`

## Context

Google Drive had accumulated scattered assets:
- Duplicate MLflow directories and misplaced tuning runs.
- Orphaned checkpoint folders outside standard lifecycle paths.
- Loose files in Drive root (`REQUIREMENTS.md`, personal PDFs).
- Multiple fragmented archive folders (`ops_archive/`, `archive/`).

## Decision

Standardize Google Drive on **3 production pillars** and a **centralized archive**:

1. **Strict 7 Root Directories** (Zero loose files in root):
   - `data/`: Medallion Data Lake (Pillar 1).
   - `models/`: Model lifecycle artifacts (Pillar 2).
   - `ops/`: MLflow tracking & operational metrics (Pillar 3).
   - `repos/`: Cloned Git repositories for compute environments.
   - `courses/`: Coursework and references.
   - `Colab Notebooks/`: Colab default notebook directory.
   - `archive/`: Centralized single archive for inactive assets.
2. **Lifecycle-First `models/` Hierarchy (No Domain Subfolders)**:
   - `models/` is partitioned strictly by lifecycle phase, never by domain:
     - `checkpoints/{NNN}_{dataset}_{model}/`: 1:1 deterministic mapping with `experiments/`.
     - `pretrained/{model_family}/`: Reusable foundation weights (DINOv2, SAM2, ResNet) shared across domains.
     - `registry/{model_name}/v{version}/`: Versioned export graphs (TorchScript, ONNX, TensorRT).
   - Application domains (`medical`, `finance`) are tracked as MLflow tags, not directory paths.
3. **Centralized `archive/`**:
   - Inactive assets are organized into `academic/`, `docs/`, `ops/`, `personal_office/`, and retired data components.

## Consequences

- **Deterministic Paths**: `experiments/` runs map directly to `models/checkpoints/` without domain lookups.
- **Single Source of Truth**: Exactly one authoritative tracking database (`ops/mlflow/mlflow.db`).
- **Clean Root**: No orphan files or ambiguity around artifact locations.
- **Trade-off**: Requires discipline to never save loose files in Google Drive root.
