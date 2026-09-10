# Quality & Testing

> Testing strategy, TDD workflow, and quality gates for deep learning models and data pipelines.

---

## 1. The Deep Learning Testing Pyramid

| Layer | Tooling | Focus | Frequency |
|---|---|---|---|
| **Unit (Shape & Math)** | `pytest tests/unit` | Model tensor shapes, loss ranges, label smoothing, transforms | Every commit / local save (<5s) |
| **Integration** | `pytest tests/integration` | 1-batch / 1-epoch training loop runs, early stopping logic | Pre-commit / PR gate (<15s) |
| **Data Lineage** | `pytest tests/unit/test_medallion_paths.py` | Medallion path resolution, SHA-256 digests, manifest integrity | Every path or catalog change |
| **Acceptance / Benchmark** | Colab Pro+ GPU runs | Epoch convergence, Dice/IoU thresholds, held-out test splits | Formal experiment runs |

---

## 2. Core Testing Principles

1. **Shape-First Testing**: Every neural network architecture must have unit tests asserting output shapes across diverse batch sizes, channel counts, and spatial resolutions (e.g. grayscale 1-channel vs RGB 3-channel).
2. **Gradient Flow & Loss Sanity**:
   - Losses must test that loss $\ge 0$.
   - Perfect predictions must yield lower loss than perturbed predictions.
   - Assert gradients are non-zero across all trainable layers after `loss.backward()`.
3. **Never Run Long Training Locally**:
   - Local tests run with tiny synthetic tensors (`torch.randn(2, 3, 64, 64)`) or 1-batch iterations (`max_batches=1`).
   - Real training runs are executed headlessly on Colab Pro+ GPUs.

---

## 3. TDD Workflow for ML

### What to Test

| Component | Test Type | Example |
|---|---|---|
| Data pipeline | Unit | "Transform outputs correct shape" |
| Model architecture | Unit | "Forward pass produces correct output shape" |
| Loss functions | Unit | "Loss is non-negative" |
| Training loop | Integration | "One epoch runs without error" |
| Metrics | Unit | "Accuracy computes correctly" |
| End-to-end | Smoke | "Training on tiny dataset succeeds" |

### What NOT to Test
- Exact model weights
- Exact loss values (use ranges)
- Exact accuracy (use minimum thresholds)

### Workflow Steps

```
1. Specify hypothesis, data contract, and acceptance gates using docs/_template/M-design-template.md
2. Explore & prototype in notebooks/ (ephemeral data on Colab /content/)
3. Write unit tests for data transforms (tests/unit/test_transforms.py)
4. Implement data pipeline in src/data/ → tests pass
5. Write unit tests for model forward pass & loss (tests/unit/test_models.py, test_losses.py)
6. Graduate architecture into src/models/ → tests pass
7. Write integration test for training loop (tests/integration/test_training.py)
8. Execute formal training on Google Colab Pro+ via experiments/
9. Evaluate on held-out test split, log to SQLite MLflow, save checkpoint to Drive
10. Document results in experiments/{NNN}_*/README.md
```

---

## 4. Running Tests

```bash
# All tests
pytest tests -v

# Unit tests only (fast)
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# With coverage
pytest --cov=src tests/
```

### Test Structure

```
tests/
├── unit/
│   ├── test_transforms.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_medallion_paths.py
│   └── test_metrics.py
├── integration/
│   ├── test_training.py
│   └── test_data_pipeline.py
└── conftest.py          # Fixtures (configured via pyproject.toml)
```

---

## 5. Pipeline Quality Gates

1. **Pre-Commit (Local):** Linting (`ruff`), type checking (`mypy`), and tensor shape tests (`pytest tests/unit/`).
2. **Pre-Training (Smoke Test):** 1-batch dry-run on CPU/GPU verifying loss computation and zero NaN values.
3. **Model Acceptance Gate:** Verification against held-out Gold test split. No model graduated to production without passing the minimum acceptance floor.
