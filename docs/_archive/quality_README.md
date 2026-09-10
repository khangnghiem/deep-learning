# Quality & Testing Strategy

> Quality gates, test pyramid, and verification standards for deep learning models and data pipelines.

---

## 1. The Deep Learning Testing Pyramid

| Layer | Tooling | Focus | Frequency |
|---|---|---|---|
| **Unit (Shape & Math)** | `pytest tests/unit` | Model tensor shapes, loss ranges, label smoothing, transforms | Every commit / local save (<5s) |
| **Integration** | `pytest tests/integration` | 1-batch / 1-epoch training loop runs, early stopping logic | Pre-commit / PR gate (<15s) |
| **Data Lineage** | `pytest tests/unit/test_medallion_paths.py` | Medallion path resolution, SHA-256 digests, manifest integrity | Every path or catalog change |
| **Acceptance / Benchmark** | Colab Pro+ GPU runs | Epoch convergence, Dice/IoU thresholds, held-out test splits | Formal experiment runs |

---

## 2. Core Testing Principles for DL

1. **Shape-First Testing**: Every neural network architecture must have unit tests asserting output shapes across diverse batch sizes, channel counts, and spatial resolutions (e.g. grayscale 1-channel vs RGB 3-channel).
2. **Gradient Flow & Loss Sanity**:
   - Losses must test that loss $\ge 0$.
   - Perfect predictions must yield lower loss than perturbed predictions.
   - Assert gradients are non-zero across all trainable layers after `loss.backward()`.
3. **Never Run Long Training Locally**:
   - Local tests run with tiny synthetic tensors (`torch.randn(2, 3, 64, 64)`) or 1-batch iterations (`max_batches=1`).
   - Real training runs are executed headlessly on Colab Pro+ GPUs.

---

## 3. Running the Test Suite

```bash
# Run all fast tests
pytest tests -v

# Run unit tests only
pytest tests/unit -v

# Run integration training tests
pytest tests/integration -v

# Run with coverage report
pytest --cov=src tests/
```

For the complete step-by-step TDD workflow, refer to the [**PyTorch TDD Guide**](tdd_guide.md).
