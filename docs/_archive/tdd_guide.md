# TDD in Machine Learning & Deep Learning

## Should You Use TDD for ML/DL?

**Short answer**: Yes, but adapted for ML workflows.

Traditional TDD doesn't fit ML perfectly because:
- Models are stochastic (non-deterministic outputs)
- "Correct" is often probabilistic (accuracy ranges, not exact values)
- Training is expensive/slow

## Adapted TDD for ML: "ML Test-Driven Development"

### What to Test

| Component | Test Type | Example |
|-----------|-----------|---------|
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

## TDD Workflow for ML

```
1. Specify hypothesis, data contract, and acceptance gates using `docs/_template/M-design-template.md` & `experiments/{NNN}_*/README.md`
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

## Test Structure

```
tests/
├── unit/
│   ├── test_transforms.py
│   ├── test_models.py
│   ├── test_losses.py
│   └── test_metrics.py
├── integration/
│   ├── test_training.py
│   └── test_data_pipeline.py
└── conftest.py          # Fixtures (configured via pyproject.toml)
```

## Running Tests

```bash
# All tests
pytest tests/

# Unit tests only (fast)
pytest tests/unit/

# With coverage
pytest --cov=src tests/
```
