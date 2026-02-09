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
1. Write requirements (REQUIREMENTS.md)
2. Design architecture (docs/designs/*.drawio)
3. Write tests for data pipeline
4. Implement data pipeline → tests pass
5. Write tests for model shape/forward pass
6. Implement model → tests pass
7. Write integration test for training loop
8. Implement training → tests pass
9. Train on real data, evaluate
10. Document results
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
├── conftest.py          # Fixtures
└── pytest.ini
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
