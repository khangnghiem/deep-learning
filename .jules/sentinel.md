
## 2025-02-18 - Prevent Insecure Deserialization in PyTorch
**Vulnerability:** Insecure deserialization via `torch.load()` which could allow arbitrary code execution when loading untrusted model weights.
**Learning:** PyTorch's default `torch.load()` uses `pickle` which is not secure against maliciously crafted files.
**Prevention:** Always set `weights_only=True` when using `torch.load()` to limit the scope of unpickling to tensors, primitive types, and dictionaries only, as enforced in `src/training/checkpoint.py` and experiment scripts.
