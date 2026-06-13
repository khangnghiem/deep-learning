## 2025-02-24 - Fix Insecure Deserialization in PyTorch Checkpoints
**Vulnerability:** `torch.load()` was called without the `weights_only=True` parameter in `src/training/checkpoint.py`. This is insecure because PyTorch uses the `pickle` module by default, which can execute arbitrary code when deserializing untrusted `.pt` files.
**Learning:** PyTorch state dicts (model and optimizer weights) are composed of simple types and tensors, so `weights_only=True` safely prevents code execution without breaking legitimate functionality.
**Prevention:** Always use `torch.load(..., weights_only=True)` when loading model checkpoints unless custom untrusted class definitions are explicitly required (which should be avoided).
