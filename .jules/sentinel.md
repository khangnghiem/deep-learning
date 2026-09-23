## 2025-02-28 - PyTorch Insecure Deserialization
**Vulnerability:** `torch.load` used without `weights_only=True` in `src/training/checkpoint.py`, leading to potential insecure deserialization (arbitrary code execution via pickle) when loading untrusted model checkpoints.
**Learning:** By default, PyTorch's `torch.load` uses the standard Python `pickle` module, which is inherently insecure. This is a common and critical security risk in deep learning applications that share or load external model files.
**Prevention:** Always use `weights_only=True` when calling `torch.load` to load state dictionaries, restricting the unpickler to safe types like tensors and primitives.
