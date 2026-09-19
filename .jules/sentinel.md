## 2026-09-19 - Prevent Insecure Deserialization
**Vulnerability:** Found an unconstrained `torch.load` call in `src/training/checkpoint.py` which could lead to arbitrary code execution if a malicious checkpoint file is loaded.
**Learning:** By default, `torch.load` uses pickle, making it dangerous. The codebase needs strict constraints on what is deserialized.
**Prevention:** Always use `weights_only=True` when loading standard PyTorch state dictionaries to restrict deserialization to primitive types and tensors.
