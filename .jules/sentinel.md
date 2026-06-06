## 2024-05-18 - Prevent Insecure Deserialization in torch.load
**Vulnerability:** Using `torch.load()` without the `weights_only=True` parameter across `src/training/checkpoint.py`, `train.py` scripts, and Jupyter notebooks allows arbitrary code execution if an attacker supplies a maliciously crafted pickled model checkpoint.
**Learning:** PyTorch uses Python's `pickle` module by default, which is inherently insecure. The vulnerability was prevalent because it's a common default pattern in older PyTorch versions and tutorials, and wasn't strictly enforced in the repository's saving/loading utilities.
**Prevention:** Always set `weights_only=True` in all `torch.load` calls to strictly limit the unpickler to safe types (tensors, primitive types, dicts).
