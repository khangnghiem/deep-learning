## 2024-05-23 - Insecure Deserialization in Model Checkpoint Loading
**Vulnerability:** Found `torch.load` calls without `weights_only=True` in `src/training/checkpoint.py` and `experiments/014_segformer_polyp/train.py`.
**Learning:** By default, `torch.load` uses Python's `pickle` module, which is vulnerable to arbitrary code execution if a maliciously crafted checkpoint file is loaded.
**Prevention:** Always use `weights_only=True` in `torch.load` to ensure only safe tensor data, primitive types, and dictionaries are unpickled.
