## 2024-05-15 - [CRITICAL] Fix insecure deserialization in checkpoint loading
**Vulnerability:** The `load_checkpoint` function in `src/training/checkpoint.py` used `torch.load` without `weights_only=True`.
**Learning:** This is a critical security vulnerability because `torch.load` can execute arbitrary code during deserialization if an attacker provides a malicious pickle file.
**Prevention:** Always use `weights_only=True` when using `torch.load` to load model checkpoints, especially when loading files from untrusted sources.
