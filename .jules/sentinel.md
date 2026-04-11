
## 2026-04-11 - Insecure Deserialization in torch.load
**Vulnerability:** The default behavior of `torch.load` allows for arbitrary code execution if a malicious checkpoint file is loaded.
**Learning:** PyTorch models saved via `torch.save` use Python's pickle module by default. When deserialized using `torch.load` without restrictions, it can execute arbitrary Python code embedded in the pickled data. This was found in checkpoint loading utilities.
**Prevention:** Always use the `weights_only=True` parameter in `torch.load` when loading PyTorch model checkpoints or tensors, ensuring only secure structures are deserialized.
