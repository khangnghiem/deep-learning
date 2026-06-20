## 2024-05-24 - [Insecure Deserialization in PyTorch Checkpoints]
**Vulnerability:** Found an insecure `torch.load()` call without `weights_only=True` in `src/training/checkpoint.py`.
**Learning:** PyTorch models heavily rely on standard python `pickle` under the hood. Default `torch.load()` behavior can lead to arbitrary code execution if malicious checkpoint files are loaded. Adding `weights_only=True` allows loading only basic primitives and PyTorch tensors, securing the loading mechanism.
**Prevention:** Always use `weights_only=True` when loading state dictionaries or models using `torch.load()` in PyTorch applications unless specifically restricted and necessary.
