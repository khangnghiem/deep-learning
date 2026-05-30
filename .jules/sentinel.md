
## 2024-05-24 - [CRITICAL] Prevent Insecure Deserialization via torch.load
**Vulnerability:** Unsafe deserialization using `torch.load()` without `weights_only=True` can lead to arbitrary code execution if loading an untrusted model file.
**Learning:** Model checkpoints and weights in this project were being loaded without the explicit `weights_only=True` argument, creating a security risk.
**Prevention:** Always use `weights_only=True` in `torch.load()` calls (e.g. `torch.load(path, map_location=device, weights_only=True)`) when loading model, optimizer, and scheduler state dictionaries.
