## 2024-05-24 - Torch Load Vulnerability
**Vulnerability:** Insecure deserialization via `torch.load`
**Learning:** `torch.load` uses `pickle` which can execute arbitrary code upon loading a manipulated file.
**Prevention:** Always use `weights_only=True` when loading untrusted weights with `torch.load`.
