## 2024-05-15 - Insecure Deserialization in Checkpoint Loading
**Vulnerability:** Arbitrary code execution via `torch.load()` due to loading untrusted checkpoints without `weights_only=True`.
**Learning:** `torch.load()` uses `pickle` by default, which can execute arbitrary code during deserialization.
**Prevention:** Always use `weights_only=True` when calling `torch.load()` to safely load checkpoints, restricting the unpickler to only basic types.
