## 2024-05-24 - [Insecure Deserialization in PyTorch Checkpoints]
**Vulnerability:** Found `torch.load()` being used without the `weights_only=True` parameter across checkpointing utilities and experiment scripts.
**Learning:** PyTorch defaults to using `pickle` for `torch.load()`, which allows arbitrary code execution via insecure deserialization if loading untrusted weights.
**Prevention:** Always append `weights_only=True` to `torch.load()` when loading model state dictionaries, optimizer states, or simple metric dictionaries to restrict deserialization to safe data types.
