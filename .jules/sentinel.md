## 2026-07-04 - [PyTorch Insecure Deserialization in Checkpoint Loading]
**Vulnerability:** Found `torch.load` being used without `weights_only=True` in `src/training/checkpoint.py`, allowing for potential arbitrary code execution via insecure pickle deserialization.
**Learning:** PyTorch uses Python`s `pickle` module by default for serialization, which is inherently insecure. Checkpoints in this repository load various state dictionaries (model, optimizer, scheduler).
**Prevention:** Always use `weights_only=True` when calling `torch.load()` on untrusted checkpoints or standard model/optimizer/scheduler state dictionaries to restrict deserialization to safe types.
