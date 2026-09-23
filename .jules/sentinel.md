## 2026-09-05 - Insecure Deserialization in PyTorch Checkpoints
**Vulnerability:** Found `torch.load()` being used without `weights_only=True` to load model and optimizer states in `src/training/checkpoint.py` and `experiments/014_segformer_polyp/train.py`.
**Learning:** PyTorch's default `torch.load` uses Python's `pickle` module, which is vulnerable to arbitrary code execution if loading untrusted files. While these scripts load local files, setting `weights_only=True` enforces security best practices by restricting unpickling to standard model/optimizer tensors and primitives.
**Prevention:** Always append `weights_only=True` to `torch.load` calls when loading state dictionaries, as standard checkpoints only require basic data types.
