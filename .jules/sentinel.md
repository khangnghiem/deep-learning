## 2025-02-27 - Insecure PyTorch Checkpoint Deserialization

**Vulnerability:** Arbitrary code execution via insecure deserialization when loading untrusted PyTorch checkpoints.
**Learning:** By default, `torch.load` relies on Python's `pickle` module, which is inherently unsafe and can execute arbitrary code encoded in the payload. The project was using `torch.load()` without constraints in the core checkpoint utility `src/training/checkpoint.py` and experiment scripts like `experiments/014_segformer_polyp/train.py`.
**Prevention:** Always append `weights_only=True` to `torch.load` calls when loading model weights, optimizer states, or simple metric dictionaries, as this restricts the unpickler to safe types (tensors, basic dicts, primitive types) and prevents arbitrary object reconstruction.
