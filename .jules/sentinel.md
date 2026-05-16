## 2024-05-16 - Insecure Deserialization via `torch.load`
**Vulnerability:** Found `torch.load()` instances in PyTorch model checkpoint loading utilities (`src/training/checkpoint.py`, `experiments/014_segformer_polyp/train.py`, and `experiments/014_segformer_polyp/014_segformer_polyp_e2e_test.ipynb`) lacking `weights_only=True`. This is susceptible to Arbitrary Code Execution via unpickling untrusted data.
**Learning:** PyTorch < 2.4 defaults `weights_only=False` in `torch.load`. Our checkpoint loading code must be explicit about the behavior since custom data could be used in explorations/experiments.
**Prevention:** Always enforce `weights_only=True` in `torch.load` during code reviews and testing, specifically for files matching model loaders like `.pt`, `.pth`, `.bin`.
