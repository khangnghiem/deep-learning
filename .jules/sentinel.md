## 2024-06-13 - [CRITICAL] Fix insecure deserialization in torch.load
**Vulnerability:** Unsafe deserialization using `torch.load()` without `weights_only=True` can lead to arbitrary code execution if an untrusted pickle file is loaded.
**Learning:** Default behavior of `torch.load` relies on Python's `pickle` module, which is not secure against maliciously constructed data.
**Prevention:** Always explicitly pass `weights_only=True` when loading state dictionaries or tensors using `torch.load()`.
