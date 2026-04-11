## 2024-04-11 - [Insecure Deserialization in torch.load]
**Vulnerability:** Code loaded model weights using `torch.load()` without the `weights_only=True` parameter, leaving it vulnerable to insecure deserialization if untrusted files are loaded.
**Learning:** PyTorch's default `torch.load` behavior is to unpickle the object, which can execute arbitrary code encoded in the file.
**Prevention:** Always use `weights_only=True` in `torch.load` to ensure only state dictionaries and valid tensor types are loaded, disabling potentially dangerous object instantiation.
