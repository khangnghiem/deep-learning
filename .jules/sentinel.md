## 2024-05-24 - Insecure Deserialization in Model Checkpoint Loading
**Vulnerability:** Found `torch.load` being used without `weights_only=True` when loading model checkpoints.
**Learning:** `torch.load` defaults to using `pickle`, which can execute arbitrary code during deserialization if an attacker provides a malicious checkpoint file.
**Prevention:** Always use `weights_only=True` with `torch.load` when loading standard model, optimizer, and scheduler state dictionaries, as they are securely composed of supported primitive types and tensors.