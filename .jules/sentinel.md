## 2025-02-14 - Fix insecure deserialization in `torch.load`
**Vulnerability:** Found `torch.load` loading untrusted checkpoints without restricting the types unpickled.
**Learning:** PyTorch 2.12.1 and older versions default to allowing `pickle` to unmarshall arbitrary objects, leading to potential RCE if the checkpoint file is compromised. `weights_only=True` prevents this.
**Prevention:** Always use `weights_only=True` with `torch.load` when loading model or optimizer state dicts, and only use `weights_only=False` if absolutely necessary and strictly loading from trusted sources.
