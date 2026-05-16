## 2024-05-24 - [CRITICAL] Prevent Insecure Deserialization via torch.load

**Vulnerability:** Found multiple instances of `torch.load` missing the `weights_only=True` parameter across `.py` and `.ipynb` files. This exposes the application to arbitrary code execution (RCE) via insecure deserialization of untrusted `.pt`/`.pth` checkpoint files containing malicious pickled payloads.
**Learning:** `torch.load` defaults to `weights_only=False` in older versions or implicitly, allowing unpickling of arbitrary Python objects. Since model weights can often come from untrusted sources (e.g., downloaded models), this is a critical security vulnerability.
**Prevention:** Always use `weights_only=True` in `torch.load()` calls. If loading custom classes is absolutely required, it should be done in a sandboxed or highly restricted environment, but for pure state dictionaries, `weights_only=True` is mandatory to enforce safe unpickling.
