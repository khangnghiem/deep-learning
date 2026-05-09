## 2024-05-30 - Argument Injection in Subprocess
**Vulnerability:** Argument injection via Kaggle identifiers (`dataset`, `competition`) passed to `subprocess.run(["kaggle", ...])`.
**Learning:** Even when `shell=False` is used, user input starting with a hyphen (e.g., `-O`) can be misinterpreted as command-line arguments by the target binary, leading to unintended behavior or arbitrary file writes.
**Prevention:** Always validate identifiers (e.g. using regex `^[a-zA-Z0-9_][a-zA-Z0-9_-]*$`) to ensure they do not start with a hyphen before appending them to subprocess command lists.
