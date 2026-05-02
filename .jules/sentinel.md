## 2026-05-02 - Prevent Command Argument Injection in subprocess
**Vulnerability:** Argument injection risk via `subprocess.run` even when `shell=False`. Unsanitized user inputs could start with dashes (e.g. `-d`) injecting unintended command flags into `kaggle` CLI.
**Learning:** `shell=False` protects against shell metacharacters (`|`, `&`, `>`) but does *not* protect against malicious flags being interpreted by the target program itself.
**Prevention:** Always strictly validate and sanitize strings that are passed as arguments to subprocesses. For identifiers, enforce a strict regex pattern (e.g., `^[a-zA-Z0-9_][a-zA-Z0-9_-]*$`) preventing dashes at the start of any argument.
