## 2024-05-18 - Argument Injection in Kaggle Helper
**Vulnerability:** Unsanitized user input passed to `subprocess.run` as part of command arguments in `src/data/kaggle.py`, allowing argument injection.
**Learning:** Even when `shell=False` is used, unsanitized inputs starting with dashes or containing malicious characters can inject unintended flags or arguments.
**Prevention:** Use strict regex validation (`^[a-zA-Z0-9_][a-zA-Z0-9_-]*$`) to sanitize identifiers before passing them to subprocess commands.
