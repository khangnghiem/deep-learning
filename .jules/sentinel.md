## 2024-06-27 - Command Injection Vulnerability in File Download
**Vulnerability:** Using `os.system` with a formatted string `f"wget ... -O {SAM2_CHECKPOINT}"` to download a model checkpoint creates a severe command injection vulnerability, particularly if the path variables are influenced by external inputs or file names.
**Learning:** `os.system` combined with string formatting for file paths or URLs is a dangerous pattern in Python that exposes the system to command injection.
**Prevention:** Always use robust, native Python modules like `urllib.request.urlretrieve` or `requests` for HTTP downloads instead of spawning shell commands with `wget` or `curl`. Ensure user inputs are never passed directly to shell execution functions.
