## 2025-02-18 - Progress Bar UX Improvement in Training Scripts
**Learning:** In CLI applications, unconfigured progress bars inside loops (like training epochs) can severely clutter the terminal output over time.
**Action:** Always configure `tqdm` progress bars with `leave=False` and `unit="batch"` for repeated/nested operations like model training, validation, and evaluation loops to ensure a clean terminal experience that replaces itself rather than stacking indefinitely.
