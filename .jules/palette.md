## 2026-05-09 - Clean up terminal output for training scripts
**Learning:** In CLI/terminal environments for deep learning scripts, progress bar clutter can be a significant UX issue. Modifying tqdm defaults can resolve this.
**Action:** Use `leave=False` and `unit="batch"` in `tqdm` to prevent multiple completed progress bars from piling up and to provide more contextual progress metrics.
