## 2024-05-18 - CLI Progress Bar Clutter
**Learning:** Dense progress bars from `tqdm` during nested or long-running training loops create terminal clutter, reducing readability and hiding important logs.
**Action:** Use `leave=False, unit="batch"` in `tqdm` loops (especially inner loops like training batches or epochs) to ensure the bar clears after completion and provides more contextual progress feedback.
