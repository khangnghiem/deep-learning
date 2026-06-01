## 2024-04-18 - Tqdm Progress Bar Clutter
**Learning:** In CLI training loops, default tqdm progress bars print a new line for every epoch, leading to massive terminal clutter over many epochs.
**Action:** Always include `leave=False` and `unit="batch"` in training/validation loop tqdm calls to overwrite the same line and clarify progress.
