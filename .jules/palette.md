## 2024-05-24 - CLI Progress Bar UX
**Learning:** Default tqdm progress bars persist in the terminal after completion, which causes severe clutter during multi-epoch or multi-batch training runs.
**Action:** Always add `leave=False` and `unit="batch"` to `tqdm` progress bars in training and validation loops to keep the terminal output clean and focus on epoch-level metrics.
