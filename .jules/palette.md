## 2024-05-02 - CLI Progress Bar Clutter
**Learning:** Default tqdm progress bars in training loops (epoch over epoch) leave completed bars in the terminal, creating massive visual clutter and making it difficult to read critical log metrics over time.
**Action:** Always add `leave=False` and `unit="batch"` to `tqdm` instances in repetitive training/validation loops to keep the terminal output clean and focus on the current progress and log messages.
