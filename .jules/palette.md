## 2024-05-24 - Prevent Terminal Clutter in Training Loops
**Learning:** Using `tqdm` with default settings in training and validation loops causes significant terminal clutter, making it hard to read logs and track progress metrics clearly.
**Action:** Always add `leave=False` and `unit="batch"` to `tqdm` progress bars in iterative loops to clean up the CLI UX.
