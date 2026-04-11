## 2024-05-24 - CLI Clutter in Training Loops
**Learning:** Default tqdm progress bars in deep learning scripts leave completed bars on the screen, creating massive terminal clutter over many epochs, making it hard to find important metrics.
**Action:** Always add `leave=False` and `unit="batch"` to training/validation tqdm loops to keep the CLI clean and readable.
