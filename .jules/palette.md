## 2025-04-04 - Clearer progress bars
**Learning:** Terminal clutter during model training is a significant UX issue when `tqdm` progress bars are nested without proper configuration.
**Action:** Always use `leave=False` to clear nested progress bars and `unit="batch"` to provide clearer, more relevant units for model training outputs.
