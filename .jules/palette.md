## 2024-05-18 - [Tqdm Progress Bar UX]
**Learning:** Terminal output from ML training loops often gets cluttered because `tqdm` progress bars default to printing a new line for each loop instead of replacing the old one, and they do not specify units. Adding `leave=False` to clear the bar after each epoch and `unit="batch"` to show batch units makes the terminal output much cleaner and more understandable.
**Action:** Always add `leave=False` and `unit="batch"` to `tqdm` in inner training/validation loops across the repository.
