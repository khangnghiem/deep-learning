## $(date +%Y-%m-%d) - Enhance CLI UX with tqdm
**Learning:** Terminal output is the UI for ML engineers. Unmanaged progress bars (like default tqdm in PyTorch training loops) cause severe terminal clutter and ambiguity (it/s vs batches/s).
**Action:** Always set `leave=False` for inner epoch loops and explicitly set `unit="batch"` when iterating over dataloaders to keep the terminal clean and intuitive.
