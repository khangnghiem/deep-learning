## 2024-05-16 - Dataloader initialization bottleneck
**Learning:** PyTorch dataset `__getitem__` often involves expensive I/O operations. Performing multiple passes over the dataset during DataLoader initialization (e.g., to compute class frequencies, then to map weights for an imbalanced sampler) is a major performance bottleneck.
**Action:** Always compute necessary statistics and collect labels in a single pass. Use advanced tensor indexing (`weights[labels_tensor]`) to map weights instead of python loops with list-to-tensor conversions which trigger PyTorch warnings and are unidiomatic.
