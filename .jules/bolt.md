## 2024-05-24 - Single Pass Dataset Label Extraction
**Learning:** In `src/data/loaders.py`, `create_imbalanced_sampler` previously extracted labels by calling `get_class_weights` (which iterates the dataset), and then iterating the dataset again to build the sample weights. Dataset iteration (`__getitem__`) often involves expensive I/O operations (like loading and augmenting images).
**Action:** When initializing samplers, avoid multiple dataset passes. Instead, extract the labels in a single pass, then compute both class frequencies and per-sample weights from the extracted labels.
