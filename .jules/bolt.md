## 2025-03-01 - Avoid Double Dataset Iteration in Sampler Creation
**Learning:** PyTorch dataloading utilities (like extracting labels and calculating class weights) can lead to O(N) repetitive bottlenecks due to expensive `__getitem__` operations (I/O, augmentation). Calculating weights and creating the sampler independently caused double iteration over the dataset.
**Action:** Extract required properties (like labels) in a single initial pass and optionally accept them in downstream functions to prevent redundant memory allocations and dataset iterations.
