## 2023-09-19 - [Dataset Iteration Bottleneck]
**Learning:** Standard Python iteration over PyTorch Datasets (`for item in dataset:`) relies on catching a `StopIteration` or `IndexError` at the end of the sequence. In PyTorch, this triggers an extra out-of-bounds `__getitem__` call at `index == len(dataset)`, causing redundant, expensive I/O and augmentation operations.
**Action:** Always use explicit index iteration (`for i in range(len(dataset)): item = dataset[i]`) when extracting properties like labels from Datasets to prevent unnecessary out-of-bounds processing.
