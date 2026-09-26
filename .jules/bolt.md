## 2024-09-26 - Single-pass PyTorch Dataset Iteration

**Learning:** Iterating over PyTorch `Dataset` instances multiple times (e.g., to compute class frequencies and then sample weights) causes redundant, expensive `__getitem__` calls which include I/O and augmentations. Standard python iteration (`for item in dataset:`) also adds an extra out-of-bounds `__getitem__` call at the end to catch `StopIteration`/`IndexError`.

**Action:** When extracting properties like labels from a dataset (such as during sampler initialization), explicitly iterate using index (`for i in range(len(dataset)):`) and do it only once to calculate all required statistics.
