## 2024-05-02 - Dataset Iteration Bottleneck
**Learning:** Multiple dataset passes during DataLoader or Sampler initialization (e.g., counting frequencies first, then building weights in a second pass) causes massive performance degradation because `__getitem__` often involves expensive disk I/O and heavy transformations.
**Action:** Always compute necessary statistics and construct data structures in a single pass over the dataset to minimize redundant expensive calls.
