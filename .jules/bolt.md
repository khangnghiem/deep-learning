## 2026-05-30 - Optimize imbalanced sampler initialization
**Learning:** Initializing the imbalanced sampler by recalculating class frequencies and performing iterative unidiomatic list-to-tensor conversions (e.g., `torch.tensor([list])`) adds unnecessary I/O overhead and computational cost, especially for large datasets.
**Action:** When computing class weights for samplers, iterate over the dataset only once to collect labels and use precomputed labels. Replace list-to-tensor conversions with efficient advanced tensor indexing (e.g., `class_weights[labels]`) to map values.
