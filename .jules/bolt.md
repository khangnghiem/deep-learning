## 2024-06-27 - Dataloader Dataset Loop Optimization
**Learning:** Initializing class weights and dataset samplers natively requires two passes over the dataset. Since `__getitem__` is often an expensive operation involving file I/O or augmentation, this can cause a severe initialization bottleneck.
**Action:** Extract labels in a single dataset pass, and use advanced tensor indexing to efficiently construct itemized weights (e.g., `class_weights[labels]`) instead of a `for` loop mapping label to weight.
