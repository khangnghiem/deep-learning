## 2024-05-24 - PyTorch Memory Bandwidth Optimization
**Learning:** In PyTorch training loops, calling `optimizer.zero_grad()` performs a read-modify-write operation on all parameter tensors, increasing memory bandwidth usage and memory footprint unnecessarily.
**Action:** Always use `optimizer.zero_grad(set_to_none=True)` in PyTorch training loops across all `.py` and `.ipynb` files to minimize memory bandwidth and reduce overall footprint.
