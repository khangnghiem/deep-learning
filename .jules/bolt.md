## 2024-04-18 - PyTorch Optimizer Memory Footprint
**Learning:** PyTorch optimizers by default set gradients to zero (`0`), which leaves the memory footprint and causes unnecessary memory bandwidth usage.
**Action:** Always use `optimizer.zero_grad(set_to_none=True)` to free the memory instead of setting it to zero, reducing memory bandwidth and footprint.
