## 2024-04-18 - Set to None in Optimizer
**Learning:** In PyTorch, calling `optimizer.zero_grad()` executes memory operations to set all gradient tensors to zero, which takes memory bandwidth. By using `optimizer.zero_grad(set_to_none=True)`, we avoid these operations, freeing memory and slightly speeding up training loops.
**Action:** Always use `set_to_none=True` in PyTorch training scripts to improve memory efficiency.
