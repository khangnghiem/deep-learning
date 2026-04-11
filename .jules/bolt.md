## 2024-05-19 - Use `set_to_none=True` in `optimizer.zero_grad()`
**Learning:** PyTorch optimizers by default clear gradients by setting them to a tensor of zeroes, rather than `None`. This increases the memory footprint slightly and requires an unnecessary iteration over the model parameters to zero out the memory.
**Action:** Replace `optimizer.zero_grad()` with `optimizer.zero_grad(set_to_none=True)` in all training scripts and Jupyter notebooks to slightly optimize memory bandwidth and allocation during training loops.
