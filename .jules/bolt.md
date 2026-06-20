## 2024-06-20 - Prevent Redundant Numpy Allocation in Loop
**Learning:** `np.array(poly).reshape(-1, 2).astype(np.float32)` was being calculated twice per iteration within an inner loop for `contourArea` and `boundingRect`.
**Action:** Cache the result of NumPy array conversions and type casts in a local variable if the resulting array is required by multiple function calls to prevent redundant memory allocations.
