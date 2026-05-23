## 2024-05-18 - Caching NumPy Array Conversions
**Learning:** In loops processing image geometry or annotations (e.g., in converters), failing to cache the result of NumPy array conversions and type casts when the array is required by multiple function calls (like `cv2.contourArea` and `cv2.boundingRect`) leads to redundant memory allocations and significant performance bottlenecks.
**Action:** Always cache the result of array conversions in a local variable if it will be used multiple times in the same loop to prevent redundant memory allocations.
