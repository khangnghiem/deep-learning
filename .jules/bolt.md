## 2026-06-20 - Redundant array allocation in mask to polygon conversion
**Learning:** Found a specific bottleneck where `cv2` bounding box and contour area functions required repeatedly casting python lists into numpy float32 arrays in an inner loop.
**Action:** When running multi-step geometry or contour processing on polygons inside loops, always cast and cache the resultant numpy arrays once before passing them to multiple OpenCV/NumPy utility functions.
