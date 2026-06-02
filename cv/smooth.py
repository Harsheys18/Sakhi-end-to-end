"""
smooth.py

Exponential moving average (EMA) smoothing for landmark arrays.

Kept at project root for backward compatibility with face_mesh.py imports.
The original implementation is preserved exactly; docstrings added.

Usage:
    smoothed, prev = smooth_landmarks(curr_np, prev_smoothed, alpha=0.4)
"""

import numpy as np


def smooth_landmarks(
    curr:  np.ndarray,
    prev:  np.ndarray | None,
    alpha: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exponential moving average smoothing over landmark arrays.

    Args:
        curr  : Current-frame landmarks. Shape (num_faces, num_points, 2).
        prev  : Previous smoothed landmarks, or None on first frame.
        alpha : Blend weight for current frame. Higher = less smoothing.
                Typical range: 0.3 – 0.6.

    Returns:
        (smoothed, prev_copy)
          smoothed   : EMA-blended landmarks, same shape as curr.
          prev_copy  : Copy of smoothed, to be passed as `prev` next frame.
    """
    # First frame or face count changed — no history to blend
    if prev is None or prev.shape != curr.shape:
        return curr, curr.copy()

    smoothed = alpha * curr + (1 - alpha) * prev
    return smoothed, smoothed.copy()
