"""
geometry/math_utils.py

Shared low-level math utilities used across all feature modules.

Rules:
  - Pure functions only — no state, no I/O.
  - All inputs/outputs are scalars or numpy arrays.
  - Every function has a fallback for degenerate inputs (zero vectors, etc.).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Scalar utilities
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp x into [lo, hi]."""
    return max(lo, min(hi, float(x)))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Division with zero guard."""
    return float(a) / float(b) if abs(b) > 1e-8 else default


def sigmoid(x: float, k: float = 1.0, center: float = 0.0) -> float:
    """Sigmoid squash → (0, 1). k controls steepness."""
    return 1.0 / (1.0 + np.exp(-k * (x - center)))


def linear_map(x: float, in_lo: float, in_hi: float,
               out_lo: float = 0.0, out_hi: float = 1.0) -> float:
    """
    Linearly remap x from [in_lo, in_hi] → [out_lo, out_hi].
    Clamps to [out_lo, out_hi].
    """
    if abs(in_hi - in_lo) < 1e-8:
        return out_lo
    t = (x - in_lo) / (in_hi - in_lo)
    return clamp(out_lo + t * (out_hi - out_lo), min(out_lo, out_hi), max(out_lo, out_hi))


# ---------------------------------------------------------------------------
# Vector utilities (2-D and 3-D numpy arrays)
# ---------------------------------------------------------------------------

def norm2(v: np.ndarray) -> float:
    """L2 norm of a vector."""
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray) -> np.ndarray:
    """Return unit vector, or zero vector if degenerate."""
    n = norm2(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


def euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """L2 distance between two points (any dimension)."""
    return float(np.linalg.norm(p1 - p2))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle in degrees between two vectors.
    Returns 0.0 for degenerate (zero-length) inputs.
    """
    u1 = normalize(v1)
    u2 = normalize(v2)
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Midpoint between two points."""
    return (p1 + p2) / 2.0


# ---------------------------------------------------------------------------
# Landmark selection helpers
# ---------------------------------------------------------------------------

def get_points(face: np.ndarray, indices: list[int]) -> np.ndarray:
    """
    Extract a subset of landmark points for one face.

    Args:
        face    : np.ndarray shape (num_points, 2|3)
        indices : list of landmark indices

    Returns:
        np.ndarray shape (len(indices), 2|3)
    """
    return face[indices]


def centroid(face: np.ndarray, indices: list[int]) -> np.ndarray:
    """Mean position of a set of landmarks."""
    return get_points(face, indices).mean(axis=0)


def bbox_from_indices(face: np.ndarray, indices: list[int]) -> tuple:
    """
    Bounding box of a set of landmarks.

    Returns:
        (x_min, y_min, x_max, y_max) in the same coordinate space as `face`.
    """
    pts = get_points(face, indices)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


# ---------------------------------------------------------------------------
# Rolling statistics (used in temporal modules)
# ---------------------------------------------------------------------------

def rolling_slope(values: np.ndarray) -> float:
    """
    Linear trend (slope) of a 1-D array using least squares.
    Positive → values trending up; negative → trending down.
    Returns 0.0 if fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float32)
    # slope via normal equations
    x_mean = x.mean()
    y_mean = values.mean()
    num    = float(((x - x_mean) * (values - y_mean)).sum())
    den    = float(((x - x_mean) ** 2).sum())
    return safe_div(num, den)


def zero_crossing_rate(values: np.ndarray, center: float = 0.0) -> float:
    """
    Fraction of consecutive pairs that cross `center`.
    Useful for detecting oscillations (nodding, speaking).
    """
    if len(values) < 2:
        return 0.0
    shifted = values - center
    crossings = np.sum(shifted[:-1] * shifted[1:] < 0)
    return float(crossings) / (len(values) - 1)


def peak_frequency_hz(values: np.ndarray, fps: float) -> float:
    """
    Dominant oscillation frequency in a signal via FFT.

    Args:
        values : 1-D signal
        fps    : sample rate (frames per second)

    Returns:
        Peak frequency in Hz (0.0 if signal is too short or flat).
    """
    n = len(values)
    if n < 8:
        return 0.0
    # Detrend: subtract mean
    y = values - values.mean()
    if np.abs(y).max() < 1e-6:
        return 0.0
    spectrum = np.abs(np.fft.rfft(y))
    freqs    = np.fft.rfftfreq(n, d=1.0 / fps)
    # Ignore DC (index 0)
    if len(spectrum) < 2:
        return 0.0
    peak_idx = int(np.argmax(spectrum[1:])) + 1
    return float(freqs[peak_idx])
