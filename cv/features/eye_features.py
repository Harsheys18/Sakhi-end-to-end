"""
features/eye_features.py

Per-frame eye feature extraction from MediaPipe FaceLandmarker landmarks.

Computes:
  - Left / right / average EAR  (Eye Aspect Ratio)
  - Per-frame blink detection    (boolean + frame-level confidence)

INPUT:
  landmarks : np.ndarray, shape (num_faces, num_points, 2), normalized [0, 1]
              x = col / width,  y = row / height
  face_idx  : int, which face to process (default 0)
  image_wh  : (width, height) tuple — REQUIRED for metric-accurate EAR.
              Without it, EAR is computed in normalized space, which
              introduces aspect-ratio distortion (~33% on 4:3 sensors).
              Blink detection still works; threshold must be recalibrated.

OUTPUT:
  dict — see `extract_eye_features` docstring.

Design rules:
  - No side effects. No printing. No drawing. No global state.
  - Pure numpy. O(1) per call.
  - image_wh optional: caller passes when available; module degrades gracefully.
"""

import numpy as np
from landmarks.indices import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX


# ---------------------------------------------------------------------------
# EAR threshold for blink detection.
#
# Pixel-space (image_wh provided):   typically 0.20 – 0.22
# Normalized space (no image_wh):    approximately 0.25 – 0.28 on 640×480
#   (vertical distances are compressed relative to horizontal in norm. space)
#
# These are empirical starting points; real deployments should calibrate
# per-user or use the temporal adapter's dynamic baseline.
# ---------------------------------------------------------------------------

EAR_BLINK_THRESHOLD_PIXEL = 0.21
EAR_BLINK_THRESHOLD_NORM  = 0.26


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """L2 distance between two 2-D points."""
    return float(np.linalg.norm(p1 - p2))


def compute_ear(
    face_landmarks: np.ndarray,
    ear_indices: list[int],
) -> float:
    """
    Compute Eye Aspect Ratio for one eye.

    EAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 * ||p1-p4|| )

    Args:
        face_landmarks : np.ndarray shape (num_points, 2)
                         Coordinates for a single face. Units depend on
                         whether caller has already scaled to pixels or not.
        ear_indices    : 6-element list [outer, up-outer, up-inner,
                                         inner, lo-inner, lo-outer]

    Returns:
        EAR as float. Returns 0.0 if denominator is near zero (eye fully
        closed or occluded landmark).
    """
    p1 = face_landmarks[ear_indices[0]]
    p2 = face_landmarks[ear_indices[1]]
    p3 = face_landmarks[ear_indices[2]]
    p4 = face_landmarks[ear_indices[3]]
    p5 = face_landmarks[ear_indices[4]]
    p6 = face_landmarks[ear_indices[5]]

    # Vertical openness (two chords across the eyelid gap)
    vert_a = _euclidean(p2, p6)
    vert_b = _euclidean(p3, p5)

    # Horizontal span (eye width)
    horiz = _euclidean(p1, p4)

    if horiz < 1e-6:
        # Degenerate — landmark collapsed or occluded
        return 0.0

    return (vert_a + vert_b) / (2.0 * horiz)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_eye_features(
    landmarks: np.ndarray,
    face_idx: int = 0,
    image_wh: tuple[int, int] | None = None,
) -> dict:
    """
    Extract per-frame eye features for one face.

    Args:
        landmarks : np.ndarray shape (num_faces, num_points, 2), normalized.
        face_idx  : Face index to process.
        image_wh  : (width, height) in pixels. When provided, landmarks are
                    scaled to pixel space before EAR computation, giving
                    aspect-ratio-correct distances.

    Returns:
        {
            "left_EAR"        : float   # EAR of left eye
            "right_EAR"       : float   # EAR of right eye
            "avg_EAR"         : float   # mean of left + right
            "blink_detected"  : bool    # True if avg_EAR < threshold
            "ear_confidence"  : float   # 1.0 if pixel-space, 0.75 if normalized
            "metric_space"    : str     # "pixel" | "normalized"
        }

    On failure (face_idx out of range, empty landmarks):
        Returns a zeroed dict with "ear_confidence": 0.0
    """
    # Guard: invalid face index
    if landmarks is None or face_idx >= landmarks.shape[0]:
        return _empty_result()

    face = landmarks[face_idx]  # shape: (num_points, 2)

    # Slice to xy only — landmarks may be (num_points, 2) or (num_points, 3)
    face_xy = face[:, :2]

    # --- Scale to pixel space if dimensions provided ---
    if image_wh is not None:
        w, h = image_wh
        scale = np.array([w, h], dtype=np.float32)
        face_scaled = face_xy * scale   # pixel coordinates
        threshold   = EAR_BLINK_THRESHOLD_PIXEL
        metric      = "pixel"
        confidence  = 1.0
    else:
        face_scaled = face_xy           # stay normalized
        threshold   = EAR_BLINK_THRESHOLD_NORM
        metric      = "normalized"
        confidence  = 0.75             # lower confidence: aspect-ratio not corrected

    # --- Compute EAR per eye ---
    left_ear  = compute_ear(face_scaled, LEFT_EYE_EAR_IDX)
    right_ear = compute_ear(face_scaled, RIGHT_EYE_EAR_IDX)
    avg_ear   = (left_ear + right_ear) / 2.0

    # --- Blink detection ---
    # A blink is declared when avg_EAR drops below threshold.
    # NOTE: this is a per-frame signal — temporal smoothing and blink
    # event counting happen in temporal/eye_temporal.py (BlinkTracker).
    blink_detected = avg_ear < threshold

    return {
        "left_EAR"       : round(left_ear,  4),
        "right_EAR"      : round(right_ear, 4),
        "avg_EAR"        : round(avg_ear,   4),
        "blink_detected" : blink_detected,
        "ear_confidence" : confidence,
        "metric_space"   : metric,
    }


def _empty_result() -> dict:
    """Return a zeroed feature dict (no face detected or invalid index)."""
    return {
        "left_EAR"       : 0.0,
        "right_EAR"      : 0.0,
        "avg_EAR"        : 0.0,
        "blink_detected" : False,
        "ear_confidence" : 0.0,
        "metric_space"   : "none",
    }
