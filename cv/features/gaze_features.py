"""
features/gaze_features.py

Per-frame gaze estimation from iris landmark position within eye boundaries.

The iris center (landmark 468 / 473) is measured relative to the eye corners
and lid midpoints to produce normalized gaze vectors.

Outputs:
  gaze_x         : horizontal gaze [-1 (full left), 0 (center), +1 (full right)]
  gaze_y         : vertical gaze   [-1 (up),        0 (center), +1 (down)]
  gaze_direction : categorical string ("center" | "left" | "right" | "up" | "down")
  eye_contact_score : 0–1, how centered both irises are (proxy for looking at camera)
  gaze_asymmetry : difference in gaze_x between left and right eye
                   (high = convergence/divergence anomaly or occlusion)

INPUT:
  landmarks : np.ndarray (num_faces, num_points, 2|3), normalized.
              Requires 478-pt model (iris landmarks 468–477).
  face_idx  : int

OUTPUT:
  dict — see extract_gaze_features() docstring.

Design: no state, no I/O.

Note on coordinate convention:
  MediaPipe normalized coords:  x increases rightward, y increases downward.
  gaze_x positive → iris displaced rightward → subject looking RIGHT.
  gaze_y positive → iris displaced downward  → subject looking DOWN.
"""

import numpy as np
from geometry.math_utils import safe_div, clamp
from landmarks.indices import (
    LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
    LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
    RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
)

# Gaze direction thresholds (fraction of full eye range)
_GAZE_THRESH_H = 0.20   # |gaze_x| > this → left or right
_GAZE_THRESH_V = 0.20   # |gaze_y| > this → up or down


def _iris_gaze(
    face:        np.ndarray,
    iris_idx:    int,
    outer_idx:   int,
    inner_idx:   int,
    top_idx:     int,
    bottom_idx:  int,
) -> tuple[float, float]:
    """
    Compute (gaze_x, gaze_y) for one eye.

    Iris position is normalized within the eye bounding span:
      0.0 = outer corner  1.0 = inner corner   →  remap to [-1, +1]
    Vertical:
      0.0 = top lid       1.0 = bottom lid      →  remap to [-1, +1]

    Returns gaze_x, gaze_y both in [-1, 1].
    """
    iris   = face[iris_idx,   :2]
    outer  = face[outer_idx,  :2]
    inner  = face[inner_idx,  :2]
    top    = face[top_idx,    :2]
    bottom = face[bottom_idx, :2]

    # Horizontal span (outer → inner)
    h_span = inner[0] - outer[0]       # positive: inner is to the right of outer
    h_pos  = safe_div(iris[0] - outer[0], h_span)  # 0 = outer, 1 = inner

    # Remap [0,1] → [-1,+1] with a sign convention:
    # "inner" is toward the nose, which is the OPPOSITE of gaze direction.
    # If the left iris is shifted toward the inner (nose) corner (h_pos → 1),
    # the subject is looking RIGHT (toward their nose side for left eye).
    # We want gaze_x positive = looking right (subject's right).
    #
    # For left eye: inner is to the right of outer (larger x).
    #   h_pos=1 → iris at inner corner → gaze RIGHT → gaze_x = +1. ✓
    # For right eye: inner is to the left of outer (smaller x in image).
    #   h_pos=1 → iris at inner corner → gaze LEFT → gaze_x = -1.
    # We resolve this by negating for the right eye in the caller.
    gaze_x = clamp(h_pos * 2.0 - 1.0, -1.0, 1.0)

    # Vertical span (top → bottom); y increases downward
    v_span = bottom[1] - top[1]
    v_pos  = safe_div(iris[1] - top[1], v_span)   # 0 = top, 1 = bottom
    gaze_y = clamp(v_pos * 2.0 - 1.0, -1.0, 1.0)

    return gaze_x, gaze_y


def _gaze_direction_label(gaze_x: float, gaze_y: float) -> str:
    """Map (gaze_x, gaze_y) to a categorical direction string."""
    if abs(gaze_x) < _GAZE_THRESH_H and abs(gaze_y) < _GAZE_THRESH_V:
        return "center"
    if abs(gaze_x) >= abs(gaze_y):
        return "right" if gaze_x > 0 else "left"
    return "down" if gaze_y > 0 else "up"


def extract_gaze_features(
    landmarks: np.ndarray,
    face_idx:  int = 0,
) -> dict:
    """
    Estimate gaze direction from iris landmarks.

    Args:
        landmarks : (num_faces, num_points, 2|3) — requires 478-pt model.
        face_idx  : face to process.

    Returns:
        {
          "left_gaze_x"       : float   # [-1, 1]
          "left_gaze_y"       : float   # [-1, 1]
          "right_gaze_x"      : float   # [-1, 1]
          "right_gaze_y"      : float   # [-1, 1]
          "avg_gaze_x"        : float   # mean of both eyes (primary signal)
          "avg_gaze_y"        : float
          "gaze_direction"    : str     # "center"|"left"|"right"|"up"|"down"
          "eye_contact_score" : float   # [0, 1]; 1 = irises perfectly centered
          "gaze_asymmetry"    : float   # |left_gaze_x - right_gaze_x|
          "confidence"        : float
        }
    """
    if landmarks is None or face_idx >= landmarks.shape[0]:
        return _empty_result()

    face = landmarks[face_idx]
    n_pts = face.shape[0]

    # Verify iris landmarks exist (require 478-pt model)
    if n_pts <= max(LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
        return _empty_result(reason="iris landmarks not available (need 478-pt model)")

    # --- Left eye gaze ---
    lgx, lgy = _iris_gaze(
        face, LEFT_IRIS_CENTER,
        LEFT_EYE_OUTER, LEFT_EYE_INNER,
        LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
    )
    # Left eye: inner is to the right in image coords → gaze_x sign is correct

    # --- Right eye gaze ---
    rgx, rgy = _iris_gaze(
        face, RIGHT_IRIS_CENTER,
        RIGHT_EYE_OUTER, RIGHT_EYE_INNER,
        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
    )
    # Right eye: inner is to the LEFT in image coords (nose side).
    # Our formula gives h_pos=1 → inner → but that means gaze LEFT.
    # Negate gaze_x so sign convention is consistent with left eye.
    rgx = -rgx

    avg_x = (lgx + rgx) / 2.0
    avg_y = (lgy + rgy) / 2.0

    # --- Eye contact score ---
    # Both irises centered → score 1.0.
    # Distance from center in gaze space.
    dist_l = (lgx ** 2 + lgy ** 2) ** 0.5
    dist_r = (rgx ** 2 + rgy ** 2) ** 0.5
    avg_dist = (dist_l + dist_r) / 2.0
    # Max possible distance in [-1,1]^2 is sqrt(2) ≈ 1.414
    eye_contact = clamp(1.0 - avg_dist / 1.414, 0.0, 1.0)

    asymmetry = abs(lgx - rgx)

    return {
        "left_gaze_x"       : round(lgx, 3),
        "left_gaze_y"       : round(lgy, 3),
        "right_gaze_x"      : round(rgx, 3),
        "right_gaze_y"      : round(rgy, 3),
        "avg_gaze_x"        : round(avg_x, 3),
        "avg_gaze_y"        : round(avg_y, 3),
        "gaze_direction"    : _gaze_direction_label(avg_x, avg_y),
        "eye_contact_score" : round(eye_contact, 3),
        "gaze_asymmetry"    : round(asymmetry, 3),
        "confidence"        : 1.0,
    }


def _empty_result(reason: str = "no face") -> dict:
    return {
        "left_gaze_x"       : 0.0,
        "left_gaze_y"       : 0.0,
        "right_gaze_x"      : 0.0,
        "right_gaze_y"      : 0.0,
        "avg_gaze_x"        : 0.0,
        "avg_gaze_y"        : 0.0,
        "gaze_direction"    : "unknown",
        "eye_contact_score" : 0.0,
        "gaze_asymmetry"    : 0.0,
        "confidence"        : 0.0,
    }
