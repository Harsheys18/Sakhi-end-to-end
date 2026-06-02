"""
features/mouth_features.py

Per-frame mouth feature extraction.

Computes:
  MAR               : Mouth Aspect Ratio (vertical/horizontal opening)
  smile_intensity   : lip corner elevation relative to lip midline
  lip_tension       : lip thinning (compressed relative to resting width)
  lip_compression   : inner-lip height relative to outer span (AU23/24 proxy)
  jaw_drop          : normalized jaw opening (AU26 proxy)

Note:
  speaking_activity and yawning_probability are TEMPORAL features — they
  require multi-frame history and live in temporal/mouth_temporal.py.
  This module supplies the per-frame primitives they depend on.

INPUT:
  landmarks : np.ndarray (num_faces, num_points, 2|3), normalized
  face_idx  : int
  image_wh  : (w, h) optional — enables pixel-space distances

OUTPUT:
  dict — see extract_mouth_features() docstring.
"""

import numpy as np
from geometry.math_utils import euclidean, safe_div, clamp
from landmarks.indices import (
    MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER,
    MOUTH_TOP_INNER, MOUTH_BOTTOM_INNER,
    MOUTH_TOP_OUTER, MOUTH_BOTTOM_OUTER,
    MOUTH_MAR_IDX,
)


def extract_mouth_features(
    landmarks: np.ndarray,
    face_idx:  int = 0,
    image_wh:  tuple[int, int] | None = None,
) -> dict:
    """
    Per-frame mouth features.

    Args:
        landmarks : (num_faces, num_points, 2|3) normalized.
        face_idx  : face to process.
        image_wh  : pixel dimensions for metric distances (optional).

    Returns:
        {
          "MAR"               : float   # Mouth Aspect Ratio [0, ~1.5]
          "smile_intensity"   : float   # [0, 1] 0=neutral, 1=wide smile
          "lip_tension"       : float   # [0, 1] 0=relaxed, 1=compressed
          "lip_compression"   : float   # [0, 1] inner gap thinning (AU23/24)
          "jaw_drop"          : float   # [0, 1] (AU26)
          "mouth_open"        : bool    # MAR > threshold
          "confidence"        : float
        }
    """
    if landmarks is None or face_idx >= landmarks.shape[0]:
        return _empty_result()

    face = landmarks[face_idx, :, :2]   # xy only

    if image_wh is not None:
        w, h  = image_wh
        scale = np.array([w, h], dtype=np.float32)
        f     = face * scale
    else:
        f = face

    left_c  = f[MOUTH_LEFT_CORNER]
    right_c = f[MOUTH_RIGHT_CORNER]
    top_in  = f[MOUTH_TOP_INNER]
    bot_in  = f[MOUTH_BOTTOM_INNER]
    top_out = f[MOUTH_TOP_OUTER]
    bot_out = f[MOUTH_BOTTOM_OUTER]

    horiz    = euclidean(left_c,  right_c)   # mouth width
    vert_in  = euclidean(top_in,  bot_in)    # inner lip gap
    vert_out = euclidean(top_out, bot_out)   # outer lip span

    # --- MAR: inner vertical / horizontal ---
    mar = safe_div(vert_in, horiz)

    # --- Smile intensity ---
    # Lip corners elevated above the lip midline → smile.
    # In normalized coords y increases downward, so corner_y < midline_y = up = smile.
    mid_y      = (top_out[1] + bot_out[1]) / 2.0
    corner_avg = (left_c[1] + right_c[1]) / 2.0
    # corner above midline → negative (corner_y - mid_y < 0) → smile
    raw_smile  = mid_y - corner_avg   # positive = smile
    # Normalise: a smile of ~5% of face height is a full smile
    # In pixel space: ~10-20px on typical 480p; use face-relative scaling.
    # Rough normalisation: divide by half the mouth width as reference span.
    smile_ref      = horiz * 0.5 if horiz > 1e-4 else 1.0
    smile_intensity = clamp(raw_smile / smile_ref, 0.0, 1.0)

    # --- Lip tension ---
    # Measure how much the lip corners are pulled laterally inward.
    # Proxy: thinning of the outer vertical span relative to inner.
    # At rest vert_out ≈ vert_in; tension → vert_out decreases more than vert_in.
    # We use the ratio vert_in / (vert_out + ε) — higher → more tension.
    # Clamp to [0, 1] using typical range 0.5–2.0.
    tension_raw = safe_div(vert_in, vert_out + 1e-6)
    lip_tension = clamp((tension_raw - 0.5) / 1.5, 0.0, 1.0)

    # --- Lip compression (AU23 / AU24 proxy) ---
    # Inner lip gap relative to mouth width; fully closed & thinned = compressed.
    lip_compression = clamp(1.0 - mar / 0.6, 0.0, 1.0)  # 0.6 ≈ relaxed open MAR

    # --- Jaw drop (AU26) ---
    # Large mouth opening relative to face width reference.
    # Normalise by mouth width (a crude but scale-invariant reference).
    jaw_drop = clamp(mar / 1.0, 0.0, 1.0)   # MAR ~1.0 = very wide open

    mouth_open = mar > 0.35

    return {
        "MAR"             : round(mar,             4),
        "smile_intensity" : round(smile_intensity,  3),
        "lip_tension"     : round(lip_tension,      3),
        "lip_compression" : round(lip_compression,  3),
        "jaw_drop"        : round(jaw_drop,         3),
        "mouth_open"      : mouth_open,
        "confidence"      : 1.0,
    }


def _empty_result() -> dict:
    return {
        "MAR"             : 0.0,
        "smile_intensity" : 0.0,
        "lip_tension"     : 0.0,
        "lip_compression" : 0.0,
        "jaw_drop"        : 0.0,
        "mouth_open"      : False,
        "confidence"      : 0.0,
    }
