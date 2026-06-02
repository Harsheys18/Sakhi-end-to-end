"""
features/brow_features.py

Per-frame eyebrow feature extraction.

Computes:
  brow_raise         : brow elevation above eye lid (AU1/2 proxy)
  brow_lower         : inner brow descent toward eye (AU4 proxy)
  brow_tension       : horizontal compression / medial crowding
  brow_asymmetry     : left-right brow height difference
  inner_brow_angle   : slope of inner brow region (AU4 diagnostic)

These signals feed into:
  - stress estimation  (brow_lower + brow_tension)
  - interest/surprise  (brow_raise)
  - confusion          (brow_asymmetry + inner_brow_angle)

INPUT:
  landmarks : np.ndarray (num_faces, num_points, 2|3), normalized
  face_idx  : int
  image_wh  : (w, h) optional

OUTPUT:
  dict — see extract_brow_features() docstring.
"""

import numpy as np
from geometry.math_utils import euclidean, safe_div, clamp
from landmarks.indices import (
    LEFT_BROW_INNER, LEFT_BROW_PEAK,  LEFT_BROW_OUTER,  LEFT_BROW_FULL,
    RIGHT_BROW_INNER, RIGHT_BROW_PEAK, RIGHT_BROW_OUTER, RIGHT_BROW_FULL,
    LEFT_EYE_TOP_REF, RIGHT_EYE_TOP_REF,
    LEFT_EYE_INNER, RIGHT_EYE_INNER,
)


def extract_brow_features(
    landmarks: np.ndarray,
    face_idx:  int = 0,
    image_wh:  tuple[int, int] | None = None,
) -> dict:
    """
    Per-frame brow features.

    Args:
        landmarks : (num_faces, num_points, 2|3) normalized.
        face_idx  : face to process.
        image_wh  : pixel dimensions (improves distance scaling).

    Returns:
        {
          "brow_raise"        : float   # [0, 1]  0=default, 1=fully raised
          "brow_lower"        : float   # [0, 1]  0=default, 1=fully lowered (AU4)
          "brow_tension"      : float   # [0, 1]  furrowing / medial crowding
          "brow_asymmetry"    : float   # [0, 1]  L-R height mismatch
          "inner_brow_angle"  : float   # degrees, inner brow slope (+ = oblique in)
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

    # Reference: eye top points
    left_eye_top  = f[LEFT_EYE_TOP_REF]
    right_eye_top = f[RIGHT_EYE_TOP_REF]

    left_brow_peak  = f[LEFT_BROW_PEAK]
    right_brow_peak = f[RIGHT_BROW_PEAK]
    left_brow_inner  = f[LEFT_BROW_INNER]
    right_brow_inner = f[RIGHT_BROW_INNER]

    # --- Brow raise ---
    # In screen coords y increases downward.
    # brow_peak_y < eye_top_y → brow is ABOVE eye → positive lift.
    left_lift  = left_eye_top[1]  - left_brow_peak[1]   # positive = raised
    right_lift = right_eye_top[1] - right_brow_peak[1]
    avg_lift   = (left_lift + right_lift) / 2.0

    # Normalise: reference span = ~10-15% of face height.
    # In pixel space on 480p face: ~20-40px typical range.
    # Use inter-eye vertical as rough face scale.
    face_scale_ref = (left_eye_top[1] + right_eye_top[1]) / 2.0
    # Resting brow-eye gap ≈ face_height * 0.05 ≈ face_scale_ref * 0.1
    # Upper bound for raise ≈ face_scale_ref * 0.2
    # We use a simpler heuristic: normalise by a fraction of the eye y-coord.
    if image_wh:
        norm_ref = image_wh[1] * 0.08   # 8% of image height ≈ typical brow travel
    else:
        norm_ref = 0.04   # 4% of normalized height
    brow_raise = clamp(avg_lift / norm_ref, 0.0, 1.0)

    # --- Brow lower (AU4) ---
    # Brow descends toward the eye → left_lift becomes small or negative.
    brow_lower = clamp(1.0 - brow_raise, 0.0, 1.0)
    # Refine: AU4 specifically involves inner brow descent.
    inner_left_lift  = left_eye_top[1]  - left_brow_inner[1]
    inner_right_lift = right_eye_top[1] - right_brow_inner[1]
    inner_avg_lift   = (inner_left_lift + inner_right_lift) / 2.0
    # If inner brow is much lower than outer, AU4 is activated more than AU1/2
    brow_lower_au4 = clamp(1.0 - inner_avg_lift / max(norm_ref, 1e-4), 0.0, 1.0)
    brow_lower = (brow_lower + brow_lower_au4) / 2.0

    # --- Brow tension / furrowing ---
    # Medial crowding: horizontal distance between inner brow endpoints.
    # When brows are drawn together, this distance decreases.
    inner_gap = abs(right_brow_inner[0] - left_brow_inner[0])
    # Reference gap: ~distance between inner eye corners
    left_eye_inner_pt  = f[LEFT_EYE_INNER]
    right_eye_inner_pt = f[RIGHT_EYE_INNER]
    eye_inner_gap = abs(right_eye_inner_pt[0] - left_eye_inner_pt[0])
    # Tension: brow gap < eye gap = brows pulled together
    gap_ratio   = safe_div(inner_gap, eye_inner_gap)
    brow_tension = clamp(1.0 - gap_ratio, 0.0, 1.0)

    # --- Brow asymmetry ---
    # Difference in peak heights between left and right brow.
    asym = abs(left_lift - right_lift)
    brow_asymmetry = clamp(asym / max(norm_ref, 1e-4), 0.0, 1.0)

    # --- Inner brow angle ---
    # Slope of inner brow segment: outer→inner endpoint direction.
    # AU4 tilts the inner brow down and in, producing a distinctive angle.
    left_brow_outer_pt = f[LEFT_BROW_OUTER]
    left_dv = left_brow_inner - left_brow_outer_pt    # vector outer→inner
    angle_l  = float(np.degrees(np.arctan2(left_dv[1], left_dv[0])))

    right_brow_outer_pt = f[RIGHT_BROW_OUTER]
    right_dv = right_brow_inner - right_brow_outer_pt
    angle_r  = float(np.degrees(np.arctan2(right_dv[1], right_dv[0])))

    # Average inner brow angle (positive = downward-inward slope)
    inner_brow_angle = round((angle_l - angle_r) / 2.0, 2)

    return {
        "brow_raise"       : round(brow_raise,      3),
        "brow_lower"       : round(brow_lower,       3),
        "brow_tension"     : round(brow_tension,     3),
        "brow_asymmetry"   : round(brow_asymmetry,   3),
        "inner_brow_angle" : inner_brow_angle,
        "confidence"       : 1.0,
    }


def _empty_result() -> dict:
    return {
        "brow_raise"       : 0.0,
        "brow_lower"       : 0.0,
        "brow_tension"     : 0.0,
        "brow_asymmetry"   : 0.0,
        "inner_brow_angle" : 0.0,
        "confidence"       : 0.0,
    }
