"""
features/action_units.py

Geometric approximations of key Facial Action Units (FACS).

Computes a subset of AUs that are reliably estimable from 2-D landmark
geometry without a trained classifier.

AUs implemented:
  AU4  : Brow Lowerer      (inner brow descent)
  AU6  : Cheek Raiser      (cheek elevation, indirect)
  AU7  : Lid Tightener     (reduced eye aperture without blink)
  AU12 : Lip Corner Puller (smile)
  AU15 : Lip Corner Depress
  AU23 : Lip Tightener     (lip narrowing)
  AU24 : Lip Pressor       (lip compression)
  AU26 : Jaw Drop
  AU43 : Eye Closure       (low EAR)

Each AU is returned as a float [0, 1] — NOT a binary label.
0 = AU inactive, 1 = maximum observed activation.

These are GEOMETRIC PROXIES, not classifier outputs. They are useful as
mid-level features for downstream fusion but should not be reported as
ground-truth FACS labels.

INPUT:
  eye_feats   : dict from features/eye_features.extract_eye_features()
  mouth_feats : dict from features/mouth_features.extract_mouth_features()
  brow_feats  : dict from features/brow_features.extract_brow_features()
  landmarks   : np.ndarray (num_faces, num_points, 2|3) for AU6 cheek
  face_idx    : int
  image_wh    : (w, h) optional

OUTPUT:
  dict of AU intensities [0, 1] + confidence.
"""

import numpy as np
from geometry.math_utils import euclidean, safe_div, clamp
from landmarks.indices import LEFT_CHEEK_CENTER, RIGHT_CHEEK_CENTER, LEFT_EYE_TOP_REF


def estimate_action_units(
    eye_feats:   dict,
    mouth_feats: dict,
    brow_feats:  dict,
    landmarks:   np.ndarray | None = None,
    face_idx:    int = 0,
    image_wh:    tuple[int, int] | None = None,
) -> dict:
    """
    Estimate action unit intensities from per-frame feature dicts.

    Args:
        eye_feats   : output of extract_eye_features()
        mouth_feats : output of extract_mouth_features()
        brow_feats  : output of extract_brow_features()
        landmarks   : raw landmark array (needed for AU6 cheek)
        face_idx    : face index
        image_wh    : pixel dims (optional)

    Returns:
        {
          "AU4"  : float   # Brow Lowerer
          "AU6"  : float   # Cheek Raiser
          "AU7"  : float   # Lid Tightener
          "AU12" : float   # Lip Corner Puller (smile)
          "AU15" : float   # Lip Corner Depressor
          "AU23" : float   # Lip Tightener
          "AU24" : float   # Lip Pressor
          "AU26" : float   # Jaw Drop
          "AU43" : float   # Eye Closure
          "facial_tension"      : float   # aggregate tension signal
          "expression_intensity": float   # overall expression strength
          "confidence"          : float
        }
    """
    # --- AU4: Brow Lowerer (inner brow descent) ---
    AU4 = clamp(brow_feats.get("brow_lower", 0.0) * 1.2, 0.0, 1.0)

    # --- AU6: Cheek Raiser ---
    # Approximated via cheek landmark elevation relative to eye.
    # Full AU6 requires 3D or trained classifier; this is a rough proxy.
    AU6 = _estimate_au6(landmarks, face_idx, image_wh)

    # --- AU7: Lid Tightener ---
    # Eyes narrow (reduced EAR) without being in a blink.
    avg_ear = eye_feats.get("avg_EAR", 0.0)
    blink   = eye_feats.get("blink_detected", False)
    # Typical relaxed EAR ~0.28–0.35; tightening → 0.20–0.26
    au7_raw = clamp((0.28 - avg_ear) / 0.12, 0.0, 1.0) if not blink else 0.0
    AU7     = au7_raw

    # --- AU12: Lip Corner Puller (smile) ---
    AU12 = mouth_feats.get("smile_intensity", 0.0)

    # --- AU15: Lip Corner Depressor ---
    # Corners pulled down → negative smile.
    # When smile_intensity is 0 and lip tension is high, corners may be depressed.
    # Proxy: lip_tension when there is no smile signal.
    smile = mouth_feats.get("smile_intensity", 0.0)
    tension = mouth_feats.get("lip_tension", 0.0)
    AU15 = clamp((1.0 - smile) * tension, 0.0, 1.0)

    # --- AU23: Lip Tightener ---
    AU23 = mouth_feats.get("lip_tension", 0.0)

    # --- AU24: Lip Pressor ---
    AU24 = mouth_feats.get("lip_compression", 0.0)

    # --- AU26: Jaw Drop ---
    AU26 = mouth_feats.get("jaw_drop", 0.0)

    # --- AU43: Eye Closure ---
    # Low EAR or blink frame
    ear_conf = eye_feats.get("ear_confidence", 0.0)
    if avg_ear > 0 or blink:
        au43_raw = clamp(1.0 - avg_ear / 0.30, 0.0, 1.0)  # 0.30 = open eye baseline
        AU43 = au43_raw if blink or avg_ear < 0.22 else 0.0
    else:
        AU43 = 0.0

    # --- Aggregate signals ---
    facial_tension = float(np.mean([AU4, AU7, AU23, AU24]))
    expression_intensity = float(np.mean([AU4, AU6, AU7, AU12, AU15, AU23, AU24, AU26]))

    return {
        "AU4"  : round(AU4,  3),
        "AU6"  : round(AU6,  3),
        "AU7"  : round(AU7,  3),
        "AU12" : round(AU12, 3),
        "AU15" : round(AU15, 3),
        "AU23" : round(AU23, 3),
        "AU24" : round(AU24, 3),
        "AU26" : round(AU26, 3),
        "AU43" : round(AU43, 3),
        "facial_tension"       : round(facial_tension,       3),
        "expression_intensity" : round(expression_intensity, 3),
        "confidence"           : min(
            eye_feats.get("ear_confidence", 0.0),
            mouth_feats.get("confidence", 0.0),
            brow_feats.get("confidence", 0.0),
        ),
    }


def _estimate_au6(
    landmarks: np.ndarray | None,
    face_idx:  int,
    image_wh:  tuple[int, int] | None,
) -> float:
    """
    AU6 cheek raiser: rough proxy via cheek landmark elevation.
    Returns 0.0 if landmarks unavailable.
    """
    if landmarks is None or face_idx >= landmarks.shape[0]:
        return 0.0

    face = landmarks[face_idx, :, :2]

    if image_wh:
        scale = np.array(image_wh, dtype=np.float32)
        f = face * scale
    else:
        f = face

    # Cheek center rises during AU6 (Duchenne marker).
    # Proxy: cheek_y relative to eye_top_y. Smaller difference = cheek raised.
    left_cheek = f[LEFT_CHEEK_CENTER]
    eye_top    = f[LEFT_EYE_TOP_REF]

    cheek_to_eye = left_cheek[1] - eye_top[1]   # positive = cheek below eye (normal)

    if image_wh:
        norm_ref = image_wh[1] * 0.15   # 15% of frame height
    else:
        norm_ref = 0.10

    # When AU6 fires, cheek elevates → cheek_to_eye shrinks.
    # At rest ≈ norm_ref; raised → approaching 0.
    au6 = clamp(1.0 - cheek_to_eye / norm_ref, 0.0, 1.0)
    return float(au6)
