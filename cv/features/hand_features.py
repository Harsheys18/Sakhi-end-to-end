"""
features/hand_features.py

Full hand feature extraction using MediaPipe HandLandmarker
and PoseLandmarker (for crossed-arms and proximity).

Computes (per-frame):
  hand_near_face           : bool — wrist within face-proximity threshold
  hand_to_face_score       : [0,1] — proximity score
  crossed_arms_probability : [0,1] — estimated from pose wrist/shoulder positions
  self_touch_behavior      : [0,1] — hand-to-face proximity (proxy for self-touch)
  gesture_frequency        : float — gestures per minute (from temporal tracker)
  open_palm_left           : bool  — left hand open (no curl)
  open_palm_right          : bool  — right hand open
  num_hands_visible        : int

Gesture frequency is temporal and lives in temporal/hand_temporal.py.
This module provides per-frame gesture primitives.

INPUT:
  hand_lm_norm  : np.ndarray (num_hands, 21, 3) normalized, or None
  hand_lm_world : np.ndarray (num_hands, 21, 3) world coords, or None
  handedness    : list[str] — 'Left'|'Right' per hand
  pose_norm     : np.ndarray (num_people, 33, 3) normalized, or None
  face_feats    : dict from features/face_features

OUTPUT:
  dict — see extract_hand_features() docstring.
"""

import numpy as np
from geometry.math_utils import clamp, euclidean, safe_div

# Hand landmark indices (MediaPipe 21-point hand model)
_WRIST       = 0
_THUMB_TIP   = 4
_INDEX_TIP   = 8
_MIDDLE_TIP  = 12
_RING_TIP    = 16
_PINKY_TIP   = 20
_INDEX_MCP   = 5
_MIDDLE_MCP  = 9
_RING_MCP    = 13
_PINKY_MCP   = 17
_THUMB_CMC   = 1

# Pose indices
_POSE_NOSE          = 0
_POSE_LEFT_WRIST    = 15
_POSE_RIGHT_WRIST   = 16
_POSE_LEFT_SHOULDER = 11
_POSE_RIGHT_SHOULDER= 12

# Face-proximity threshold in normalized image space
_FACE_PROX_THRESH = 0.15


def extract_hand_features(
    hand_lm_norm:  np.ndarray | None = None,
    hand_lm_world: np.ndarray | None = None,
    handedness:    list[str] | None  = None,
    pose_norm:     np.ndarray | None = None,
    face_feats:    dict | None       = None,
) -> dict:
    """
    Extract per-frame hand features.

    Args:
        hand_lm_norm  : (num_hands, 21, 3) normalized image coords, or None.
        hand_lm_world : (num_hands, 21, 3) world coords, or None.
        handedness    : list of 'Left'|'Right' strings, or None.
        pose_norm     : (num_people, 33, 3) normalized pose, or None.
        face_feats    : dict from face_features (for face center reference).

    Returns:
        {
          "num_hands_visible"       : int
          "hand_near_face"          : bool
          "hand_to_face_score"      : float  # [0,1]
          "self_touch_behavior"     : float  # [0,1]
          "crossed_arms_probability": float  # [0,1]
          "open_palm_left"          : bool
          "open_palm_right"         : bool
          "gesture_primitive"       : str    # "open"|"fist"|"point"|"pinch"|"none"
          "hand_model_active"       : bool
          "pose_model_active"       : bool
        }
    """
    hand_active = hand_lm_norm is not None and hand_lm_norm.shape[0] > 0
    pose_active = pose_norm    is not None and pose_norm.shape[0]    > 0
    ff          = face_feats or {}

    num_hands       = hand_lm_norm.shape[0] if hand_active else 0
    handedness      = handedness or (["Unknown"] * num_hands)

    # ── Hand-to-face proximity ────────────────────────────────────────────────
    face_cx, face_cy = ff.get("face_center", [0.5, 0.5])
    face_pt          = np.array([face_cx, face_cy], dtype=np.float32)

    hand_near_face   = False
    hand_face_score  = 0.0

    if hand_active:
        min_dist = float("inf")
        for hi in range(num_hands):
            wrist = hand_lm_norm[hi, _WRIST, :2]
            dist  = euclidean(wrist, face_pt)
            if dist < min_dist:
                min_dist = dist
        hand_face_score = clamp(1.0 - min_dist / _FACE_PROX_THRESH, 0.0, 1.0)
        hand_near_face  = min_dist < _FACE_PROX_THRESH

    elif pose_active:
        # Fallback: use pose wrist positions
        p = pose_norm[0]
        nose_pt = p[_POSE_NOSE, :2]
        dist_l  = euclidean(p[_POSE_LEFT_WRIST,  :2], nose_pt)
        dist_r  = euclidean(p[_POSE_RIGHT_WRIST, :2], nose_pt)
        min_dist = min(dist_l, dist_r)
        hand_face_score = clamp(1.0 - min_dist / _FACE_PROX_THRESH, 0.0, 1.0)
        hand_near_face  = min_dist < _FACE_PROX_THRESH

    # ── Crossed arms ─────────────────────────────────────────────────────────
    crossed_prob = _crossed_arms(pose_norm) if pose_active else 0.0

    # ── Per-hand: open palm, gesture primitive ────────────────────────────────
    open_left  = False
    open_right = False
    gesture    = "none"

    if hand_active:
        for hi, side in enumerate(handedness):
            hand = hand_lm_norm[hi]
            is_open, prim = _hand_gesture(hand)
            if side == "Left":
                open_left = is_open
            else:
                open_right = is_open
            if prim != "none":
                gesture = prim   # last hand wins; caller can track both

    return {
        "num_hands_visible"       : num_hands,
        "hand_near_face"          : hand_near_face,
        "hand_to_face_score"      : round(hand_face_score, 3),
        "self_touch_behavior"     : round(hand_face_score, 3),  # same signal
        "crossed_arms_probability": round(crossed_prob,    3),
        "open_palm_left"          : open_left,
        "open_palm_right"         : open_right,
        "gesture_primitive"       : gesture,
        "hand_model_active"       : hand_active,
        "pose_model_active"       : pose_active,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _finger_curl(hand: np.ndarray, tip_idx: int, mcp_idx: int) -> float:
    """
    Curl of one finger: ratio of tip-to-wrist vs mcp-to-wrist distance.
    > 1 = extended, < 1 = curled.
    """
    wrist = hand[_WRIST, :2]
    tip   = hand[tip_idx, :2]
    mcp   = hand[mcp_idx, :2]
    d_tip = euclidean(tip, wrist)
    d_mcp = euclidean(mcp, wrist)
    return safe_div(d_tip, d_mcp)


def _hand_gesture(hand: np.ndarray) -> tuple[bool, str]:
    """
    Classify a single hand into a primitive gesture.

    Returns:
        (is_open_palm, gesture_label)
        gesture_label: "open"|"fist"|"point"|"pinch"|"none"
    """
    from geometry.math_utils import safe_div

    # Finger extensions (ratio > 1.2 = extended)
    index_ext  = _finger_curl(hand, _INDEX_TIP,  _INDEX_MCP)  > 1.2
    middle_ext = _finger_curl(hand, _MIDDLE_TIP, _MIDDLE_MCP) > 1.2
    ring_ext   = _finger_curl(hand, _RING_TIP,   _RING_MCP)   > 1.2
    pinky_ext  = _finger_curl(hand, _PINKY_TIP,  _PINKY_MCP)  > 1.2

    n_extended = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    is_open = n_extended >= 3

    if n_extended >= 3:
        gesture = "open"
    elif n_extended == 0:
        gesture = "fist"
    elif index_ext and not middle_ext and not ring_ext:
        # Pinch: check index tip near thumb tip
        pinch_dist = euclidean(hand[_INDEX_TIP, :2], hand[_THUMB_TIP, :2])
        gesture    = "pinch" if pinch_dist < 0.05 else "point"
    else:
        gesture = "none"

    return is_open, gesture


def _crossed_arms(pose_norm: np.ndarray) -> float:
    """
    Estimate crossed-arms probability from pose wrist/shoulder positions.
    Returns [0,1].
    """
    p = pose_norm[0]
    lw = p[_POSE_LEFT_WRIST,    :2]
    rw = p[_POSE_RIGHT_WRIST,   :2]
    ls = p[_POSE_LEFT_SHOULDER, :2]
    rs = p[_POSE_RIGHT_SHOULDER,:2]

    mid_x = float((ls[0] + rs[0]) / 2.0)

    # Hard crossing: left wrist right of center AND right wrist left of center
    left_crossed  = lw[0] > mid_x + 0.02
    right_crossed = rw[0] < mid_x - 0.02

    if left_crossed and right_crossed:
        # Confidence scales with how far past center each wrist is
        over_l = clamp((lw[0] - mid_x) / 0.15, 0.0, 1.0)
        over_r = clamp((mid_x - rw[0]) / 0.15, 0.0, 1.0)
        return float((over_l + over_r) / 2.0)

    # Partial: one wrist past midline
    partial = 0.0
    if left_crossed:  partial += 0.3
    if right_crossed: partial += 0.3
    return float(partial)
