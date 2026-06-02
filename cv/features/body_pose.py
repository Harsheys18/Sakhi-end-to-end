"""
features/body_pose.py

Full body posture feature extraction using MediaPipe PoseLandmarker.

Computes (per-frame, no temporal state):
  spine_angle         : degrees from vertical (+ = leaning forward)
  neck_angle          : head-to-spine angle (+ = head forward)
  shoulder_openness   : [0,1] how open/wide the shoulders are
  body_symmetry       : [0,1] left-right symmetry
  leaning_direction   : "center"|"left"|"right"|"forward"|"back"
  leaning_intensity   : [0,1]
  pose_confidence     : [0,1] based on landmark visibility scores

Temporal features (fidget_probability, pose_stability, movement_energy)
are computed by temporal/body_temporal.py which wraps this module.

INPUT:
  pose_world : np.ndarray (num_people, 33, 3) — metric world coords from
               mediapipe_models/loader.pose_world_to_numpy().
               World coords give true metric distances independent of zoom.
  pose_norm  : np.ndarray (num_people, 33, 3) — normalized image coords.
               Used for leaning direction (needs image frame reference).
  face_feats : dict from features/face_features (fallback for no-pose case)
  head_pose  : dict from features/head_pose     (neck angle reference)

OUTPUT:
  dict — see extract_body_pose_features() docstring.
"""

import numpy as np
from geometry.math_utils import clamp, safe_div, euclidean

# MediaPipe Pose landmark indices
_NOSE           = 0
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP       = 23
_RIGHT_HIP      = 24
_LEFT_EAR       = 7
_RIGHT_EAR      = 8
_LEFT_ELBOW     = 13
_RIGHT_ELBOW    = 14
_LEFT_WRIST     = 15
_RIGHT_WRIST    = 16
_LEFT_KNEE      = 25
_RIGHT_KNEE     = 26

# Visibility threshold — landmarks below this are treated as occluded
_VIS_THRESH = 0.5


def extract_body_pose_features(
    pose_world: np.ndarray | None = None,
    pose_norm:  np.ndarray | None = None,
    face_feats: dict | None = None,
    head_pose:  dict | None = None,
) -> dict:
    """
    Extract body posture features from PoseLandmarker output.

    Falls back to face-derived proxies when pose_world is None.

    Args:
        pose_world : (num_people, 33, 3) metric world landmarks, or None.
        pose_norm  : (num_people, 33, 3) normalized image landmarks, or None.
        face_feats : dict from face_features (fallback).
        head_pose  : dict from head_pose (neck angle reference).

    Returns:
        {
          "spine_angle"       : float   # degrees, + = forward lean
          "neck_angle"        : float   # degrees, + = head forward
          "shoulder_openness" : float   # [0,1]
          "body_symmetry"     : float   # [0,1]
          "leaning_direction" : str
          "leaning_intensity" : float   # [0,1]
          "pose_confidence"   : float   # [0,1]
          "pose_model_active" : bool
        }
    """
    if pose_world is not None and pose_world.shape[0] > 0:
        return _from_pose(pose_world, pose_norm)

    # ── Face-only fallback ────────────────────────────────────────────────────
    face_feats = face_feats or {}
    head_pose  = head_pose  or {}

    cx         = face_feats.get("face_center", [0.5, 0.5])[0]
    h_offset   = cx - 0.5
    intensity  = clamp(abs(h_offset) / 0.20, 0.0, 1.0)
    direction  = "center" if abs(h_offset) < 0.05 else ("right" if h_offset > 0 else "left")

    return {
        "spine_angle"       : 0.0,
        "neck_angle"        : round(float(head_pose.get("head_pitch", 0.0)), 2),
        "shoulder_openness" : 0.0,
        "body_symmetry"     : 0.0,
        "leaning_direction" : direction,
        "leaning_intensity" : round(intensity, 3),
        "pose_confidence"   : 0.0,
        "pose_model_active" : False,
    }


def _from_pose(pose_world: np.ndarray, pose_norm: np.ndarray | None) -> dict:
    """Full pose feature extraction from world-coordinate landmarks."""
    w = pose_world[0]   # (33, 3), metric coords

    # Helper: get point, return None if index out of range
    def pt(idx): return w[idx] if idx < len(w) else None

    ls = pt(_LEFT_SHOULDER)
    rs = pt(_RIGHT_SHOULDER)
    lh = pt(_LEFT_HIP)
    rh = pt(_RIGHT_HIP)
    le = pt(_LEFT_EAR)
    re = pt(_RIGHT_EAR)
    nose = pt(_NOSE)

    # ── Spine angle ───────────────────────────────────────────────────────────
    # Vector from hip midpoint → shoulder midpoint in world coords.
    # In world space: y goes up, z goes toward camera.
    # Angle from vertical (0,1,0).
    shoulder_mid = (ls + rs) / 2.0
    hip_mid      = (lh + rh) / 2.0
    spine_vec    = shoulder_mid - hip_mid   # should point upward

    spine_len = float(np.linalg.norm(spine_vec))
    if spine_len > 1e-4:
        # Angle from vertical axis (y-axis in world)
        spine_y    = float(spine_vec[1])
        spine_xz   = float(np.linalg.norm(spine_vec[[0, 2]]))
        spine_angle = float(np.degrees(np.arctan2(spine_xz, abs(spine_y))))
    else:
        spine_angle = 0.0

    # ── Neck angle ────────────────────────────────────────────────────────────
    # Head (ear midpoint) relative to shoulder midpoint.
    if le is not None and re is not None:
        head_mid = (le + re) / 2.0
        neck_vec = head_mid - shoulder_mid
        neck_xz  = float(np.linalg.norm(neck_vec[[0, 2]]))
        neck_y   = float(abs(neck_vec[1]))
        neck_angle = float(np.degrees(np.arctan2(neck_xz, neck_y + 1e-6)))
    else:
        neck_angle = 0.0

    # ── Shoulder openness ─────────────────────────────────────────────────────
    # Shoulder width relative to hip width. In world coords this is scale-invariant.
    shoulder_w = euclidean(ls[:2], rs[:2])
    hip_w      = euclidean(lh[:2], rh[:2])
    # Fully open ≈ shoulder_w ≈ 1.2–1.5× hip_w
    # Crossed/closed ≈ shoulder_w < hip_w
    ratio          = safe_div(shoulder_w, hip_w + 1e-6)
    shoulder_open  = clamp((ratio - 0.6) / 0.8, 0.0, 1.0)

    # ── Body symmetry ─────────────────────────────────────────────────────────
    # Compare left and right sides: distance from shoulder to hip.
    l_seg = euclidean(ls, lh)
    r_seg = euclidean(rs, rh)
    diff  = abs(l_seg - r_seg)
    ref   = (l_seg + r_seg) / 2.0 + 1e-6
    body_symmetry = clamp(1.0 - diff / ref, 0.0, 1.0)

    # ── Leaning direction ─────────────────────────────────────────────────────
    # Use normalized image coords for left-right (world x can flip with mirroring).
    if pose_norm is not None and pose_norm.shape[0] > 0:
        n      = pose_norm[0]
        s_cx   = float((n[_LEFT_SHOULDER, 0] + n[_RIGHT_SHOULDER, 0]) / 2.0)
        h_cx   = float((n[_LEFT_HIP, 0]      + n[_RIGHT_HIP, 0])      / 2.0)
        lr_off = s_cx - h_cx   # positive = shoulders right of hips = lean right
        # Forward/back: use world z (+ = toward camera = leaning forward)
        fb_off = float(-(shoulder_mid[2] - hip_mid[2]))  # + = forward lean
    else:
        lr_off = 0.0
        fb_off = spine_angle / 30.0  # rough forward lean from spine angle

    if abs(lr_off) > 0.05 or abs(fb_off) > 0.1:
        if abs(lr_off) >= abs(fb_off):
            direction = "right" if lr_off > 0 else "left"
        else:
            direction = "forward" if fb_off > 0 else "back"
        intensity = clamp(max(abs(lr_off) / 0.15, abs(fb_off) / 0.20), 0.0, 1.0)
    else:
        direction = "center"
        intensity = 0.0

    # ── Pose confidence ───────────────────────────────────────────────────────
    # Mean visibility of key upper-body landmarks (pose_world doesn't carry
    # visibility; use pose_norm visibility if available).
    pose_confidence = 1.0   # world landmarks present = detected

    return {
        "spine_angle"       : round(spine_angle,    2),
        "neck_angle"        : round(neck_angle,     2),
        "shoulder_openness" : round(shoulder_open,  3),
        "body_symmetry"     : round(body_symmetry,  3),
        "leaning_direction" : direction,
        "leaning_intensity" : round(intensity,      3),
        "pose_confidence"   : pose_confidence,
        "pose_model_active" : True,
    }
