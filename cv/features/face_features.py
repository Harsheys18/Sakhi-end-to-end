"""
features/face_features.py

Per-frame face-level features:
  - face_presence       : bool
  - face_count          : int
  - face_bbox           : (x_min, y_min, x_max, y_max) normalized
  - face_bbox_size      : normalized area
  - face_distance_cm    : rough metric estimate (requires image_wh + focal_px)
  - face_center         : (cx, cy) normalized
  - face_yaw_proxy      : quick left-right offset without full PnP
  - frame_displacement  : mean landmark displacement from previous frame (stability)

INPUT:
  landmarks   : np.ndarray (num_faces, num_points, 2|3), normalized
  face_idx    : int
  prev_face   : np.ndarray (num_points, 2|3) | None — previous frame's landmarks
  image_wh    : (w, h) in pixels (optional; needed for distance estimate)

OUTPUT:
  dict — see extract_face_features() docstring.

Design: pure functions, no state.
"""

import numpy as np
from geometry.math_utils import bbox_from_indices, safe_div
from landmarks.indices import FACE_OVAL, NOSE_TIP, CHIN

# Approximate average adult face width in mm (used for distance estimate)
_FACE_WIDTH_MM = 150.0


def extract_face_features(
    landmarks: np.ndarray,
    face_idx:  int = 0,
    prev_face: np.ndarray | None = None,
    image_wh:  tuple[int, int] | None = None,
) -> dict:
    """
    Extract per-frame face-level features.

    Args:
        landmarks : (num_faces, num_points, 2|3) normalized landmarks.
        face_idx  : which face to process.
        prev_face : (num_points, 2|3) landmarks from the immediately
                    preceding frame — used to compute frame displacement.
        image_wh  : (width, height) pixels — needed for distance estimate.

    Returns:
        {
          "face_presence"       : bool
          "face_count"          : int
          "face_bbox"           : [x_min, y_min, x_max, y_max]
          "face_bbox_size"      : float   # normalized area (0–1)
          "face_center"         : [cx, cy]
          "face_distance_cm"    : float   # 0.0 if image_wh not provided
          "face_yaw_proxy"      : float   # [-1, 1] left/right lean (approx)
          "frame_displacement"  : float   # mean px movement; 0.0 if no prev
          "confidence"          : float
        }
    """
    num_faces = landmarks.shape[0] if landmarks is not None else 0

    if landmarks is None or face_idx >= num_faces:
        return _empty_result()

    face = landmarks[face_idx, :, :2]   # (num_points, 2) xy only

    # --- Bounding box from face oval landmarks ---
    x_min, y_min, x_max, y_max = bbox_from_indices(face, FACE_OVAL)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_area = bbox_w * bbox_h
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    # --- Distance estimate ---
    # d = (face_width_mm * focal_length_px) / face_width_px
    # focal_length_px ≈ image_width (standard 60° FOV approximation)
    distance_cm = 0.0
    if image_wh is not None and bbox_w > 1e-4:
        w_px        = image_wh[0]
        face_w_px   = bbox_w * w_px
        focal_px    = float(w_px)
        distance_cm = (_FACE_WIDTH_MM * focal_px / face_w_px) / 10.0  # mm → cm

    # --- Yaw proxy: horizontal offset of nose from face bbox center ---
    nose_x = float(face[NOSE_TIP, 0])
    # Positive → nose right of center → face turning left (subject's left = yaw right)
    face_yaw_proxy = safe_div(nose_x - cx, bbox_w * 0.5)

    # --- Frame displacement (landmark stability between frames) ---
    frame_displacement = 0.0
    if prev_face is not None and prev_face.shape[0] >= face.shape[0]:
        prev_xy = prev_face[:, :2]
        # Scale to pixels if possible so the value is interpretable
        if image_wh is not None:
            scale = np.array(image_wh, dtype=np.float32)
            diff  = (face - prev_xy[:face.shape[0]]) * scale
        else:
            diff  = face - prev_xy[:face.shape[0]]
        frame_displacement = float(np.linalg.norm(diff, axis=1).mean())

    return {
        "face_presence"      : True,
        "face_count"         : num_faces,
        "face_bbox"          : [round(x_min, 4), round(y_min, 4),
                                 round(x_max, 4), round(y_max, 4)],
        "face_bbox_size"     : round(float(bbox_area), 5),
        "face_center"        : [round(cx, 4), round(cy, 4)],
        "face_distance_cm"   : round(distance_cm, 1),
        "face_yaw_proxy"     : round(float(face_yaw_proxy), 3),
        "frame_displacement" : round(frame_displacement, 3),
        "confidence"         : 1.0,
    }


def _empty_result() -> dict:
    return {
        "face_presence"      : False,
        "face_count"         : 0,
        "face_bbox"          : [0.0, 0.0, 0.0, 0.0],
        "face_bbox_size"     : 0.0,
        "face_center"        : [0.0, 0.0],
        "face_distance_cm"   : 0.0,
        "face_yaw_proxy"     : 0.0,
        "frame_displacement" : 0.0,
        "confidence"         : 0.0,
    }
