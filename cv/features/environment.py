"""
features/environment.py

Full environmental feature extraction using:
  - Raw frame luminance analysis  → lighting_condition, lighting_score
  - MediaPipe ObjectDetector      → device_usage, environmental_distraction_level

device_usage detects electronic devices in frame (phone, laptop, tablet, TV,
remote, keyboard, mouse). environmental_distraction_level counts moving/salient
non-person objects.

INPUT:
  rgb_frame   : np.ndarray HxWx3 uint8
  detections  : list[dict] from mediapipe_models/loader.detections_to_list()
                Each dict: {"label": str, "score": float, "bbox": [x,y,w,h]}
  face_feats  : dict from features/face_features (for distance passthrough)
  image_wh    : (width, height) in pixels (for bbox normalisation)

OUTPUT:
  dict — see extract_environment_features() docstring.
"""

import numpy as np
from geometry.math_utils import clamp

# Luminance thresholds
_DARK_THRESH   = 40
_DIM_THRESH    = 90
_BRIGHT_THRESH = 200

# Object labels that count as electronic devices
_DEVICE_LABELS = {
    "cell phone", "laptop", "tablet", "television", "tv",
    "remote", "keyboard", "mouse", "monitor", "computer",
}

# Object labels that count as environmental distractors
_DISTRACTOR_LABELS = {
    "person", "cat", "dog", "bird", "car", "bicycle",
    "cup", "book", "bottle",
}


def extract_environment_features(
    rgb_frame:  np.ndarray,
    detections: list[dict] | None    = None,
    face_feats: dict       | None    = None,
    image_wh:   tuple[int,int] | None = None,
) -> dict:
    """
    Extract environmental context features.

    Args:
        rgb_frame  : HxWx3 uint8 RGB camera frame.
        detections : list from detections_to_list() — may be None if
                     object detector is not running this frame.
        face_feats : dict from face_features (distance passthrough).
        image_wh   : (w, h) pixels — used to normalise detector bboxes.

    Returns:
        {
          "lighting_condition"             : str    # "dark"|"dim"|"normal"|"bright"
          "lighting_score"                 : float  # [0,1]
          "lighting_uniformity"            : float  # [0,1]
          "device_usage"                   : float  # [0,1] confidence device in scene
          "device_label"                   : str    # e.g. "cell phone" or ""
          "environmental_distraction_level": float  # [0,1]
          "num_objects_detected"           : int
          "distance_from_robot_cm"         : float
          "object_detector_active"         : bool
        }
    """
    # ── Lighting ──────────────────────────────────────────────────────────────
    r = rgb_frame[:, :, 0].astype(np.float32)
    g = rgb_frame[:, :, 1].astype(np.float32)
    b = rgb_frame[:, :, 2].astype(np.float32)
    luma = 0.299 * r + 0.587 * g + 0.114 * b

    mean_luma = float(luma.mean())
    std_luma  = float(luma.std())

    if mean_luma < _DARK_THRESH:
        condition = "dark"
    elif mean_luma < _DIM_THRESH:
        condition = "dim"
    elif mean_luma > _BRIGHT_THRESH:
        condition = "bright"
    else:
        condition = "normal"

    if mean_luma <= 120:
        score = clamp(mean_luma / 120.0, 0.0, 1.0)
    else:
        score = clamp(1.0 - (mean_luma - 180.0) / 75.0, 0.0, 1.0)

    uniformity = clamp(1.0 - std_luma / max(mean_luma, 1.0), 0.0, 1.0)

    # ── Object detection ──────────────────────────────────────────────────────
    det_active     = detections is not None
    device_score   = 0.0
    device_label   = ""
    distraction    = 0.0
    num_objects    = 0

    if det_active and detections:
        h_frame, w_frame = rgb_frame.shape[:2]
        iw = image_wh[0] if image_wh else w_frame
        ih = image_wh[1] if image_wh else h_frame

        num_objects = len(detections)
        device_scores = []
        distractor_count = 0

        for d in detections:
            label = d["label"].lower().strip()
            score = d["score"]

            if label in _DEVICE_LABELS:
                device_scores.append(score)
                if score > device_score:
                    device_score = score
                    device_label = label

            elif label in _DISTRACTOR_LABELS and label != "person":
                distractor_count += 1

        # Distraction: normalise by expected max of ~3 distracting objects
        distraction = clamp(distractor_count / 3.0, 0.0, 1.0)

    dist = float((face_feats or {}).get("face_distance_cm", 0.0))

    return {
        "lighting_condition"              : condition,
        "lighting_score"                  : round(score,       3),
        "lighting_uniformity"             : round(uniformity,  3),
        "device_usage"                    : round(device_score,3),
        "device_label"                    : device_label,
        "environmental_distraction_level" : round(distraction, 3),
        "num_objects_detected"            : num_objects,
        "distance_from_robot_cm"          : round(dist,        1),
        "object_detector_active"          : det_active,
    }
