"""
mediapipe_models/loader.py

All MediaPipe model loading and result-to-numpy conversion.
Supports: FaceLandmarker, PoseLandmarker, HandLandmarker, ObjectDetector.
"""

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)


def _find(filename: str, explicit: str | None = None) -> str:
    candidates = [c for c in [
        explicit,
        os.environ.get(filename.upper().replace(".", "_")),
        os.path.join(_HERE,   filename),
        os.path.join(_PARENT, filename),
    ] if c]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{filename} not found. Searched: {candidates}")


# ── Face ──────────────────────────────────────────────────────────────────────

def build_detector(model_path=None, num_faces=1,
                   detection_confidence=0.5, presence_confidence=0.5,
                   tracking_confidence=0.5):
    path = _find("face_landmarker.task", model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=detection_confidence,
        min_face_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def landmarks_to_numpy(result, dtype=np.float32):
    """→ (num_faces, num_pts, 2) normalized xy, or None."""
    lm = result.face_landmarks
    if not lm:
        return None
    arr = np.empty((len(lm), len(lm[0]), 2), dtype=dtype)
    for fi, face in enumerate(lm):
        for pi, pt in enumerate(face):
            arr[fi, pi, 0] = pt.x
            arr[fi, pi, 1] = pt.y
    return arr


def make_mp_image(rgb_frame: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


# ── Pose ──────────────────────────────────────────────────────────────────────

def build_pose_detector(model_path=None, num_poses=1,
                        detection_confidence=0.5, presence_confidence=0.5,
                        tracking_confidence=0.5):
    path = _find("pose_landmarker.task", model_path)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=detection_confidence,
        min_pose_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def pose_to_numpy(result, dtype=np.float32):
    """→ (num_people, 33, 3) normalized [x,y,z], or None."""
    lm = result.pose_landmarks
    if not lm:
        return None
    arr = np.zeros((len(lm), 33, 3), dtype=dtype)
    for pi, person in enumerate(lm):
        for li, pt in enumerate(person):
            arr[pi, li] = [pt.x, pt.y, pt.z]
    return arr


def pose_world_to_numpy(result, dtype=np.float32):
    """→ (num_people, 33, 3) metric world coords (meters), or None."""
    lm = result.pose_world_landmarks
    if not lm:
        return None
    arr = np.zeros((len(lm), 33, 3), dtype=dtype)
    for pi, person in enumerate(lm):
        for li, pt in enumerate(person):
            arr[pi, li] = [pt.x, pt.y, pt.z]
    return arr


# ── Hand ──────────────────────────────────────────────────────────────────────

def build_hand_detector(model_path=None, num_hands=2,
                        detection_confidence=0.5, presence_confidence=0.5,
                        tracking_confidence=0.5):
    path = _find("hand_landmarker.task", model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def hands_to_numpy(result, dtype=np.float32):
    """→ (num_hands, 21, 3) normalized [x,y,z], or None."""
    lm = result.hand_landmarks
    if not lm:
        return None
    arr = np.zeros((len(lm), 21, 3), dtype=dtype)
    for hi, hand in enumerate(lm):
        for li, pt in enumerate(hand):
            arr[hi, li] = [pt.x, pt.y, pt.z]
    return arr


def hand_world_to_numpy(result, dtype=np.float32):
    """→ (num_hands, 21, 3) metric world coords, or None."""
    lm = result.hand_world_landmarks
    if not lm:
        return None
    arr = np.zeros((len(lm), 21, 3), dtype=dtype)
    for hi, hand in enumerate(lm):
        for li, pt in enumerate(hand):
            arr[hi, li] = [pt.x, pt.y, pt.z]
    return arr


def hand_handedness(result) -> list[str]:
    """→ list of 'Left'|'Right' per detected hand."""
    return [h[0].display_name if h else "Unknown" for h in result.handedness]


# ── Object Detector ───────────────────────────────────────────────────────────

def build_object_detector(model_path=None, max_results=5, score_threshold=0.35):
    path = _find("efficientdet_lite0.tflite", model_path)
    opts = mp_vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path),
        running_mode=mp_vision.RunningMode.VIDEO,
        max_results=max_results,
        score_threshold=score_threshold,
    )
    return mp_vision.ObjectDetector.create_from_options(opts)


def detections_to_list(result) -> list[dict]:
    """
    → [{"label": str, "score": float, "bbox_norm": [x,y,w,h]}, ...]
    bbox values are raw pixels from the detector (not normalized).
    Caller normalises by dividing by image dimensions.
    """
    out = []
    for det in result.detections:
        if not det.categories:
            continue
        cat = det.categories[0]
        bb  = det.bounding_box
        out.append({
            "label": cat.category_name,
            "score": round(float(cat.score), 3),
            "bbox" : [bb.origin_x, bb.origin_y, bb.width, bb.height],
        })
    return out


# ── Backward-compat alias ─────────────────────────────────────────────────────

def _resolve_model_path(explicit=None):
    return _find("face_landmarker.task", explicit)
