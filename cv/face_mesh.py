"""
face_mesh.py

Visualization of MediaPipe face landmarks on an RGB image.

Kept at project root for backward compatibility.

The original draw_landmarks_on_image() accepted the raw MediaPipe
landmark objects. This version is adapted to accept the numpy array
produced by the pipeline (shape: num_faces × num_points × 2, normalized),
which is what main.py passes after smoothing.

Color scheme (RGB):
  Eyes / iris  : green / cyan
  Eyebrows     : yellow
  Lips         : red
  Nose         : blue
  Face oval    : white
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

FLC = vision.FaceLandmarksConnections

# RGB colors
_GREEN  = (0,   255,   0)
_BLUE   = (0,     0, 255)
_RED    = (255,   0,   0)
_CYAN   = (0,   255, 255)
_YELLOW = (255, 255,   0)
_WHITE  = (255, 255, 255)

# Connection groups to draw
_REGIONS = [
    (FLC.FACE_LANDMARKS_LEFT_EYE   + FLC.FACE_LANDMARKS_RIGHT_EYE,       _GREEN),
    (FLC.FACE_LANDMARKS_LEFT_IRIS  + FLC.FACE_LANDMARKS_RIGHT_IRIS,       _CYAN),
    (FLC.FACE_LANDMARKS_LEFT_EYEBROW + FLC.FACE_LANDMARKS_RIGHT_EYEBROW,  _YELLOW),
    (FLC.FACE_LANDMARKS_LIPS,                                              _RED),
    (FLC.FACE_LANDMARKS_NOSE,                                              _BLUE),
    (FLC.FACE_LANDMARKS_FACE_OVAL,                                         _WHITE),
]


def draw_landmarks_on_image(
    rgb_image:           np.ndarray,
    face_landmarks_list: np.ndarray,
) -> np.ndarray:
    """
    Draw face mesh connections on an RGB image.

    Args:
        rgb_image          : HxWx3 uint8 RGB image.
        face_landmarks_list: numpy array (num_faces, num_points, 2),
                             landmarks in normalized [0, 1] coordinates.

    Returns:
        Annotated copy of rgb_image (RGB uint8).
    """
    annotated = np.copy(rgb_image)
    h, w = rgb_image.shape[:2]

    num_faces  = face_landmarks_list.shape[0]
    num_points = face_landmarks_list.shape[1]

    for fi in range(num_faces):
        face = face_landmarks_list[fi]   # (num_points, 2)

        for connections, color in _REGIONS:
            for pair in connections:
                s, e = pair.start, pair.end
                if s >= num_points or e >= num_points:
                    continue
                x1 = int(face[s, 0] * w)
                y1 = int(face[s, 1] * h)
                x2 = int(face[e, 0] * w)
                y2 = int(face[e, 1] * h)
                cv.line(annotated, (x1, y1), (x2, y2), color, 1, cv.LINE_AA)

    return annotated
