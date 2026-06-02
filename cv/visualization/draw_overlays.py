"""
visualization/draw_overlays.py

Draws all MediaPipe model outputs onto a BGR OpenCV frame:
  - Face mesh  (eyes, iris, brows, lips, nose, oval)
  - Body pose  (skeleton with colour-coded limb groups)
  - Hands      (finger skeleton with per-hand colour)
  - Objects    (bounding boxes + labels from EfficientDet)

Design rules:
  - All functions receive a BGR numpy array and draw in-place.
  - Each function is independent — call whichever are available.
  - No state. No side effects beyond modifying the passed frame.
  - Normalized landmarks ([0,1]) are scaled by (w,h) inside each function.

Color scheme (BGR):
  Face eyes/iris  : green / cyan
  Face brows      : yellow
  Face lips       : red-orange
  Face oval       : dim white
  Pose torso      : blue-violet
  Pose left arm   : green
  Pose right arm  : orange
  Pose left leg   : cyan
  Pose right leg  : pink
  Hand left       : lime green with finger tips highlighted
  Hand right      : sky blue with finger tips highlighted
  Objects         : magenta boxes with white label
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mpv

# ── Color constants (BGR) ──────────────────────────────────────────────────────

_C_EYE      = (0,   255,   0)    # green
_C_IRIS     = (255, 255,   0)    # cyan
_C_BROW     = (0,   255, 255)    # yellow
_C_LIPS     = (0,   100, 255)    # red-orange
_C_NOSE     = (255, 100,   0)    # blue-ish
_C_OVAL     = (160, 160, 160)    # dim white

_C_TORSO    = (200,  60, 120)    # blue-violet
_C_L_ARM    = (0,   200,  80)    # green
_C_R_ARM    = (0,   140, 220)    # orange
_C_L_LEG    = (200, 200,   0)    # cyan
_C_R_LEG    = (160,  80, 200)    # pink
_C_FACE_SK  = (180, 180, 180)    # gray (face skeleton on pose)

_C_HAND_L   = (80,  220,  80)    # lime green
_C_HAND_R   = (220, 160,  40)    # sky blue
_C_TIP      = (255, 255, 255)    # white fingertip dots

_C_OBJ_BOX  = (255,   0, 200)    # magenta
_C_OBJ_TXT  = (255, 255, 255)    # white

# ── Face mesh connections ──────────────────────────────────────────────────────

FLC = mpv.FaceLandmarksConnections

_FACE_REGIONS = [
    (FLC.FACE_LANDMARKS_LEFT_EYE    + FLC.FACE_LANDMARKS_RIGHT_EYE,      _C_EYE,  1),
    (FLC.FACE_LANDMARKS_LEFT_IRIS   + FLC.FACE_LANDMARKS_RIGHT_IRIS,     _C_IRIS, 1),
    (FLC.FACE_LANDMARKS_LEFT_EYEBROW+ FLC.FACE_LANDMARKS_RIGHT_EYEBROW,  _C_BROW, 1),
    (FLC.FACE_LANDMARKS_LIPS,                                             _C_LIPS, 1),
    (FLC.FACE_LANDMARKS_NOSE,                                             _C_NOSE, 1),
    (FLC.FACE_LANDMARKS_FACE_OVAL,                                        _C_OVAL, 1),
]

# ── Pose skeleton — limb groups with individual colors ────────────────────────

PLC = mpv.PoseLandmarksConnections

# Build per-connection color map from all pose connections
def _build_pose_color_map() -> list[tuple]:
    """
    Returns list of (Connection, color, thickness) for every pose connection.
    Groups connections by body region for color-coding.
    """
    all_conn = PLC.POSE_LANDMARKS

    # Landmark index groups
    FACE_IDS    = {0,1,2,3,4,5,6,7,8,9,10}
    L_ARM_IDS   = {11,13,15,17,19,21}
    R_ARM_IDS   = {12,14,16,18,20,22}
    TORSO_IDS   = {11,12,23,24}
    L_LEG_IDS   = {23,25,27,29,31}
    R_LEG_IDS   = {24,26,28,30,32}

    result = []
    for conn in all_conn:
        s, e = conn.start, conn.end
        pair = {s, e}
        if pair & FACE_IDS:
            col, thick = _C_FACE_SK, 1
        elif pair & L_ARM_IDS and not pair & R_ARM_IDS:
            col, thick = _C_L_ARM, 2
        elif pair & R_ARM_IDS and not pair & L_ARM_IDS:
            col, thick = _C_R_ARM, 2
        elif pair <= TORSO_IDS or (pair & {11,12} and pair & {23,24}):
            col, thick = _C_TORSO, 2
        elif pair & L_LEG_IDS and not pair & R_LEG_IDS:
            col, thick = _C_L_LEG, 2
        elif pair & R_LEG_IDS and not pair & L_LEG_IDS:
            col, thick = _C_R_LEG, 2
        else:
            col, thick = _C_TORSO, 1
        result.append((conn, col, thick))
    return result

_POSE_CONN_COLORS = _build_pose_color_map()

# Landmark dot radii (key landmarks slightly larger)
_POSE_KEY_IDS = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

# ── Hand skeleton ──────────────────────────────────────────────────────────────

HLC = mpv.HandLandmarksConnections
_HAND_CONNECTIONS = HLC.HAND_CONNECTIONS

# Fingertip landmark IDs
_FINGERTIPS = {4, 8, 12, 16, 20}   # thumb, index, middle, ring, pinky tips
_KNUCKLES   = {5, 9, 13, 17}       # MCP joints (base of each finger)


# ══════════════════════════════════════════════════════════════════════════════
# Public drawing functions
# ══════════════════════════════════════════════════════════════════════════════

def draw_face_mesh(
    frame:           np.ndarray,
    face_landmarks:  np.ndarray,
) -> None:
    """
    Draw face mesh connections in-place on a BGR frame.

    Args:
        frame          : HxWx3 uint8 BGR image (modified in-place).
        face_landmarks : (num_faces, num_points, 2) normalized landmarks.
    """
    if face_landmarks is None or face_landmarks.shape[0] == 0:
        return

    h, w = frame.shape[:2]
    num_faces  = face_landmarks.shape[0]
    num_points = face_landmarks.shape[1]

    for fi in range(num_faces):
        face = face_landmarks[fi]

        for connections, color, thick in _FACE_REGIONS:
            for pair in connections:
                s, e = pair.start, pair.end
                if s >= num_points or e >= num_points:
                    continue
                x1 = int(face[s, 0] * w)
                y1 = int(face[s, 1] * h)
                x2 = int(face[e, 0] * w)
                y2 = int(face[e, 1] * h)
                cv.line(frame, (x1, y1), (x2, y2), color, thick, cv.LINE_AA)


def draw_pose(
    frame:        np.ndarray,
    pose_norm:    np.ndarray,
    min_vis:      float = 0.5,
) -> None:
    """
    Draw body pose skeleton in-place on a BGR frame.

    Args:
        frame     : HxWx3 uint8 BGR (modified in-place).
        pose_norm : (num_people, 33, 3) normalized [x,y,z] landmarks.
        min_vis   : visibility threshold below which a landmark is skipped.
                    Note: pose_norm from pose_to_numpy() does not carry
                    visibility; all landmarks are drawn (vis check skipped).
    """
    if pose_norm is None or pose_norm.shape[0] == 0:
        return

    h, w = frame.shape[:2]

    for pi in range(pose_norm.shape[0]):
        person = pose_norm[pi]   # (33, 3)

        # ── Draw connections ──────────────────────────────────────────────────
        for conn, color, thick in _POSE_CONN_COLORS:
            s, e = conn.start, conn.end
            if s >= len(person) or e >= len(person):
                continue
            xs, ys = int(person[s, 0] * w), int(person[s, 1] * h)
            xe, ye = int(person[e, 0] * w), int(person[e, 1] * h)

            # Skip clearly off-frame landmarks
            if not (_in_frame(xs, ys, w, h) or _in_frame(xe, ye, w, h)):
                continue

            cv.line(frame, (xs, ys), (xe, ye), color, thick, cv.LINE_AA)

        # ── Draw landmark dots ────────────────────────────────────────────────
        for li in range(len(person)):
            x = int(person[li, 0] * w)
            y = int(person[li, 1] * h)
            if not _in_frame(x, y, w, h, margin=10):
                continue
            r = 4 if li in _POSE_KEY_IDS else 2
            cv.circle(frame, (x, y), r, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, (x, y), r, (0, 0, 0),       1,  cv.LINE_AA)


def draw_hands(
    frame:        np.ndarray,
    hand_norm:    np.ndarray,
    handedness:   list[str],
) -> None:
    """
    Draw hand skeletons in-place on a BGR frame.

    Args:
        frame      : HxWx3 uint8 BGR (modified in-place).
        hand_norm  : (num_hands, 21, 3) normalized landmarks.
        handedness : list of "Left"|"Right" per hand.
    """
    if hand_norm is None or hand_norm.shape[0] == 0:
        return

    h, w = frame.shape[:2]

    for hi in range(hand_norm.shape[0]):
        hand  = hand_norm[hi]   # (21, 3)
        side  = handedness[hi] if hi < len(handedness) else "Unknown"
        color = _C_HAND_L if side == "Left" else _C_HAND_R

        # ── Draw connections ──────────────────────────────────────────────────
        for conn in _HAND_CONNECTIONS:
            s, e = conn.start, conn.end
            if s >= len(hand) or e >= len(hand):
                continue
            xs, ys = int(hand[s, 0] * w), int(hand[s, 1] * h)
            xe, ye = int(hand[e, 0] * w), int(hand[e, 1] * h)
            cv.line(frame, (xs, ys), (xe, ye), color, 2, cv.LINE_AA)

        # ── Draw landmark dots ────────────────────────────────────────────────
        for li in range(len(hand)):
            x = int(hand[li, 0] * w)
            y = int(hand[li, 1] * h)

            if li in _FINGERTIPS:
                # Fingertips: larger white dot with colored ring
                cv.circle(frame, (x, y), 6, color,    -1, cv.LINE_AA)
                cv.circle(frame, (x, y), 6, (0, 0, 0), 1, cv.LINE_AA)
                cv.circle(frame, (x, y), 3, _C_TIP,   -1, cv.LINE_AA)
            elif li in _KNUCKLES:
                # Knuckles: medium colored dot
                cv.circle(frame, (x, y), 4, color,    -1, cv.LINE_AA)
                cv.circle(frame, (x, y), 4, (0, 0, 0), 1, cv.LINE_AA)
            else:
                # Other joints: small dot
                cv.circle(frame, (x, y), 3, color,    -1, cv.LINE_AA)

        # ── Hand label ────────────────────────────────────────────────────────
        wrist_x = int(hand[0, 0] * w)
        wrist_y = int(hand[0, 1] * h)
        cv.putText(frame, side[0],   # "L" or "R"
                   (wrist_x + 8, wrist_y + 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)


def draw_objects(
    frame:      np.ndarray,
    detections: list[dict],
    image_wh:   tuple[int, int],
) -> None:
    """
    Draw object detection bounding boxes in-place on a BGR frame.

    Args:
        frame      : HxWx3 uint8 BGR (modified in-place).
        detections : list of dicts from mediapipe_models/loader.detections_to_list().
                     Each: {"label": str, "score": float, "bbox": [x, y, w, h]}
                     bbox values are raw pixels from the detector.
        image_wh   : (width, height) of frame — used to clamp bbox to frame.
    """
    if not detections:
        return

    fw, fh = image_wh

    for det in detections:
        label = det.get("label", "?")
        score = det.get("score", 0.0)
        bbox  = det.get("bbox",  [0, 0, 0, 0])

        x1 = max(0,  int(bbox[0]))
        y1 = max(0,  int(bbox[1]))
        x2 = min(fw, int(bbox[0] + bbox[2]))
        y2 = min(fh, int(bbox[1] + bbox[3]))

        if x2 <= x1 or y2 <= y1:
            continue

        # Box
        cv.rectangle(frame, (x1, y1), (x2, y2), _C_OBJ_BOX, 2, cv.LINE_AA)

        # Label background
        text  = f"{label} {score:.0%}"
        (tw, th), bl = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(y1 - 4, th + 4)
        cv.rectangle(frame,
                     (x1, label_y - th - bl - 2),
                     (x1 + tw + 4, label_y + 2),
                     _C_OBJ_BOX, -1)
        cv.putText(frame, text, (x1 + 2, label_y - bl),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, _C_OBJ_TXT, 1, cv.LINE_AA)


def draw_all(
    frame:       np.ndarray,
    face_lm:     np.ndarray | None  = None,
    pose_norm:   np.ndarray | None  = None,
    hand_norm:   np.ndarray | None  = None,
    handedness:  list[str]  | None  = None,
    detections:  list[dict] | None  = None,
) -> None:
    """
    Convenience wrapper — draw all available detections in one call.

    Args:
        frame      : HxWx3 uint8 BGR (modified in-place).
        face_lm    : (num_faces, 478, 2) normalized, or None.
        pose_norm  : (num_people, 33, 3) normalized, or None.
        hand_norm  : (num_hands, 21, 3) normalized, or None.
        handedness : list of "Left"|"Right", or None.
        detections : list of object detection dicts, or None.
    """
    h, w = frame.shape[:2]

    # Draw in back-to-front order so face mesh is on top
    if pose_norm  is not None: draw_pose(frame, pose_norm)
    if hand_norm  is not None: draw_hands(frame, hand_norm, handedness or [])
    if face_lm    is not None: draw_face_mesh(frame, face_lm)
    if detections is not None: draw_objects(frame, detections, (w, h))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _in_frame(x: int, y: int, w: int, h: int, margin: int = 0) -> bool:
    """True if point is within frame bounds (with optional margin)."""
    return (-margin <= x < w + margin) and (-margin <= y < h + margin)
