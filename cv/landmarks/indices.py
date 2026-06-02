"""
landmarks/indices.py

MediaPipe FaceLandmarker 478-point model landmark index constants.

Points 0–467  : face mesh (same topology as classic FaceMesh 468-pt model)
Points 468–477: iris  (left: 468–472, right: 473–477)

Convention: "left/right" follows the SUBJECT's perspective (not the camera's).
In a front-facing webcam image, the subject's LEFT eye is on the RIGHT side
of the frame.

All indices reference a SINGLE face array: landmarks[face_idx][IDX].
"""

import numpy as np

# ---------------------------------------------------------------------------
# Eyes — EAR (Eye Aspect Ratio)
# 6-point scheme: [outer, up-outer, up-inner, inner, lo-inner, lo-outer]
# EAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 * ||p1-p4|| )
# ---------------------------------------------------------------------------

LEFT_EYE_EAR_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

# ---------------------------------------------------------------------------
# Eyes — gaze bounding landmarks
# Outer = temporal side (farther from nose), Inner = nasal side
# ---------------------------------------------------------------------------

LEFT_EYE_OUTER  = 33    # temporal corner
LEFT_EYE_INNER  = 133   # nasal corner
LEFT_EYE_TOP    = 159   # superior limbus center
LEFT_EYE_BOTTOM = 145   # inferior limbus center

RIGHT_EYE_OUTER  = 263
RIGHT_EYE_INNER  = 362
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374

# Full contour loops
LEFT_EYE_CONTOUR = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_CONTOUR = [
    362, 382, 381, 380, 374, 373, 372, 371, 368,
    264, 467, 260, 259, 257, 258, 286, 414, 463,
]

# ---------------------------------------------------------------------------
# Iris (478-point model only — points 468–477)
# ---------------------------------------------------------------------------

LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

LEFT_IRIS  = [468, 469, 470, 471, 472]   # center + 4 cardinal
RIGHT_IRIS = [473, 474, 475, 476, 477]

# ---------------------------------------------------------------------------
# Mouth
# ---------------------------------------------------------------------------

MOUTH_LEFT_CORNER  = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_TOP_OUTER    = 0      # philtrum dip (top of outer lip)
MOUTH_BOTTOM_OUTER = 17     # bottom of chin chin-lip fold
MOUTH_TOP_INNER    = 13     # top of inner lip opening
MOUTH_BOTTOM_INNER = 14     # bottom of inner lip opening

# 6-point MAR: [left-corner, up-outer, up-inner, right-corner, lo-inner, lo-outer]
MOUTH_MAR_IDX = [61, 39, 0, 291, 17, 269]

# Upper/lower lateral points for lip width ratio
MOUTH_UPPER_LATERAL = [39, 37]    # left, right of upper lip
MOUTH_LOWER_LATERAL = [181, 405]  # left, right of lower lip

# ---------------------------------------------------------------------------
# Eyebrows
# ---------------------------------------------------------------------------

LEFT_BROW_INNER  = 107   # medial end (closest to nose bridge)
LEFT_BROW_PEAK   = 105   # highest point
LEFT_BROW_OUTER  = 70    # lateral end
LEFT_BROW_FULL   = [70, 63, 105, 66, 107]  # outer → inner

RIGHT_BROW_INNER = 336
RIGHT_BROW_PEAK  = 334
RIGHT_BROW_OUTER = 300
RIGHT_BROW_FULL  = [336, 296, 334, 293, 300]

# Eye-top reference for brow-raise distance
LEFT_EYE_TOP_REF  = 159
RIGHT_EYE_TOP_REF = 386

# ---------------------------------------------------------------------------
# Nose
# ---------------------------------------------------------------------------

NOSE_TIP  = 1
NOSE_BASE = 6   # root of nose (between eyes)

# ---------------------------------------------------------------------------
# Head pose — 6-point PnP model
# Indices into the 2D landmark array; paired with HEAD_POSE_3D_MODEL below.
# ---------------------------------------------------------------------------

HEAD_POSE_IDX_2D = [1, 152, 263, 33, 291, 61]
# 3D canonical face model points (mm, right-handed, nose-tip origin):
#   +x → subject's right, +y → up, +z → toward camera
HEAD_POSE_3D_MODEL = np.array([
    (  0.0,    0.0,    0.0),   # 1:   nose tip
    (  0.0, -330.0,  -65.0),   # 152: chin
    (-225.0,  170.0, -135.0),  # 263: left eye outer corner (subject's left)
    ( 225.0,  170.0, -135.0),  # 33:  right eye outer corner (subject's right)
    (-150.0, -150.0, -125.0),  # 291: left mouth corner
    ( 150.0, -150.0, -125.0),  # 61:  right mouth corner
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Face oval — for bounding box and stability
# ---------------------------------------------------------------------------

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
    67, 109,
]

# ---------------------------------------------------------------------------
# Cheeks (for AU6 / cheek-raise approximation)
# ---------------------------------------------------------------------------

LEFT_CHEEK_CENTER  = 116
RIGHT_CHEEK_CENTER = 345

# ---------------------------------------------------------------------------
# Chin / jaw
# ---------------------------------------------------------------------------

CHIN = 152
JAW_LEFT  = 234
JAW_RIGHT = 454
