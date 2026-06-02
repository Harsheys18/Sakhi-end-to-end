"""
features/head_pose.py

Head pose estimation from 6 facial landmark correspondences.

Uses OpenCV's solvePnP (EPNP algorithm) with a canonical 3-D face model
to recover the camera-relative rotation. Rotation vector is converted to
Euler angles via RQDecomp3x3.

Output angles (degrees):
  pitch :  + = looking down  (nodding forward)
  yaw   :  + = turning right (subject's right)
  roll  :  + = tilting right

INPUT:
  landmarks  : np.ndarray (num_faces, num_points, 2|3), normalized.
               xy must be in [0, 1] relative to image.
  face_idx   : int
  image_wh   : (width, height) pixels — REQUIRED for correct projection.

OUTPUT:
  dict — see extract_head_pose() docstring.

Design: no state, no I/O. Returns zeroed dict on failure.
"""

import cv2
import numpy as np
from landmarks.indices import HEAD_POSE_IDX_2D, HEAD_POSE_3D_MODEL


def extract_head_pose(
    landmarks: np.ndarray,
    face_idx:  int = 0,
    image_wh:  tuple[int, int] | None = None,
) -> dict:
    """
    Estimate head pose for one face.

    Args:
        landmarks : (num_faces, num_points, 2|3) normalized.
        face_idx  : face to process.
        image_wh  : (width, height) in pixels.

    Returns:
        {
          "head_yaw"          : float   # degrees, + = right
          "head_pitch"        : float   # degrees, + = down
          "head_roll"         : float   # degrees, + = right tilt
          "rotation_vector"   : list[float]  # Rodrigues vector (3,)
          "reprojection_error": float   # px; lower = more confident
          "confidence"        : float   # [0, 1]
        }
    """
    if landmarks is None or face_idx >= landmarks.shape[0]:
        return _empty_result()
    if image_wh is None:
        # Cannot run solvePnP without camera projection parameters
        return _empty_result(reason="image_wh required for head pose")

    w, h = image_wh
    face  = landmarks[face_idx]   # (num_points, 2|3)

    # --- Build 2-D image points (pixel space) ---
    pts_2d = np.array(
        [(face[i, 0] * w, face[i, 1] * h) for i in HEAD_POSE_IDX_2D],
        dtype=np.float64,
    )

    # --- Approximate camera matrix (principal point at image center) ---
    # focal_length ≈ image_width gives a reasonable approximation for
    # typical webcam horizontal FOV (~60°).
    focal_px = float(w)
    cam_mat  = np.array([
        [focal_px,   0.0,    w / 2.0],
        [  0.0,   focal_px, h / 2.0],
        [  0.0,     0.0,       1.0 ],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # assume no lens distortion

    # --- Solve PnP ---
    success, rvec, tvec = cv2.solvePnP(
        HEAD_POSE_3D_MODEL,
        pts_2d,
        cam_mat,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success:
        return _empty_result(reason="solvePnP failed")

    # --- Rotation vector → rotation matrix → Euler angles ---
    rot_mat, _ = cv2.Rodrigues(rvec)
    # RQDecomp3x3 returns angles in degrees
    euler, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    pitch_deg, yaw_deg, roll_deg = euler

    # --- Reprojection error → confidence ---
    proj_pts, _ = cv2.projectPoints(
        HEAD_POSE_3D_MODEL, rvec, tvec, cam_mat, dist_coeffs
    )
    proj_pts = proj_pts.reshape(-1, 2)
    reproj_err = float(np.linalg.norm(pts_2d - proj_pts, axis=1).mean())

    # Map reprojection error → confidence
    # < 2 px → high confidence; > 15 px → low confidence
    confidence = float(np.clip(1.0 - (reproj_err / 15.0), 0.0, 1.0))

    return {
        "head_pitch"         : round(float(pitch_deg), 2),
        "head_yaw"           : round(float(yaw_deg),   2),
        "head_roll"          : round(float(roll_deg),  2),
        "rotation_vector"    : [round(float(v), 4) for v in rvec.flatten()],
        "reprojection_error" : round(reproj_err, 3),
        "confidence"         : round(confidence, 3),
    }


def _empty_result(reason: str = "no face") -> dict:
    return {
        "head_pitch"         : 0.0,
        "head_yaw"           : 0.0,
        "head_roll"          : 0.0,
        "rotation_vector"    : [0.0, 0.0, 0.0],
        "reprojection_error" : 0.0,
        "confidence"         : 0.0,
    }
