"""
temporal/face_temporal.py

Temporal tracking of face-level signals.

Consumes per-frame output of features/face_features.extract_face_features()
and produces:

  - face_stability        : inverse of face center jitter over time
  - movement_energy       : recent mean frame displacement (proxy for body movement)
  - movement_variability  : std of frame displacement (erratic vs steady)
  - movement_trend        : slope of movement energy (increasing = fidgeting)
  - prolonged_stillness   : ms of continuous very-low movement
  - restlessness_score    : high movement_variability over window

Design: all state in FaceTracker instance. No globals. No I/O.
"""

from collections import deque
import numpy as np
from geometry.math_utils import clamp, rolling_slope

_EMA_ALPHA        = 0.30
_BUFFER_LEN       = 90       # ~3 s at 30 fps
_STILL_THRESH     = 1.5      # px displacement below this = stillness
_STILL_MIN_MS     = 3000     # ms to declare prolonged stillness


class FaceTemporalTracker:
    """
    Stateful temporal tracker for face-level movement signals.

    Usage:
        tracker = FaceTemporalTracker()
        state   = tracker.update(face_feats, timestamp_ms)
    """

    def __init__(self):
        self._disp_buf: deque[float] = deque(maxlen=_BUFFER_LEN)
        self._cx_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)
        self._cy_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)
        self._ts_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)
        self._still_since_ms: float | None = None

    def update(self, face_feats: dict, timestamp_ms: float) -> dict:
        """
        Args:
            face_feats   : dict from features/face_features.extract_face_features()
            timestamp_ms : current time in ms

        Returns:
            {
              "face_stability"       : float   # [0,1] 1=very steady
              "movement_energy"      : float   # mean px displacement (recent)
              "movement_variability" : float   # [0,1] std of displacement
              "movement_trend"       : float   # slope; + = increasing movement
              "prolonged_stillness"  : bool
              "stillness_duration_ms": float
              "restlessness_score"   : float   # [0,1]
            }
        """
        disp = face_feats.get("frame_displacement", 0.0)
        cx, cy = face_feats.get("face_center", [0.5, 0.5])

        self._disp_buf.append(disp)
        self._cx_buf.append(cx)
        self._cy_buf.append(cy)
        self._ts_buf.append(timestamp_ms)

        # Face stability: inverse of jitter in center position
        if len(self._cx_buf) >= 4:
            cx_arr = np.array(self._cx_buf, dtype=np.float32)
            cy_arr = np.array(self._cy_buf, dtype=np.float32)
            jitter = float((cx_arr.std() + cy_arr.std()) / 2.0)
            # jitter > 0.02 normalized = unstable (~13px on 640px)
            face_stability = clamp(1.0 - jitter / 0.02, 0.0, 1.0)
        else:
            face_stability = 1.0

        # Movement energy: rolling mean displacement
        arr = np.array(self._disp_buf, dtype=np.float32)
        movement_energy = float(arr.mean()) if len(arr) > 0 else 0.0

        # Movement variability: std of displacement
        movement_var_raw = float(arr.std()) if len(arr) > 1 else 0.0
        # Normalise: std > 5px = highly variable
        movement_variability = clamp(movement_var_raw / 5.0, 0.0, 1.0)

        # Trend: slope of displacement over window
        movement_trend = rolling_slope(arr) if len(arr) >= 4 else 0.0

        # Prolonged stillness: displacement below threshold continuously
        if disp < _STILL_THRESH:
            if self._still_since_ms is None:
                self._still_since_ms = timestamp_ms
            still_dur = timestamp_ms - self._still_since_ms
        else:
            self._still_since_ms = None
            still_dur = 0.0

        prolonged_stillness = still_dur >= _STILL_MIN_MS

        # Restlessness: high variability sustained over window
        restlessness = movement_variability

        return {
            "face_stability"       : round(face_stability,      3),
            "movement_energy"      : round(movement_energy,     3),
            "movement_variability" : round(movement_variability,3),
            "movement_trend"       : round(float(movement_trend),5),
            "prolonged_stillness"  : prolonged_stillness,
            "stillness_duration_ms": round(still_dur,           1),
            "restlessness_score"   : round(restlessness,        3),
        }

    def reset(self) -> None:
        self._disp_buf.clear()
        self._cx_buf.clear()
        self._cy_buf.clear()
        self._ts_buf.clear()
        self._still_since_ms = None
