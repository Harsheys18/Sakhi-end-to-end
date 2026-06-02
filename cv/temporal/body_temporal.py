"""
temporal/body_temporal.py

Temporal body posture tracking.

Consumes per-frame pose world landmarks and body_pose feature dicts to
compute signals that require frame history:

  pose_stability       : inverse of pose jitter
  movement_energy      : mean landmark velocity (world units/s)
  movement_variability : std of frame-to-frame movement
  fidget_probability   : high-frequency small movements of wrists/hands
  restlessness_score   : sustained elevated movement energy

Design: all state in BodyTemporalTracker. No I/O.
"""

from collections import deque
import numpy as np
from geometry.math_utils import clamp, rolling_slope

_BUFFER_LEN      = 90    # ~3 s at 30 fps
_FIDGET_WINDOW   = 30    # frames for fidget detection
_STILL_THRESH    = 0.005 # world-space displacement (meters) below = still
_STILL_MIN_MS    = 3000

# Pose landmarks used for movement energy (upper body focus)
_ENERGY_IDXS = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips


class BodyTemporalTracker:
    """
    Stateful temporal tracker for body pose signals.

    Usage:
        tracker = BodyTemporalTracker(fps=30)
        state   = tracker.update(pose_world_np, body_feats, timestamp_ms)
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

        self._prev_pose:    np.ndarray | None = None
        self._disp_buf:     deque[float] = deque(maxlen=_BUFFER_LEN)
        self._spine_buf:    deque[float] = deque(maxlen=_BUFFER_LEN)
        self._wrist_l_buf:  deque[np.ndarray] = deque(maxlen=_FIDGET_WINDOW)
        self._wrist_r_buf:  deque[np.ndarray] = deque(maxlen=_FIDGET_WINDOW)
        self._still_since:  float | None = None

    def update(
        self,
        pose_world:  np.ndarray | None,
        body_feats:  dict,
        timestamp_ms: float,
    ) -> dict:
        """
        Args:
            pose_world  : (num_people, 33, 3) world coords, or None.
            body_feats  : dict from features/body_pose.extract_body_pose_features()
            timestamp_ms: current time ms

        Returns:
            {
              "pose_stability"       : float  # [0,1]
              "movement_energy"      : float  # m/s mean velocity
              "movement_variability" : float  # [0,1]
              "fidget_probability"   : float  # [0,1]
              "restlessness_score"   : float  # [0,1]
              "prolonged_stillness"  : bool
              "stillness_duration_ms": float
              "spine_trend"          : float  # + = leaning forward more
            }
        """
        spine = body_feats.get("spine_angle", 0.0)
        self._spine_buf.append(spine)

        if pose_world is None or pose_world.shape[0] == 0:
            self._prev_pose = None
            return self._empty()

        p = pose_world[0]   # (33, 3)

        # ── Frame displacement ────────────────────────────────────────────────
        if self._prev_pose is not None and self._prev_pose.shape == p.shape:
            diff = p[_ENERGY_IDXS] - self._prev_pose[_ENERGY_IDXS]
            disp = float(np.linalg.norm(diff, axis=1).mean())
        else:
            disp = 0.0
        self._prev_pose = p.copy()
        self._disp_buf.append(disp)

        # Convert displacement to velocity (world units = meters)
        dt_s = 1.0 / self.fps
        velocity = disp / dt_s

        # ── Wrist positions for fidget ────────────────────────────────────────
        self._wrist_l_buf.append(p[15, :2].copy())   # left wrist xy
        self._wrist_r_buf.append(p[16, :2].copy())   # right wrist xy

        # ── Metrics ───────────────────────────────────────────────────────────
        arr = np.array(self._disp_buf, dtype=np.float32)
        movement_energy = float(arr.mean()) * (1.0 / dt_s)  # m/s

        var_raw = float(arr.std())
        movement_var = clamp(var_raw / 0.02, 0.0, 1.0)   # 0.02 m std = max var

        # Pose stability: inverse of recent variance
        pose_stability = 1.0 - movement_var

        # Fidget: high-frequency wrist movement
        fidget = self._compute_fidget()

        # Restlessness: sustained elevated movement
        restlessness = clamp(movement_energy / 0.3, 0.0, 1.0)   # 0.3 m/s = max

        # Prolonged stillness
        if disp < _STILL_THRESH:
            if self._still_since is None:
                self._still_since = timestamp_ms
            still_dur = timestamp_ms - self._still_since
        else:
            self._still_since = None
            still_dur = 0.0

        prolonged = still_dur >= _STILL_MIN_MS

        # Spine trend
        spine_arr   = np.array(self._spine_buf, dtype=np.float32)
        spine_trend = float(rolling_slope(spine_arr)) if len(spine_arr) >= 8 else 0.0

        return {
            "pose_stability"       : round(pose_stability, 3),
            "movement_energy"      : round(movement_energy, 4),
            "movement_variability" : round(movement_var,    3),
            "fidget_probability"   : round(fidget,          3),
            "restlessness_score"   : round(restlessness,    3),
            "prolonged_stillness"  : prolonged,
            "stillness_duration_ms": round(still_dur,       1),
            "spine_trend"          : round(spine_trend,     5),
        }

    def reset(self) -> None:
        self._prev_pose  = None
        self._disp_buf.clear()
        self._spine_buf.clear()
        self._wrist_l_buf.clear()
        self._wrist_r_buf.clear()
        self._still_since = None

    def _compute_fidget(self) -> float:
        """
        Detect fidgeting from wrist micro-movements.
        High-frequency (>2 Hz) small oscillations of the wrists.
        """
        if len(self._wrist_l_buf) < 8:
            return 0.0
        from geometry.math_utils import peak_frequency_hz
        l_arr = np.array(self._wrist_l_buf, dtype=np.float32)
        r_arr = np.array(self._wrist_r_buf, dtype=np.float32)
        # Use x-axis movement
        l_x = l_arr[:, 0]
        r_x = r_arr[:, 0]
        l_amp = float(l_x.max() - l_x.min())
        r_amp = float(r_x.max() - r_x.min())
        avg_amp = (l_amp + r_amp) / 2.0
        if avg_amp < 0.01:   # less than 1cm movement = not fidgeting
            return 0.0
        l_freq = peak_frequency_hz(l_x - l_x.mean(), self.fps)
        r_freq = peak_frequency_hz(r_x - r_x.mean(), self.fps)
        avg_freq = (l_freq + r_freq) / 2.0
        # 2–6 Hz wrist oscillation = fidget
        in_range = 2.0 <= avg_freq <= 6.0
        fidget = clamp(avg_amp / 0.05, 0.0, 1.0) if in_range else 0.0
        return float(fidget)

    def _empty(self) -> dict:
        return {
            "pose_stability"       : 1.0,
            "movement_energy"      : 0.0,
            "movement_variability" : 0.0,
            "fidget_probability"   : 0.0,
            "restlessness_score"   : 0.0,
            "prolonged_stillness"  : False,
            "stillness_duration_ms": 0.0,
            "spine_trend"          : 0.0,
        }
