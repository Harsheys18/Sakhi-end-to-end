"""
temporal/head_pose_temporal.py

Temporal tracking of head pose over a rolling window.

Consumes per-frame output of features/head_pose.extract_head_pose() and
maintains rolling deques to produce stable estimates of:

  - Smoothed yaw / pitch / roll (EMA)
  - Attention direction (categorical)
  - Head movement velocity (deg/s)
  - Nodding detection  (pitch oscillation at 1–4 Hz)
  - Head-shaking detection (yaw oscillation at 1–4 Hz)
  - Head stability score
  - Sustained off-axis attention duration

Design: all state lives in HeadPoseTracker instance. No globals. No I/O.
"""

from collections import deque
import numpy as np
from geometry.math_utils import rolling_slope, peak_frequency_hz, clamp

# ── Configuration ─────────────────────────────────────────────────────────────

_EMA_ALPHA         = 0.25     # EMA smoothing for angles (lower = smoother)
_BUFFER_LEN        = 90       # history depth in frames (~3 s at 30 fps)
_VELOCITY_WINDOW   = 5        # frames for velocity finite-difference

# Attention thresholds (degrees, post-smoothing)
_YAW_CENTER_THRESH   = 15.0   # |yaw|  < this → looking forward
_PITCH_CENTER_THRESH = 15.0   # |pitch|< this → looking forward
_YAW_FAR_THRESH      = 35.0   # |yaw|  > this → clearly away
_PITCH_FAR_THRESH    = 30.0   # |pitch|> this → clearly away

# Nod / shake detection
_OSCILLATION_LO_HZ = 0.8     # minimum frequency to classify as nod/shake
_OSCILLATION_HI_HZ = 4.0     # maximum (above this is noise)
_OSCILLATION_AMP   = 5.0     # minimum peak-to-peak amplitude (degrees)


class HeadPoseTracker:
    """
    Stateful temporal tracker for head pose for one face.

    Usage:
        tracker = HeadPoseTracker(fps=30)
        state   = tracker.update(head_pose_dict, timestamp_ms)
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

        # EMA state
        self._ema_yaw:   float = 0.0
        self._ema_pitch: float = 0.0
        self._ema_roll:  float = 0.0
        self._initialized = False

        # Rolling buffers for velocity and oscillation
        self._yaw_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)
        self._pitch_buf: deque[float] = deque(maxlen=_BUFFER_LEN)

        # Timestamp buffer for velocity estimation
        self._ts_buf: deque[float] = deque(maxlen=_BUFFER_LEN)

        # Off-axis attention timer
        self._offaxis_since_ms: float | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, head_pose: dict, timestamp_ms: float) -> dict:
        """
        Ingest one frame's head pose and return temporal state.

        Args:
            head_pose    : dict from features/head_pose.extract_head_pose()
            timestamp_ms : current time in ms (monotonic)

        Returns:
            {
              # Smoothed angles
              "yaw"          : float   # degrees, EMA-smoothed
              "pitch"        : float   # degrees, EMA-smoothed
              "roll"         : float   # degrees, EMA-smoothed

              # Movement
              "yaw_velocity"   : float   # deg/s
              "pitch_velocity" : float   # deg/s

              # Oscillation (nodding/shaking)
              "nod_detected"   : bool
              "shake_detected" : bool
              "nod_frequency"  : float   # Hz
              "shake_frequency": float   # Hz

              # Attention
              "attention_direction"       : str    # "forward"|"left"|"right"|"up"|"down"|"away"
              "attention_score"           : float  # [0,1] 1=fully forward
              "offaxis_duration_ms"       : float  # ms spent off-axis continuously

              # Stability
              "head_stability"   : float  # [0,1] 1=very still

              # Meta
              "confidence"       : float
            }
        """
        confidence = head_pose.get("confidence", 0.0)
        raw_yaw    = head_pose.get("head_yaw",   0.0)
        raw_pitch  = head_pose.get("head_pitch", 0.0)
        raw_roll   = head_pose.get("head_roll",  0.0)

        # ── EMA smoothing ────────────────────────────────────────────────────
        if not self._initialized:
            self._ema_yaw   = raw_yaw
            self._ema_pitch = raw_pitch
            self._ema_roll  = raw_roll
            self._initialized = True
        else:
            a = _EMA_ALPHA
            self._ema_yaw   = a * raw_yaw   + (1 - a) * self._ema_yaw
            self._ema_pitch = a * raw_pitch + (1 - a) * self._ema_pitch
            self._ema_roll  = a * raw_roll  + (1 - a) * self._ema_roll

        yaw   = self._ema_yaw
        pitch = self._ema_pitch
        roll  = self._ema_roll

        # ── Update buffers ───────────────────────────────────────────────────
        self._yaw_buf.append(yaw)
        self._pitch_buf.append(pitch)
        self._ts_buf.append(timestamp_ms)

        # ── Velocity (finite difference over short window) ───────────────────
        yaw_vel, pitch_vel = self._compute_velocity()

        # ── Oscillation detection ────────────────────────────────────────────
        nod_freq   = self._oscillation_freq(self._pitch_buf)
        shake_freq = self._oscillation_freq(self._yaw_buf)

        nod_amp   = self._amplitude(self._pitch_buf)
        shake_amp = self._amplitude(self._yaw_buf)

        nod_detected   = (
            _OSCILLATION_LO_HZ <= nod_freq   <= _OSCILLATION_HI_HZ
            and nod_amp   > _OSCILLATION_AMP
        )
        shake_detected = (
            _OSCILLATION_LO_HZ <= shake_freq  <= _OSCILLATION_HI_HZ
            and shake_amp > _OSCILLATION_AMP
        )

        # ── Attention direction ──────────────────────────────────────────────
        direction, attn_score = self._attention(yaw, pitch)

        # Track duration of continuous off-axis attention
        if direction == "forward":
            self._offaxis_since_ms = None
            offaxis_dur = 0.0
        else:
            if self._offaxis_since_ms is None:
                self._offaxis_since_ms = timestamp_ms
            offaxis_dur = timestamp_ms - self._offaxis_since_ms

        # ── Head stability ───────────────────────────────────────────────────
        stability = self._head_stability()

        return {
            "yaw"                  : round(yaw,   2),
            "pitch"                : round(pitch, 2),
            "roll"                 : round(roll,  2),
            "yaw_velocity"         : round(yaw_vel,   2),
            "pitch_velocity"       : round(pitch_vel, 2),
            "nod_detected"         : nod_detected,
            "shake_detected"       : shake_detected,
            "nod_frequency"        : round(nod_freq,   3),
            "shake_frequency"      : round(shake_freq, 3),
            "attention_direction"  : direction,
            "attention_score"      : round(attn_score, 3),
            "offaxis_duration_ms"  : round(offaxis_dur, 1),
            "head_stability"       : round(stability, 3),
            "confidence"           : confidence,
        }

    def reset(self) -> None:
        self._yaw_buf.clear()
        self._pitch_buf.clear()
        self._ts_buf.clear()
        self._initialized     = False
        self._offaxis_since_ms = None

    # ── Private ────────────────────────────────────────────────────────────────

    def _compute_velocity(self) -> tuple[float, float]:
        """deg/s over the last _VELOCITY_WINDOW frames."""
        n = min(len(self._yaw_buf), _VELOCITY_WINDOW)
        if n < 2 or len(self._ts_buf) < n:
            return 0.0, 0.0
        dt_ms = self._ts_buf[-1] - self._ts_buf[-n]
        if dt_ms < 1.0:
            return 0.0, 0.0
        dt_s     = dt_ms / 1000.0
        yaw_list  = list(self._yaw_buf)
        pitch_list = list(self._pitch_buf)
        yaw_vel   = (yaw_list[-1]   - yaw_list[-n])   / dt_s
        pitch_vel = (pitch_list[-1] - pitch_list[-n]) / dt_s
        return float(yaw_vel), float(pitch_vel)

    def _oscillation_freq(self, buf: deque) -> float:
        """Peak frequency of angle buffer, using FFT."""
        if len(buf) < 16:
            return 0.0
        arr = np.array(buf, dtype=np.float32)
        return peak_frequency_hz(arr, self.fps)

    def _amplitude(self, buf: deque) -> float:
        """Peak-to-peak amplitude in degrees over the buffer."""
        if len(buf) < 2:
            return 0.0
        arr = np.array(buf, dtype=np.float32)
        return float(arr.max() - arr.min())

    def _attention(self, yaw: float, pitch: float) -> tuple[str, float]:
        """
        Map (yaw, pitch) to a categorical direction and a forward attention score.
        """
        abs_yaw   = abs(yaw)
        abs_pitch = abs(pitch)

        if abs_yaw < _YAW_CENTER_THRESH and abs_pitch < _PITCH_CENTER_THRESH:
            direction = "forward"
        elif abs_yaw > _YAW_FAR_THRESH or abs_pitch > _PITCH_FAR_THRESH:
            direction = "away"
        elif abs_yaw >= abs_pitch:
            direction = "right" if yaw > 0 else "left"
        else:
            direction = "down" if pitch > 0 else "up"

        # Attention score: 1 = perfectly forward, decays with angle
        yaw_score   = clamp(1.0 - abs_yaw   / _YAW_FAR_THRESH,   0.0, 1.0)
        pitch_score = clamp(1.0 - abs_pitch / _PITCH_FAR_THRESH, 0.0, 1.0)
        attn_score  = yaw_score * pitch_score   # product: both must be forward

        return direction, float(attn_score)

    def _head_stability(self) -> float:
        """
        Inverse of recent head movement magnitude.
        1.0 = completely still, 0.0 = large rapid movements.
        """
        n = min(len(self._yaw_buf), 15)
        if n < 2:
            return 1.0
        yaw_arr   = np.array(list(self._yaw_buf)[-n:], dtype=np.float32)
        pitch_arr = np.array(list(self._pitch_buf)[-n:], dtype=np.float32)
        std = float((yaw_arr.std() + pitch_arr.std()) / 2.0)
        # std > 10 deg → unstable; < 1 deg → very stable
        return clamp(1.0 - std / 10.0, 0.0, 1.0)
