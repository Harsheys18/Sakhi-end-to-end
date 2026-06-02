"""
temporal/mouth_temporal.py

Temporal mouth tracking over a rolling window.

Consumes per-frame output of features/mouth_features.extract_mouth_features()
and produces:

  - EMA-smoothed MAR, smile, tension
  - Speaking activity detection (MAR oscillation at 2–8 Hz)
  - Yawning detection (sustained high jaw_drop > threshold)
  - Lip tension trend (rising tension is a stress signal)
  - Expression stability

Design: all state in MouthTracker. No globals. No I/O.
"""

from collections import deque
import numpy as np
from geometry.math_utils import clamp, peak_frequency_hz, rolling_slope

# ── Configuration ─────────────────────────────────────────────────────────────

_EMA_ALPHA         = 0.30
_BUFFER_LEN        = 90       # ~3 s at 30 fps

# Speaking: MAR oscillates while talking
_SPEAK_FREQ_LO_HZ  = 1.5     # Hz minimum for speech
_SPEAK_FREQ_HI_HZ  = 8.0     # Hz maximum for speech
_SPEAK_MAR_AMP     = 0.05    # minimum MAR amplitude to count as speaking

# Yawning: jaw_drop sustained above threshold
_YAWN_JAW_THRESH   = 0.65    # jaw_drop value
_YAWN_MIN_DUR_MS   = 1500    # minimum duration of wide opening
_YAWN_MAX_DUR_MS   = 6000    # above this, classify as something else

# Tension trend window
_TREND_WINDOW      = 60      # frames for trend computation


class MouthTracker:
    """
    Stateful temporal tracker for mouth features for one face.

    Usage:
        tracker = MouthTracker(fps=30)
        state   = tracker.update(mouth_feats, timestamp_ms)
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

        # EMA state
        self._mar:     float = 0.0
        self._smile:   float = 0.0
        self._tension: float = 0.0
        self._initialized = False

        # Rolling buffers
        self._mar_buf:     deque[float] = deque(maxlen=_BUFFER_LEN)
        self._smile_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)
        self._tension_buf: deque[float] = deque(maxlen=_BUFFER_LEN)
        self._jaw_buf:     deque[float] = deque(maxlen=_BUFFER_LEN)
        self._ts_buf:      deque[float] = deque(maxlen=_BUFFER_LEN)

        # Yawn state machine
        self._yawn_start_ms: float | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, mouth_feats: dict, timestamp_ms: float) -> dict:
        """
        Ingest one frame's mouth features and return temporal state.

        Args:
            mouth_feats  : dict from features/mouth_features.extract_mouth_features()
            timestamp_ms : current time in ms

        Returns:
            {
              "MAR"                   : float   # EMA-smoothed
              "smile_intensity"       : float   # EMA-smoothed
              "lip_tension"           : float   # EMA-smoothed
              "lip_compression"       : float   # pass-through
              "speaking_activity"     : float   # [0,1] probability of speaking
              "speaking_frequency_hz" : float   # detected MAR oscillation Hz
              "yawning"               : bool
              "yawn_duration_ms"      : float
              "tension_trend"         : float   # + = tension rising (stress)
              "smile_trend"           : float   # + = smile increasing
              "expression_stability"  : float   # [0,1] 1 = stable mouth
              "confidence"            : float
            }
        """
        raw_mar     = mouth_feats.get("MAR",             0.0)
        raw_smile   = mouth_feats.get("smile_intensity", 0.0)
        raw_tension = mouth_feats.get("lip_tension",     0.0)
        raw_jaw     = mouth_feats.get("jaw_drop",        0.0)
        lip_comp    = mouth_feats.get("lip_compression", 0.0)
        conf        = mouth_feats.get("confidence",      0.0)

        # ── EMA smoothing ────────────────────────────────────────────────────
        if not self._initialized:
            self._mar     = raw_mar
            self._smile   = raw_smile
            self._tension = raw_tension
            self._initialized = True
        else:
            a = _EMA_ALPHA
            self._mar     = a * raw_mar     + (1 - a) * self._mar
            self._smile   = a * raw_smile   + (1 - a) * self._smile
            self._tension = a * raw_tension + (1 - a) * self._tension

        # ── Buffer update ────────────────────────────────────────────────────
        self._mar_buf.append(self._mar)
        self._smile_buf.append(self._smile)
        self._tension_buf.append(self._tension)
        self._jaw_buf.append(raw_jaw)
        self._ts_buf.append(timestamp_ms)

        # ── Speaking detection ───────────────────────────────────────────────
        speak_prob, speak_hz = self._detect_speaking()

        # ── Yawn detection ───────────────────────────────────────────────────
        yawning, yawn_dur = self._update_yawn(raw_jaw, timestamp_ms)

        # ── Trends ───────────────────────────────────────────────────────────
        tension_trend = self._trend(self._tension_buf)
        smile_trend   = self._trend(self._smile_buf)

        # ── Expression stability ─────────────────────────────────────────────
        stability = self._expression_stability()

        return {
            "MAR"                   : round(self._mar,     4),
            "smile_intensity"       : round(self._smile,   3),
            "lip_tension"           : round(self._tension, 3),
            "lip_compression"       : round(lip_comp,      3),
            "speaking_activity"     : round(speak_prob,    3),
            "speaking_frequency_hz" : round(speak_hz,      3),
            "yawning"               : yawning,
            "yawn_duration_ms"      : round(yawn_dur,      1),
            "tension_trend"         : round(tension_trend, 4),
            "smile_trend"           : round(smile_trend,   4),
            "expression_stability"  : round(stability,     3),
            "confidence"            : conf,
        }

    def reset(self) -> None:
        self._mar_buf.clear()
        self._smile_buf.clear()
        self._tension_buf.clear()
        self._jaw_buf.clear()
        self._ts_buf.clear()
        self._initialized    = False
        self._yawn_start_ms  = None

    # ── Private ────────────────────────────────────────────────────────────────

    def _detect_speaking(self) -> tuple[float, float]:
        """
        Detect speech from MAR oscillation frequency.

        Speech produces characteristic 2–8 Hz oscillations in jaw opening.
        Returns (probability, dominant_hz).
        """
        if len(self._mar_buf) < 16:
            return 0.0, 0.0

        arr = np.array(self._mar_buf, dtype=np.float32)
        amp = float(arr.max() - arr.min())

        if amp < _SPEAK_MAR_AMP:
            return 0.0, 0.0

        freq = peak_frequency_hz(arr, self.fps)

        in_range = _SPEAK_FREQ_LO_HZ <= freq <= _SPEAK_FREQ_HI_HZ
        if not in_range:
            return 0.0, float(freq)

        # Probability scales with amplitude
        prob = clamp(amp / 0.30, 0.0, 1.0)
        return float(prob), float(freq)

    def _update_yawn(self, jaw_drop: float, ts: float) -> tuple[bool, float]:
        """
        State machine: detect yawn as sustained wide jaw opening.
        Returns (is_yawning, duration_ms).
        """
        if jaw_drop >= _YAWN_JAW_THRESH:
            if self._yawn_start_ms is None:
                self._yawn_start_ms = ts
            dur = ts - self._yawn_start_ms
            yawning = _YAWN_MIN_DUR_MS <= dur <= _YAWN_MAX_DUR_MS
            return yawning, float(dur)
        else:
            self._yawn_start_ms = None
            return False, 0.0

    def _trend(self, buf: deque) -> float:
        """Linear slope over the trend window. Positive = rising."""
        n = min(len(buf), _TREND_WINDOW)
        if n < 4:
            return 0.0
        arr = np.array(list(buf)[-n:], dtype=np.float32)
        return float(rolling_slope(arr))

    def _expression_stability(self) -> float:
        """
        1 - normalised std of MAR over recent history.
        High std = rapidly changing mouth = low stability.
        """
        if len(self._mar_buf) < 4:
            return 1.0
        arr = np.array(self._mar_buf, dtype=np.float32)
        std = float(arr.std())
        # std > 0.3 → very unstable
        return clamp(1.0 - std / 0.30, 0.0, 1.0)
