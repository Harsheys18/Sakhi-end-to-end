"""
temporal/gaze_temporal.py

Temporal gaze tracking over a rolling window.

Consumes per-frame output of features/gaze_features.extract_gaze_features()
and produces:

  - Smoothed gaze_x / gaze_y (EMA)
  - Gaze volatility (instability of gaze direction)
  - Fixation detection (gaze held in one region)
  - Fixation dwell time (ms)
  - Eye contact duration (sustained centered gaze)
  - Gaze pattern classification ("fixating"|"scanning"|"avoidant")

Design: all state in GazeTracker. No globals. No I/O.
"""

from collections import deque
import numpy as np
from geometry.math_utils import clamp, rolling_slope

# ── Configuration ─────────────────────────────────────────────────────────────

_EMA_ALPHA           = 0.30   # gaze smoothing (higher = more responsive)
_BUFFER_LEN          = 60     # history depth (~2 s at 30 fps)

# Fixation: gaze held within a radius for a minimum duration
_FIXATION_RADIUS     = 0.20   # gaze units (same space as avg_gaze_x/y ∈ [-1,1])
_FIXATION_MIN_MS     = 200    # minimum dwell to count as fixation

# Eye contact: centered gaze held long enough
_EYE_CONTACT_SCORE_THRESH = 0.75   # eye_contact_score threshold
_EYE_CONTACT_MIN_MS       = 500    # ms to declare sustained eye contact

# Gaze volatility: std of gaze positions over the buffer
_VOLATILITY_STABLE    = 0.05   # below this → fixating
_VOLATILITY_SCANNING  = 0.25   # above this → scanning


class GazeTracker:
    """
    Stateful temporal gaze tracker for one face.

    Usage:
        tracker = GazeTracker()
        state   = tracker.update(gaze_feats, timestamp_ms)
    """

    def __init__(self):
        self._gaze_x: float = 0.0
        self._gaze_y: float = 0.0
        self._initialized    = False

        # Rolling gaze history
        self._gx_buf: deque[float] = deque(maxlen=_BUFFER_LEN)
        self._gy_buf: deque[float] = deque(maxlen=_BUFFER_LEN)
        self._ec_buf: deque[float] = deque(maxlen=_BUFFER_LEN)  # eye_contact_score
        self._ts_buf: deque[float] = deque(maxlen=_BUFFER_LEN)

        # Fixation state
        self._fix_anchor_x:  float = 0.0
        self._fix_anchor_y:  float = 0.0
        self._fix_start_ms:  float = 0.0
        self._in_fixation:   bool  = False

        # Eye contact streak
        self._ec_start_ms:   float | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, gaze_feats: dict, timestamp_ms: float) -> dict:
        """
        Ingest one frame's gaze features and return temporal state.

        Args:
            gaze_feats   : dict from features/gaze_features.extract_gaze_features()
            timestamp_ms : current time in ms

        Returns:
            {
              "gaze_x"                   : float   # EMA-smoothed [-1,1]
              "gaze_y"                   : float   # EMA-smoothed [-1,1]
              "gaze_direction"           : str
              "gaze_volatility"          : float   # [0,1] 0=still, 1=erratic
              "fixation_active"          : bool
              "fixation_dwell_ms"        : float
              "gaze_pattern"             : str     # "fixating"|"scanning"|"avoidant"
              "eye_contact_score"        : float   # current frame score
              "eye_contact_sustained_ms" : float   # ms of continuous eye contact
              "eye_contact_rate"         : float   # fraction of window with eye contact
              "confidence"               : float
            }
        """
        raw_gx = gaze_feats.get("avg_gaze_x",        0.0)
        raw_gy = gaze_feats.get("avg_gaze_y",        0.0)
        ec_sc  = gaze_feats.get("eye_contact_score", 0.0)
        conf   = gaze_feats.get("confidence",        0.0)

        # ── EMA smoothing ────────────────────────────────────────────────────
        if not self._initialized:
            self._gaze_x     = raw_gx
            self._gaze_y     = raw_gy
            self._initialized = True
        else:
            a = _EMA_ALPHA
            self._gaze_x = a * raw_gx + (1 - a) * self._gaze_x
            self._gaze_y = a * raw_gy + (1 - a) * self._gaze_y

        gx, gy = self._gaze_x, self._gaze_y

        # ── Buffer update ────────────────────────────────────────────────────
        self._gx_buf.append(gx)
        self._gy_buf.append(gy)
        self._ec_buf.append(ec_sc)
        self._ts_buf.append(timestamp_ms)

        # ── Gaze volatility ──────────────────────────────────────────────────
        volatility = self._compute_volatility()

        # ── Fixation detection ───────────────────────────────────────────────
        fix_active, fix_dwell = self._update_fixation(gx, gy, timestamp_ms)

        # ── Eye contact sustained ────────────────────────────────────────────
        ec_sustained = self._update_eye_contact(ec_sc, timestamp_ms)

        # Fraction of window with eye contact
        ec_rate = float(
            sum(s > _EYE_CONTACT_SCORE_THRESH for s in self._ec_buf)
            / max(len(self._ec_buf), 1)
        )

        # ── Gaze pattern ─────────────────────────────────────────────────────
        pattern = self._classify_pattern(volatility, ec_rate)

        # Direction from smoothed gaze
        direction = gaze_feats.get("gaze_direction", "unknown")

        return {
            "gaze_x"                   : round(gx, 3),
            "gaze_y"                   : round(gy, 3),
            "gaze_direction"           : direction,
            "gaze_volatility"          : round(volatility, 3),
            "fixation_active"          : fix_active,
            "fixation_dwell_ms"        : round(fix_dwell, 1),
            "gaze_pattern"             : pattern,
            "eye_contact_score"        : round(ec_sc, 3),
            "eye_contact_sustained_ms" : round(ec_sustained, 1),
            "eye_contact_rate"         : round(ec_rate, 3),
            "confidence"               : conf,
        }

    def reset(self) -> None:
        self._gx_buf.clear()
        self._gy_buf.clear()
        self._ec_buf.clear()
        self._ts_buf.clear()
        self._initialized    = False
        self._in_fixation    = False
        self._ec_start_ms    = None

    # ── Private ────────────────────────────────────────────────────────────────

    def _compute_volatility(self) -> float:
        """
        Joint std of gaze_x and gaze_y over the rolling buffer.
        Normalised to [0, 1].
        """
        if len(self._gx_buf) < 4:
            return 0.0
        gx = np.array(self._gx_buf, dtype=np.float32)
        gy = np.array(self._gy_buf, dtype=np.float32)
        std = float((gx.std() + gy.std()) / 2.0)
        # std > 0.4 → very erratic (scanning/distracted)
        return clamp(std / 0.40, 0.0, 1.0)

    def _update_fixation(
        self, gx: float, gy: float, ts: float
    ) -> tuple[bool, float]:
        """
        Update fixation state machine.

        A fixation starts when gaze enters a _FIXATION_RADIUS region and
        holds for ≥ _FIXATION_MIN_MS.

        Returns:
            (is_fixating, dwell_ms)
        """
        if not self._in_fixation:
            # Start new candidate fixation
            self._fix_anchor_x = gx
            self._fix_anchor_y = gy
            self._fix_start_ms = ts
            self._in_fixation  = True
            return False, 0.0

        # Check if gaze is still within radius of the anchor
        dist = ((gx - self._fix_anchor_x) ** 2 + (gy - self._fix_anchor_y) ** 2) ** 0.5
        if dist > _FIXATION_RADIUS:
            # Saccade occurred — restart candidate
            self._fix_anchor_x = gx
            self._fix_anchor_y = gy
            self._fix_start_ms = ts
            return False, 0.0

        dwell = ts - self._fix_start_ms
        is_fix = dwell >= _FIXATION_MIN_MS
        return is_fix, float(dwell)

    def _update_eye_contact(self, ec_score: float, ts: float) -> float:
        """
        Track sustained eye contact duration.
        Returns ms of current continuous eye contact streak.
        """
        if ec_score >= _EYE_CONTACT_SCORE_THRESH:
            if self._ec_start_ms is None:
                self._ec_start_ms = ts
            return float(ts - self._ec_start_ms)
        else:
            self._ec_start_ms = None
            return 0.0

    def _classify_pattern(self, volatility: float, ec_rate: float) -> str:
        """
        Classify overall gaze behaviour into one of three patterns.

        "fixating"  → low volatility, gaze held in one place
        "scanning"  → high volatility, actively looking around
        "avoidant"  → low eye contact rate, gaze directed away from camera
        """
        if ec_rate < 0.20 and volatility < _VOLATILITY_SCANNING:
            return "avoidant"
        if volatility >= _VOLATILITY_SCANNING:
            return "scanning"
        return "fixating"
