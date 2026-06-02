"""
temporal/social_temporal.py

Temporal tracking of derived social-state dimensions.

Consumes the output of state/social_state.estimate_social_state() on each
frame and builds rolling trend and sustained-state signals that cannot be
computed in a single frame.

Tracks:
  engagement_trend         : slope of engagement over window
  attention_trend          : slope of attention over window
  gaze_trend               : slope of eye_contact_rate over window
  emotion_transition_rate  : how rapidly valence/arousal changes
  emotional_stability      : inverse of emotion_transition_rate
  sustained_attention      : ms of continuous high attention
  sustained_disengagement  : ms of continuous low engagement
  attention_stability      : rolling std of attention_score
  interaction_stability    : rolling std of interaction_willingness
  prolonged_downward_gaze  : ms of continuous downward gaze

Design: all state in SocialTemporalTracker. No globals. No I/O.
"""

from collections import deque
import numpy as np
from geometry.math_utils import clamp, rolling_slope

_BUFFER_LEN             = 150    # ~5 s at 30 fps
_HIGH_ATTENTION_THRESH  = 0.65
_LOW_ENGAGEMENT_THRESH  = 0.30
_SUSTAINED_MIN_MS       = 2000   # ms to declare "sustained" state
_DOWNWARD_GAZE_THRESH   = 0.30   # gaze_y > this = looking down
_DOWNWARD_GAZE_MIN_MS   = 1500


class SocialTemporalTracker:
    """
    Tracks rolling trends of social state dimensions.

    Usage:
        tracker = SocialTemporalTracker()
        trends  = tracker.update(social_state, gaze_temporal, timestamp_ms)
    """

    def __init__(self):
        self._eng_buf:  deque[float] = deque(maxlen=_BUFFER_LEN)
        self._att_buf:  deque[float] = deque(maxlen=_BUFFER_LEN)
        self._ec_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)   # eye_contact_rate
        self._iw_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)
        self._val_buf:  deque[float] = deque(maxlen=_BUFFER_LEN)
        self._aro_buf:  deque[float] = deque(maxlen=_BUFFER_LEN)
        self._gy_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)   # gaze_y

        # Sustained state timers
        self._high_att_since_ms:  float | None = None
        self._low_eng_since_ms:   float | None = None
        self._down_gaze_since_ms: float | None = None

    def update(
        self,
        social_state: dict,
        gaze_temporal: dict,
        timestamp_ms:  float,
    ) -> dict:
        """
        Args:
            social_state  : dict from state/social_state.estimate_social_state()
            gaze_temporal : dict from temporal/gaze_temporal.GazeTracker.update()
            timestamp_ms  : current time in ms

        Returns:
            {
              "engagement_trend"        : float   # + rising, - falling
              "attention_trend"         : float
              "gaze_trend"              : float   # eye contact rate trend
              "emotion_transition_rate" : float   # [0,1] how fast affect changes
              "emotional_stability"     : float   # [0,1] 1 = stable
              "sustained_attention"     : bool
              "sustained_attention_ms"  : float
              "sustained_disengagement" : bool
              "sustained_disengagement_ms": float
              "attention_stability"     : float   # [0,1] 1 = consistent
              "interaction_stability"   : float   # [0,1] 1 = consistent
              "prolonged_downward_gaze" : bool
              "prolonged_downward_gaze_ms": float
            }
        """
        eng = social_state.get("engagement",  {}).get("value", 0.0)
        att = social_state.get("attention",   {}).get("value", 0.0)
        iw  = social_state.get("interaction_willingness", {}).get("value", 0.0)
        val = social_state.get("valence",     {}).get("value", 0.5)
        aro = social_state.get("arousal",     {}).get("value", 0.0)
        ec  = gaze_temporal.get("eye_contact_rate", 0.0)
        gy  = gaze_temporal.get("gaze_y", 0.0)

        self._eng_buf.append(eng)
        self._att_buf.append(att)
        self._ec_buf.append(ec)
        self._iw_buf.append(iw)
        self._val_buf.append(val)
        self._aro_buf.append(aro)
        self._gy_buf.append(gy)

        # ── Trend slopes ─────────────────────────────────────────────────────
        def _trend(buf):
            if len(buf) < 8:
                return 0.0
            return float(rolling_slope(np.array(buf, dtype=np.float32)))

        eng_trend  = _trend(self._eng_buf)
        att_trend  = _trend(self._att_buf)
        gaze_trend = _trend(self._ec_buf)

        # ── Emotion transition rate ───────────────────────────────────────────
        # How rapidly valence and arousal change frame-to-frame
        if len(self._val_buf) >= 4:
            val_arr = np.array(self._val_buf, dtype=np.float32)
            aro_arr = np.array(self._aro_buf, dtype=np.float32)
            val_diff = float(np.abs(np.diff(val_arr[-30:])).mean()) if len(val_arr) >= 2 else 0.0
            aro_diff = float(np.abs(np.diff(aro_arr[-30:])).mean()) if len(aro_arr) >= 2 else 0.0
            # Normalise: diff > 0.05 per frame = high rate
            etr = clamp((val_diff + aro_diff) / 2.0 / 0.05, 0.0, 1.0)
        else:
            etr = 0.0
        emotional_stability = 1.0 - etr

        # ── Attention / engagement stability ──────────────────────────────────
        def _stability(buf):
            if len(buf) < 4:
                return 1.0
            arr = np.array(buf, dtype=np.float32)
            # std > 0.25 = very unstable
            return clamp(1.0 - float(arr.std()) / 0.25, 0.0, 1.0)

        att_stability = _stability(self._att_buf)
        iw_stability  = _stability(self._iw_buf)

        # ── Sustained attention ───────────────────────────────────────────────
        if att >= _HIGH_ATTENTION_THRESH:
            if self._high_att_since_ms is None:
                self._high_att_since_ms = timestamp_ms
            att_dur = timestamp_ms - self._high_att_since_ms
        else:
            self._high_att_since_ms = None
            att_dur = 0.0
        sustained_attention = att_dur >= _SUSTAINED_MIN_MS

        # ── Sustained disengagement ───────────────────────────────────────────
        if eng <= _LOW_ENGAGEMENT_THRESH:
            if self._low_eng_since_ms is None:
                self._low_eng_since_ms = timestamp_ms
            dis_dur = timestamp_ms - self._low_eng_since_ms
        else:
            self._low_eng_since_ms = None
            dis_dur = 0.0
        sustained_disengagement = dis_dur >= _SUSTAINED_MIN_MS

        # ── Prolonged downward gaze ───────────────────────────────────────────
        if gy >= _DOWNWARD_GAZE_THRESH:
            if self._down_gaze_since_ms is None:
                self._down_gaze_since_ms = timestamp_ms
            dg_dur = timestamp_ms - self._down_gaze_since_ms
        else:
            self._down_gaze_since_ms = None
            dg_dur = 0.0
        prolonged_downward_gaze = dg_dur >= _DOWNWARD_GAZE_MIN_MS

        return {
            "engagement_trend"             : round(eng_trend,  5),
            "attention_trend"              : round(att_trend,  5),
            "gaze_trend"                   : round(gaze_trend, 5),
            "emotion_transition_rate"      : round(etr,        3),
            "emotional_stability"          : round(emotional_stability, 3),
            "sustained_attention"          : sustained_attention,
            "sustained_attention_ms"       : round(att_dur,    1),
            "sustained_disengagement"      : sustained_disengagement,
            "sustained_disengagement_ms"   : round(dis_dur,    1),
            "attention_stability"          : round(att_stability, 3),
            "interaction_stability"        : round(iw_stability,  3),
            "prolonged_downward_gaze"      : prolonged_downward_gaze,
            "prolonged_downward_gaze_ms"   : round(dg_dur,     1),
        }

    def reset(self) -> None:
        self._eng_buf.clear()
        self._att_buf.clear()
        self._ec_buf.clear()
        self._iw_buf.clear()
        self._val_buf.clear()
        self._aro_buf.clear()
        self._gy_buf.clear()
        self._high_att_since_ms   = None
        self._low_eng_since_ms    = None
        self._down_gaze_since_ms  = None
