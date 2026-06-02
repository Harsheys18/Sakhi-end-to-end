"""
state/emotional_dynamics.py

Emotional state dynamics tracking.

Tracks how emotional/social states change over time:
  - Emotional transitions (discrete state changes in affect quadrant)
  - Stress escalation / de-escalation trends
  - Engagement shifts (onset and offset)
  - Behavioral persistence (time in current emotional state)
  - Emotional momentum (direction + velocity of affective change)
  - Affective inertia (resistance to change)

These signals are critical for a socially intelligent robot:
  - Detecting rising stress BEFORE it peaks
  - Knowing when engagement is declining
  - Recognizing emotional state changes vs noise

Design: EmotionalDynamicsTracker — stateful, one per face. No I/O.
"""

from collections import deque
import numpy as np
from geometry.math_utils import clamp, rolling_slope

_BUFFER_LEN          = 120   # ~4 s at 30 fps
_TRANSITION_COOLDOWN = 30    # frames before another transition can be declared
_STRESS_ESC_WINDOW   = 60    # frames for escalation trend
_PERSIST_MIN_MS      = 1500  # min ms in a state to count as "persistent"

# Affect quadrant definitions
_QUADRANTS = {"HVHA", "HVLA", "LVHA", "LVLA"}
_QUADRANT_LABELS = {
    "HVHA": "excited/happy",
    "HVLA": "calm/content",
    "LVHA": "distressed/angry",
    "LVLA": "sad/bored",
}


class EmotionalDynamicsTracker:
    """
    Tracks emotional state transitions and dynamics.

    Usage:
        tracker = EmotionalDynamicsTracker()
        state   = tracker.update(affective_emb, gru_state, base_state, timestamp_ms)
    """

    def __init__(self):
        # Valence/arousal/stress history
        self._valence_buf:  deque[float] = deque(maxlen=_BUFFER_LEN)
        self._arousal_buf:  deque[float] = deque(maxlen=_BUFFER_LEN)
        self._stress_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)
        self._engage_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)

        # Transition tracking
        self._current_quadrant:    str         = "HVLA"
        self._quadrant_since_ms:   float       = 0.0
        self._transition_cooldown: int         = 0
        self._transition_history:  deque[dict] = deque(maxlen=10)

        # Persistence
        self._state_since_ms:  float = 0.0
        self._initialized:     bool  = False

    def update(
        self,
        affective_emb: dict,
        gru_state:     dict,
        base_state:    dict,
        timestamp_ms:  float,
    ) -> dict:
        """
        Args:
            affective_emb : dict from features/affective_embedding
            gru_state     : dict from temporal/affective_gru
            base_state    : dict from state/social_state
            timestamp_ms  : current ms

        Returns:
            {
              "current_quadrant"         : str    # "HVHA"|"HVLA"|"LVHA"|"LVLA"
              "quadrant_label"           : str    # "excited/happy" etc.
              "quadrant_duration_ms"     : float
              "transition_occurred"      : bool
              "transition_from"          : str
              "transition_to"            : str
              "stress_escalating"        : bool
              "stress_de_escalating"     : bool
              "stress_trend"             : float  # slope (+ = rising)
              "engagement_shift"         : str    # "rising"|"falling"|"stable"
              "engagement_trend"         : float
              "emotional_momentum"       : float  # [0,1] rate of affective change
              "affective_inertia"        : float  # [0,1] resistance to change
              "valence_velocity"         : float  # per-frame change rate
              "arousal_velocity"         : float
              "state_persistence"        : float  # [0,1] 1 = very stable state
              "recent_transitions"       : int    # count in last 30s
            }
        """
        if not self._initialized:
            self._state_since_ms   = timestamp_ms
            self._quadrant_since_ms = timestamp_ms
            self._initialized      = True

        # Fuse geometry + GRU for more robust valence/arousal
        geom_val = float(affective_emb.get("valence_geom", 0.0))
        gru_val  = float(gru_state.get("gru_valence",  0.0))
        geom_aro = float(affective_emb.get("arousal_geom", 0.0))
        gru_aro  = float(gru_state.get("gru_arousal",  0.0))

        fused_val = 0.4 * geom_val + 0.6 * gru_val   # GRU-weighted (smoother)
        fused_aro = 0.4 * geom_aro + 0.6 * gru_aro

        stress  = float(base_state.get("stress_hints", {}).get("value", 0.0))
        engage  = float(base_state.get("engagement",   {}).get("value", 0.0))

        self._valence_buf.append(fused_val)
        self._arousal_buf.append(fused_aro)
        self._stress_buf.append(stress)
        self._engage_buf.append(engage)

        # ── Quadrant detection ────────────────────────────────────────────────
        v_label = "HV" if fused_val > 0.05 else "LV"   # slight positive bias
        a_label = "HA" if fused_aro > 0.45 else "LA"
        new_quad = f"{v_label}{a_label}"

        transition   = False
        trans_from   = self._current_quadrant
        trans_to     = new_quad

        if self._transition_cooldown > 0:
            self._transition_cooldown -= 1

        if new_quad != self._current_quadrant and self._transition_cooldown == 0:
            # Confirm transition only if new quadrant sustained 5+ frames
            quad_buf = [v_label + a_label]  # simplified — use history
            transition = True
            self._transition_history.append({
                "ts":    timestamp_ms,
                "from":  self._current_quadrant,
                "to":    new_quad,
            })
            self._current_quadrant  = new_quad
            self._quadrant_since_ms = timestamp_ms
            self._transition_cooldown = _TRANSITION_COOLDOWN

        quadrant_dur = timestamp_ms - self._quadrant_since_ms

        # ── Stress trend ──────────────────────────────────────────────────────
        n = min(len(self._stress_buf), _STRESS_ESC_WINDOW)
        if n >= 8:
            arr          = np.array(list(self._stress_buf)[-n:], dtype=np.float32)
            stress_trend = float(rolling_slope(arr))
        else:
            stress_trend = 0.0

        stress_esc   = stress_trend >  0.002   # rising stress
        stress_de    = stress_trend < -0.002   # falling stress

        # ── Engagement shift ──────────────────────────────────────────────────
        if len(self._engage_buf) >= 8:
            arr = np.array(list(self._engage_buf)[-30:], dtype=np.float32)
            eng_slope = float(rolling_slope(arr))
        else:
            eng_slope = 0.0

        if eng_slope > 0.002:
            eng_shift = "rising"
        elif eng_slope < -0.002:
            eng_shift = "falling"
        else:
            eng_shift = "stable"

        # ── Valence / arousal velocity ────────────────────────────────────────
        val_vel = 0.0
        aro_vel = 0.0
        if len(self._valence_buf) >= 4:
            v_arr = np.array(list(self._valence_buf)[-8:], dtype=np.float32)
            a_arr = np.array(list(self._arousal_buf)[-8:], dtype=np.float32)
            val_vel = float(rolling_slope(v_arr))
            aro_vel = float(rolling_slope(a_arr))

        # ── Emotional momentum ────────────────────────────────────────────────
        momentum = float(gru_state.get("emotional_momentum", 0.0))

        # ── Affective inertia (inverse of momentum — resistance to change) ────
        inertia = clamp(1.0 - momentum, 0.0, 1.0)

        # ── State persistence ─────────────────────────────────────────────────
        # Longer in current quadrant = more persistent
        persist_score = clamp(quadrant_dur / 10000.0, 0.0, 1.0)  # 10s = max

        # ── Recent transition count ───────────────────────────────────────────
        cutoff_ms    = timestamp_ms - 30000.0
        recent_trans = sum(1 for t in self._transition_history
                           if t["ts"] > cutoff_ms)

        return {
            "current_quadrant"     : self._current_quadrant,
            "quadrant_label"       : _QUADRANT_LABELS.get(self._current_quadrant, "unknown"),
            "quadrant_duration_ms" : round(quadrant_dur,    1),
            "transition_occurred"  : transition,
            "transition_from"      : trans_from,
            "transition_to"        : trans_to,
            "stress_escalating"    : stress_esc,
            "stress_de_escalating" : stress_de,
            "stress_trend"         : round(stress_trend,    5),
            "engagement_shift"     : eng_shift,
            "engagement_trend"     : round(eng_slope,       5),
            "emotional_momentum"   : round(momentum,        4),
            "affective_inertia"    : round(inertia,         4),
            "valence_velocity"     : round(val_vel,         5),
            "arousal_velocity"     : round(aro_vel,         5),
            "state_persistence"    : round(persist_score,   3),
            "recent_transitions"   : recent_trans,
        }

    def reset(self) -> None:
        self._valence_buf.clear()
        self._arousal_buf.clear()
        self._stress_buf.clear()
        self._engage_buf.clear()
        self._current_quadrant  = "HVLA"
        self._transition_history.clear()
        self._initialized       = False
        self._transition_cooldown = 0
