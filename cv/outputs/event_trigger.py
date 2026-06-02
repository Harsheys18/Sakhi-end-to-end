"""
outputs/event_trigger.py

Event-driven LLM emission controller.

Problem with fixed-rate emission:
  - Spams LLM with identical state every 200ms
  - Wastes tokens when nothing is changing
  - Misses rapid critical events (stress spike between ticks)
  - No concept of "something important just happened"

Solution:
  EventTrigger decides WHEN to emit based on:
    1. Significant state change  (delta above threshold)
    2. Critical event            (stress spike, distress onset, mood shift)
    3. Minimum heartbeat         (emit at least every N seconds to confirm alive)
    4. Support urgency change    (urgency level changes → always emit immediately)

Also:
  - Tracks which dimensions changed most → tells LLM what to focus on
  - Computes change_summary for efficient LLM prompt injection
  - Distinguishes gradual drift vs sudden shift

Design: stateful (EmissionController). One per session.
"""

from collections import deque
import time
from geometry.math_utils import clamp

# Thresholds for "significant change" per tracked dimension
_THRESHOLDS = {
    "distress_level"       : 0.08,
    "engagement"           : 0.10,
    "valence_geom"         : 0.10,
    "arousal_geom"         : 0.10,
    "gru_valence"          : 0.08,
    "openness_to_support"  : 0.08,
    "overwhelm_probability": 0.07,
    "emotional_suppression": 0.10,
    "emotional_withdrawal" : 0.08,
    "comfort_seeking"      : 0.08,
    "social_comfort"       : 0.10,
    "rapport_signal"       : 0.10,
}

# Always emit immediately when these events occur
_CRITICAL_EVENTS = {
    "stress_escalating",
    "transition_occurred",
    "overwhelm_probability",   # if > 0.6
    "support_urgency",         # if changed
}

_HEARTBEAT_S     = 8.0    # emit at least every 8 seconds even if stable
_MIN_INTERVAL_S  = 0.5    # never emit faster than 2Hz regardless


class EmissionController:
    """
    Decides when the LLM should receive a new state update.

    Usage:
        ctrl = EmissionController()

        # In main loop:
        should_emit, reason = ctrl.should_emit(full_state)
        if should_emit:
            print(format_llm_context(full_state))
            ctrl.mark_emitted(full_state)
    """

    def __init__(
        self,
        heartbeat_s:    float = _HEARTBEAT_S,
        min_interval_s: float = _MIN_INTERVAL_S,
    ):
        self.heartbeat_s    = heartbeat_s
        self.min_interval_s = min_interval_s

        self._last_emit_ts:  float = 0.0
        self._last_state:    dict  = {}
        self._last_urgency:  str   = "none"

    def should_emit(self, state: dict) -> tuple[bool, str]:
        """
        Decide whether to emit now.

        Args:
            state : full merged state dict

        Returns:
            (should_emit, reason_string)
            reason is a short label for logging/debugging.
        """
        now = time.perf_counter()
        elapsed = now - self._last_emit_ts

        # ── Rate limiter ──────────────────────────────────────────────────────
        if elapsed < self.min_interval_s:
            return False, "rate_limited"

        # ── First emission ────────────────────────────────────────────────────
        if not self._last_state:
            return True, "first_emission"

        # ── Support urgency change → always immediate ─────────────────────────
        cur_urgency = state.get("support_urgency", "none")
        if cur_urgency != self._last_urgency:
            return True, f"urgency_changed:{self._last_urgency}→{cur_urgency}"

        # ── Critical events ───────────────────────────────────────────────────
        if state.get("stress_escalating") and not self._last_state.get("stress_escalating"):
            return True, "stress_escalation_onset"

        if state.get("transition_occurred"):
            return True, f"affect_transition:{state.get('transition_from','')}→{state.get('transition_to','')}"

        overwhelm = float(state.get("overwhelm_probability", 0.0))
        prev_ov   = float(self._last_state.get("overwhelm_probability", 0.0))
        if overwhelm > 0.60 and prev_ov <= 0.60:
            return True, "overwhelm_threshold_crossed"

        # ── Significant state change ──────────────────────────────────────────
        changed_dims = self._compute_changes(state)
        if changed_dims:
            top = sorted(changed_dims, key=lambda x: x[1], reverse=True)
            return True, f"state_change:{top[0][0]}Δ{top[0][1]:.2f}"

        # ── Heartbeat ─────────────────────────────────────────────────────────
        if elapsed >= self.heartbeat_s:
            return True, "heartbeat"

        return False, "stable"

    def mark_emitted(self, state: dict) -> None:
        """Call after emitting to update internal state."""
        self._last_emit_ts = time.perf_counter()
        self._last_state   = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in state.items()
            if not isinstance(v, (dict, list))
        }
        self._last_urgency = state.get("support_urgency", "none")

    def get_changed_dims(self, state: dict) -> list[tuple[str, float]]:
        """Return list of (dim_name, delta) sorted by magnitude."""
        return sorted(self._compute_changes(state), key=lambda x: x[1], reverse=True)

    def reset(self) -> None:
        self._last_emit_ts = 0.0
        self._last_state   = {}
        self._last_urgency = "none"

    def _compute_changes(self, state: dict) -> list[tuple[str, float]]:
        """Find dimensions that changed beyond threshold."""
        changed = []
        for dim, thresh in _THRESHOLDS.items():
            cur  = self._get_val(state,            dim)
            prev = self._get_val(self._last_state, dim)
            if cur is None or prev is None:
                continue
            delta = abs(cur - prev)
            if delta >= thresh:
                changed.append((dim, delta))
        return changed

    @staticmethod
    def _get_val(state: dict, key: str):
        v = state.get(key)
        if isinstance(v, dict):
            return v.get("value")
        if isinstance(v, (int, float)):
            return float(v)
        return None
