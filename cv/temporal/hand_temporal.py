"""
temporal/hand_temporal.py

Temporal hand gesture tracking.

Tracks gesture events over a rolling window to compute:
  gesture_frequency         : gestures per minute
  hand_to_face_frequency    : hand-to-face events per minute
  gesture_variety           : number of distinct gesture types seen
  dominant_gesture          : most-frequent gesture in window

Design: all state in HandTemporalTracker. No I/O.
"""

from collections import deque, Counter
import numpy as np
from geometry.math_utils import clamp

_WINDOW_SECONDS   = 30.0
_GESTURE_DEBOUNCE = 500    # ms — ignore same gesture within this window
_FACE_TOUCH_THRESH= 0.6    # hand_to_face_score above this = event


class HandTemporalTracker:
    """
    Stateful tracker for hand gesture frequency.

    Usage:
        tracker = HandTemporalTracker()
        state   = tracker.update(hand_feats, timestamp_ms)
    """

    def __init__(self, window_seconds: float = _WINDOW_SECONDS):
        self.window_ms = window_seconds * 1000.0

        self._gesture_times:   deque[tuple[float, str]] = deque()  # (ts, label)
        self._face_touch_times: deque[float] = deque()

        self._last_gesture:    str   = "none"
        self._last_gesture_ts: float = 0.0
        self._in_face_touch:   bool  = False
        self._face_touch_start:float = 0.0

    def update(self, hand_feats: dict, timestamp_ms: float) -> dict:
        """
        Args:
            hand_feats   : dict from features/hand_features.extract_hand_features()
            timestamp_ms : current time ms

        Returns:
            {
              "gesture_frequency"      : float  # gestures/min
              "hand_to_face_frequency" : float  # events/min
              "gesture_variety"        : int    # distinct gestures in window
              "dominant_gesture"       : str
            }
        """
        gesture  = hand_feats.get("gesture_primitive",   "none")
        hf_score = hand_feats.get("hand_to_face_score",  0.0)

        # ── Record gesture event (debounced) ──────────────────────────────────
        if (gesture != "none"
                and gesture != self._last_gesture
                and (timestamp_ms - self._last_gesture_ts) > _GESTURE_DEBOUNCE):
            self._gesture_times.append((timestamp_ms, gesture))
            self._last_gesture    = gesture
            self._last_gesture_ts = timestamp_ms

        # ── Record hand-to-face event ─────────────────────────────────────────
        if hf_score > _FACE_TOUCH_THRESH:
            if not self._in_face_touch:
                self._face_touch_times.append(timestamp_ms)
                self._in_face_touch   = True
                self._face_touch_start = timestamp_ms
        else:
            self._in_face_touch = False

        # ── Expire old events ─────────────────────────────────────────────────
        cutoff = timestamp_ms - self.window_ms
        while self._gesture_times   and self._gesture_times[0][0]   < cutoff:
            self._gesture_times.popleft()
        while self._face_touch_times and self._face_touch_times[0] < cutoff:
            self._face_touch_times.popleft()

        # ── Metrics ───────────────────────────────────────────────────────────
        elapsed_s  = min(self.window_ms, timestamp_ms) / 1000.0
        elapsed_s  = max(elapsed_s, 1.0)

        gesture_n   = len(self._gesture_times)
        face_n      = len(self._face_touch_times)

        gesture_freq  = (gesture_n / elapsed_s) * 60.0
        face_freq     = (face_n    / elapsed_s) * 60.0

        labels        = [g for _, g in self._gesture_times]
        variety       = len(set(labels))
        dominant      = Counter(labels).most_common(1)[0][0] if labels else "none"

        return {
            "gesture_frequency"      : round(gesture_freq, 2),
            "hand_to_face_frequency" : round(face_freq,    2),
            "gesture_variety"        : variety,
            "dominant_gesture"       : dominant,
        }

    def reset(self) -> None:
        self._gesture_times.clear()
        self._face_touch_times.clear()
        self._in_face_touch    = False
        self._last_gesture     = "none"
        self._last_gesture_ts  = 0.0
