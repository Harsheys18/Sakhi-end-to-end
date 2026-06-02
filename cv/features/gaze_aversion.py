"""
features/gaze_aversion.py

Gaze aversion frequency: how often the subject breaks eye contact
and looks away per minute.

Consumes gaze_temporal output (already has eye_contact_score) and
counts transitions from "contact" to "aversion" over a rolling window.

This is a TEMPORAL feature that needs rolling state, so it lives in
features/ but is stateful (like the blink tracker).

Output:
  gaze_aversion_frequency : float  — aversions per minute
  current_aversion        : bool   — currently in an aversion event
  aversion_duration_ms    : float  — ms of current aversion streak
  mean_aversion_dur_ms    : float  — mean duration of past aversions

Usage:
    tracker = GazeAversionTracker()
    result  = tracker.update(gaze_temporal_dict, timestamp_ms)
"""

from collections import deque
import numpy as np

_WINDOW_SECONDS        = 60.0    # rolling window for frequency count
_CONTACT_THRESH        = 0.55    # eye_contact_score above = "in contact"
_AVERSION_THRESH       = 0.35    # below = "in aversion"
_MIN_AVERSION_DUR_MS   = 300     # ignore aversions shorter than this (noise)


class GazeAversionTracker:
    """
    Stateful tracker for gaze aversion events.

    Usage:
        tracker = GazeAversionTracker()
        result  = tracker.update(gaze_temporal, timestamp_ms)
    """

    def __init__(self, window_seconds: float = _WINDOW_SECONDS):
        self.window_ms = window_seconds * 1000.0

        # Timestamps of confirmed aversion events within the rolling window
        self._aversion_times: deque[float] = deque()
        # Completed aversion durations (ms)
        self._aversion_durations: deque[float] = deque(maxlen=30)

        # State machine
        self._in_contact:      bool         = True   # assume starting in contact
        self._aversion_start:  float | None = None

    def update(self, gaze_temporal: dict, timestamp_ms: float) -> dict:
        """
        Args:
            gaze_temporal : dict from temporal/gaze_temporal.GazeTracker.update()
            timestamp_ms  : current time in ms

        Returns:
            {
              "gaze_aversion_frequency": float   # aversions per minute
              "current_aversion"       : bool
              "aversion_duration_ms"   : float
              "mean_aversion_dur_ms"   : float
            }
        """
        ec_score = gaze_temporal.get("eye_contact_score", 0.0)

        # ── Hysteresis state machine ─────────────────────────────────────────
        if self._in_contact:
            if ec_score < _AVERSION_THRESH:
                # Transition: contact → aversion
                self._in_contact     = False
                self._aversion_start = timestamp_ms
        else:
            if ec_score > _CONTACT_THRESH:
                # Transition: aversion → contact
                if self._aversion_start is not None:
                    dur = timestamp_ms - self._aversion_start
                    if dur >= _MIN_AVERSION_DUR_MS:
                        self._aversion_times.append(timestamp_ms)
                        self._aversion_durations.append(dur)
                self._in_contact     = True
                self._aversion_start = None

        # ── Expire old events outside the rolling window ──────────────────────
        cutoff = timestamp_ms - self.window_ms
        while self._aversion_times and self._aversion_times[0] < cutoff:
            self._aversion_times.popleft()

        # ── Metrics ───────────────────────────────────────────────────────────
        n = len(self._aversion_times)
        if n >= 2:
            elapsed_s = (self._aversion_times[-1] - self._aversion_times[0]) / 1000.0
            freq = (n / elapsed_s * 60.0) if elapsed_s > 0.5 else 0.0
        elif n == 1:
            freq = 60.0 / (self.window_ms / 1000.0)
        else:
            freq = 0.0

        mean_dur = float(np.mean(self._aversion_durations)) if self._aversion_durations else 0.0
        current_aversion = not self._in_contact
        aversion_dur = (timestamp_ms - self._aversion_start
                        if (current_aversion and self._aversion_start) else 0.0)

        return {
            "gaze_aversion_frequency": round(freq,         2),
            "current_aversion"       : current_aversion,
            "aversion_duration_ms"   : round(aversion_dur, 1),
            "mean_aversion_dur_ms"   : round(mean_dur,     1),
        }

    def reset(self) -> None:
        self._aversion_times.clear()
        self._aversion_durations.clear()
        self._in_contact      = True
        self._aversion_start  = None
