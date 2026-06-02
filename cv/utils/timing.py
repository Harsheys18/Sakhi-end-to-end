"""
utils/timing.py

Lightweight timing utilities for the realtime pipeline.

Provides:
  FPSCounter  : rolling-window frame rate estimator
  RateGate    : throttle a code path to at most N Hz
  PipelineTimer: per-stage latency profiler (zero overhead when disabled)

Design: no dependencies outside stdlib + numpy. No I/O side effects.
"""

import time
from collections import deque
import numpy as np


class FPSCounter:
    """
    Estimates realtime frame rate over a rolling window.

    Usage:
        fps_counter = FPSCounter(window=60)

        while True:
            fps_counter.tick()
            fps = fps_counter.fps
    """

    def __init__(self, window: int = 60):
        """
        Args:
            window : number of recent frames to average over.
        """
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Record the current frame timestamp."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Estimated frames per second. Returns 0.0 if fewer than 2 frames."""
        n = len(self._timestamps)
        if n < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed < 1e-6:
            return 0.0
        return (n - 1) / elapsed

    @property
    def frame_time_ms(self) -> float:
        """Mean milliseconds per frame over the window."""
        f = self.fps
        return (1000.0 / f) if f > 0 else 0.0

    def reset(self) -> None:
        self._timestamps.clear()


class RateGate:
    """
    Throttle a code path to at most `max_hz` calls per second.

    Usage:
        gate = RateGate(max_hz=5.0)

        while True:
            if gate.allow():
                send_state_to_llm(state)
    """

    def __init__(self, max_hz: float):
        """
        Args:
            max_hz : maximum allowed call rate.
        """
        self._min_interval = 1.0 / max_hz if max_hz > 0 else 0.0
        self._last_ts: float = 0.0

    def allow(self) -> bool:
        """Returns True if the call is within the allowed rate."""
        now = time.perf_counter()
        if now - self._last_ts >= self._min_interval:
            self._last_ts = now
            return True
        return False

    def reset(self) -> None:
        self._last_ts = 0.0


class PipelineTimer:
    """
    Per-stage latency profiler.

    Measures wall-clock time of named pipeline stages. Results are
    accumulated into a rolling deque per stage.

    When `enabled=False` (production default), all calls are near-zero
    overhead no-ops.

    Usage:
        timer = PipelineTimer(enabled=True, window=30)

        timer.start("mediapipe")
        run_mediapipe(frame)
        timer.end("mediapipe")

        timer.start("features")
        extract_features(landmarks)
        timer.end("features")

        report = timer.report()
        # → {"mediapipe": {"mean_ms": 4.2, "max_ms": 7.1}, ...}
    """

    def __init__(self, enabled: bool = False, window: int = 30):
        """
        Args:
            enabled : if False, timer is a no-op (default for production).
            window  : rolling window depth per stage.
        """
        self.enabled = enabled
        self._window  = window
        self._starts:   dict[str, float]          = {}
        self._buffers:  dict[str, deque[float]]   = {}

    def start(self, stage: str) -> None:
        if not self.enabled:
            return
        self._starts[stage] = time.perf_counter()

    def end(self, stage: str) -> None:
        if not self.enabled:
            return
        t0 = self._starts.get(stage)
        if t0 is None:
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if stage not in self._buffers:
            self._buffers[stage] = deque(maxlen=self._window)
        self._buffers[stage].append(elapsed_ms)

    def report(self) -> dict[str, dict]:
        """
        Returns per-stage latency statistics.

        Returns:
            {
              "stage_name": {
                "mean_ms": float,
                "max_ms":  float,
                "min_ms":  float,
              },
              ...
            }
        """
        if not self.enabled:
            return {}
        out = {}
        for stage, buf in self._buffers.items():
            if not buf:
                continue
            arr = np.array(buf, dtype=np.float32)
            out[stage] = {
                "mean_ms": round(float(arr.mean()), 2),
                "max_ms":  round(float(arr.max()),  2),
                "min_ms":  round(float(arr.min()),  2),
            }
        return out

    def reset(self) -> None:
        self._starts.clear()
        self._buffers.clear()
