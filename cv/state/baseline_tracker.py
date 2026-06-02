"""
state/baseline_tracker.py

Per-person within-session baseline estimation.

Problem: population-average thresholds cause false positives for people whose
resting face differs from the mean (naturally tense brows, smaller eyes, etc.).

Solution: infer personal resting levels from observed calm moments using rolling
percentiles — no dedicated calibration window required.

After ~3 minutes of face presence, baselines become active and state estimation
uses deviations from personal resting levels instead of absolute thresholds.

Design: stateful. One PersonalBaselineTracker per tracked face per session.
"""

from collections import deque
import numpy as np
from geometry.math_utils import clamp

# Signals tracked for baseline (name → population default mean/std)
_POPULATION = {
    "blink_rate":        {"mean": 15.0,  "std": 5.0},
    "avg_EAR":           {"mean": 0.28,  "std": 0.04},
    "brow_tension":      {"mean": 0.15,  "std": 0.08},
    "facial_tension":    {"mean": 0.20,  "std": 0.08},
    "lip_tension":       {"mean": 0.10,  "std": 0.06},
    "gaze_volatility":   {"mean": 0.20,  "std": 0.10},
    "AU4":               {"mean": 0.12,  "std": 0.08},
    "expression_intensity": {"mean": 0.20, "std": 0.10},
}

# Higher value = more relaxed/open for these signals
_HIGH_IS_REST = {"avg_EAR"}

_ACTIVATION_MS   = 180_000   # 3 minutes face time before baseline is trusted
_MIN_SAMPLES     = 300       # ~10 s at 30 fps minimum buffer
_WINDOW_MS       = 300_000   # 5-minute rolling window
_PERCENTILE_LOW  = 10        # 10th percentile ≈ calm/resting for tension signals
_PERCENTILE_HIGH = 90        # 90th percentile ≈ most open eyes


class PersonalBaselineTracker:
    """
    Rolling within-session baseline estimator.

    Usage:
        baseline = PersonalBaselineTracker()
        baseline.update(signals_dict, ts_ms)
        if baseline.is_active():
            stress = baseline.elevated(brow_tension, "brow_tension")
    """

    def __init__(self, activation_ms: float = _ACTIVATION_MS):
        self.activation_ms = activation_ms
        self._buffers: dict[str, deque[tuple[float, float]]] = {
            name: deque(maxlen=9000) for name in _POPULATION
        }
        self._face_since_ms: float | None = None
        self._face_time_ms:   float = 0.0
        self._last_ts:       float | None = None
        self._external:       dict | None = None   # optional long-term baselines

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf.clear()
        self._face_since_ms = None
        self._face_time_ms  = 0.0
        self._last_ts       = None

    def load_external(self, external_baselines: dict | None) -> None:
        """Load persisted baselines from long-term memory (optional)."""
        self._external = external_baselines

    def update(self, signals: dict, ts_ms: float) -> None:
        """Record per-frame signal samples."""
        if self._face_since_ms is None:
            self._face_since_ms = ts_ms

        if self._last_ts is not None:
            dt = max(0.0, ts_ms - self._last_ts)
            self._face_time_ms += dt
        self._last_ts = ts_ms

        for name in _POPULATION:
            val = signals.get(name)
            if val is None:
                continue
            self._buffers[name].append((ts_ms, float(val)))

        self._trim(ts_ms)

    def _trim(self, ts_ms: float) -> None:
        cutoff = ts_ms - _WINDOW_MS
        for buf in self._buffers.values():
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def is_active(self) -> bool:
        if self._face_time_ms < self.activation_ms:
            return False
        return all(len(self._buffers[n]) >= _MIN_SAMPLES for n in _POPULATION)

    def get_rest_level(self, signal: str) -> float:
        """Personal resting level for a signal (or population mean if inactive)."""
        if signal in _HIGH_IS_REST:
            return self._percentile(signal, _PERCENTILE_HIGH)
        return self._percentile(signal, _PERCENTILE_LOW)

    def get_std(self, signal: str) -> float:
        vals = self._values(signal)
        if len(vals) < 30:
            pop = _POPULATION.get(signal, {"std": 0.1})
            return float(pop["std"])
        return max(float(np.std(vals)), 0.02)

    def elevated(self, value: float, signal: str) -> float:
        """
        How far above personal resting level (0 = at rest, 1 = strongly elevated).
        Uses 3× personal std as scale.
        """
        rest = self.get_rest_level(signal)
        scale = max(3.0 * self.get_std(signal), 0.05)
        if signal == "blink_rate":
            scale = max(scale, 5.0)
        return clamp((float(value) - rest) / scale, 0.0, 1.0)

    def ear_droop(self, avg_ear: float) -> float:
        """Lid droop relative to personal maximum EAR (most open = relaxed)."""
        if avg_ear <= 0:
            return 0.0
        rest_open = self.get_rest_level("avg_EAR")
        rest_open = max(rest_open, 0.18)
        return clamp(1.0 - float(avg_ear) / rest_open, 0.0, 1.0)

    def blink_rate_abnormal(self, blink_rate: float) -> float:
        """Abnormal blink rate vs personal resting rate."""
        if not self.is_active():
            if blink_rate > 25.0:
                return clamp((blink_rate - 25.0) / 15.0, 0.0, 1.0)
            if 0 < blink_rate < 8.0:
                return clamp((8.0 - blink_rate) / 8.0, 0.0, 1.0)
            return 0.0
        rest = self.get_rest_level("blink_rate")
        rest = max(rest, 8.0)
        high_dev = max(0.0, blink_rate - rest - 5.0)
        low_dev  = max(0.0, rest - 5.0 - blink_rate) if blink_rate > 0 else 0.0
        return clamp(max(high_dev, low_dev) / 12.0, 0.0, 1.0)

    def export_snapshot(self) -> dict:
        """Current baseline snapshot for JSON / debugging."""
        snap = {
            "active":           self.is_active(),
            "face_time_ms":     round(self._face_time_ms, 0),
            "activation_ms":    self.activation_ms,
            "population_fallback": not self.is_active(),
            "signals": {},
        }
        for name in _POPULATION:
            pop = _POPULATION[name]
            snap["signals"][name] = {
                "rest":     round(self.get_rest_level(name), 4),
                "std":      round(self.get_std(name), 4),
                "pop_mean": pop["mean"],
                "samples":  len(self._buffers[name]),
            }
        return snap

    def _values(self, signal: str) -> list[float]:
        return [v for _, v in self._buffers.get(signal, [])]

    def _percentile(self, signal: str, pct: float) -> float:
        vals = self._values(signal)
        pop  = _POPULATION.get(signal, {"mean": 0.0, "std": 0.1})

        if self._external and signal in self._external:
            ext = self._external[signal]
            if isinstance(ext, dict) and "mean" in ext:
                ext_mean = float(ext["mean"])
            else:
                ext_mean = float(ext)
            if len(vals) < _MIN_SAMPLES:
                return ext_mean

        if len(vals) < 30:
            return float(pop["mean"])

        observed = float(np.percentile(vals, pct))

        if self._external and signal in self._external:
            ext = self._external[signal]
            ext_mean = float(ext.get("mean", pop["mean"]) if isinstance(ext, dict) else ext)
            # Blend external with observed as session progresses
            w = min(1.0, len(vals) / 900.0)
            return (1.0 - w) * ext_mean + w * observed

        return observed
