"""
fusion/signal_fusion.py

Signal fusion utilities for combining multiple noisy feature streams
into stable, confidence-weighted estimates.

Used by state/social_state.py but also available for direct use when
building custom fusion pipelines.

Provides:
  weighted_mean()      : confidence-weighted scalar fusion
  fuse_signals()       : fuse a list of (value, weight, confidence) tuples
  TemporalDampener     : prevent rapid oscillation of derived state values
  HysteresisFilter     : add hysteresis to boolean state transitions

Design: pure functions + stateful classes. No I/O. No globals.
"""

from collections import deque
import numpy as np
from geometry.math_utils import clamp


# ── Stateless fusion ───────────────────────────────────────────────────────────

def weighted_mean(values: list[float], weights: list[float]) -> float:
    """
    Weighted arithmetic mean. Weights do not need to sum to 1.

    Args:
        values  : List of scalar values.
        weights : Corresponding positive weights.

    Returns:
        Weighted mean, or 0.0 if total weight is zero.

    Example:
        weighted_mean([0.8, 0.4], [0.9, 0.3])  # → 0.7 (high-confidence val dominates)
    """
    total_w = sum(weights)
    if total_w < 1e-8:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def fuse_signals(
    signals: list[tuple[float, float, float]],
    clamp_output: bool = True,
) -> tuple[float, float]:
    """
    Fuse a list of (value, weight, confidence) triples into one estimate.

    The effective weight of each signal is weight × confidence, so signals
    with low confidence contribute proportionally less to the result.

    Args:
        signals       : List of (value, weight, confidence) tuples.
                        value      ∈ [0, 1]
                        weight     > 0   (relative importance)
                        confidence ∈ [0, 1]
        clamp_output  : If True, clamp result to [0, 1].

    Returns:
        (fused_value, fused_confidence)
          fused_value      : weighted mean of values
          fused_confidence : weighted mean of confidences

    Example:
        fuse_signals([
            (0.8, 0.5, 0.9),   # strong signal, high confidence
            (0.3, 0.3, 0.2),   # weak signal, low confidence
            (0.6, 0.2, 0.7),
        ])
        # → (≈0.70, ≈0.73)
    """
    if not signals:
        return 0.0, 0.0

    eff_weights = [w * c for _, w, c in signals]
    total_eff   = sum(eff_weights)

    if total_eff < 1e-8:
        # All signals have near-zero confidence — return simple mean
        vals = [v for v, _, _ in signals]
        return float(np.mean(vals)), 0.0

    fused_val  = sum(v * ew for (v, _, _), ew in zip(signals, eff_weights)) / total_eff
    fused_conf = sum(c * w  for (_, w, c) in signals) / sum(w for _, w, _ in signals)

    if clamp_output:
        fused_val = clamp(fused_val, 0.0, 1.0)

    return float(fused_val), float(clamp(fused_conf, 0.0, 1.0))


# ── Stateful: TemporalDampener ─────────────────────────────────────────────────

class TemporalDampener:
    """
    Prevent rapid oscillation of a derived scalar state.

    Uses an asymmetric EMA: rise and fall can have different speeds.
    This is useful for signals like "stress" or "engagement" where you want:
      - Fast response to increases (alert robot quickly)
      - Slow decay on decreases (don't flip-flop)

    Usage:
        dampener = TemporalDampener(rise_alpha=0.3, fall_alpha=0.08)
        smoothed = dampener.update(raw_value)
    """

    def __init__(
        self,
        rise_alpha: float = 0.25,
        fall_alpha: float = 0.10,
        initial:    float = 0.0,
    ):
        """
        Args:
            rise_alpha : EMA alpha when value is increasing. Higher = faster rise.
            fall_alpha : EMA alpha when value is decreasing. Lower = slower fall.
            initial    : Starting value.
        """
        self._value     = initial
        self.rise_alpha = rise_alpha
        self.fall_alpha = fall_alpha

    def update(self, raw: float) -> float:
        """
        Apply asymmetric EMA and return the dampened value.

        Args:
            raw : New raw signal value.

        Returns:
            Dampened scalar.
        """
        alpha = self.rise_alpha if raw > self._value else self.fall_alpha
        self._value = alpha * raw + (1 - alpha) * self._value
        return float(self._value)

    def reset(self, value: float = 0.0) -> None:
        self._value = value

    @property
    def value(self) -> float:
        return float(self._value)


# ── Stateful: HysteresisFilter ────────────────────────────────────────────────

class HysteresisFilter:
    """
    Add hysteresis to boolean state transitions.

    Prevents rapid toggling between True/False by requiring the input to
    cross different thresholds to change state:
      - OFF → ON  : input must exceed `high_thresh`
      - ON  → OFF : input must fall below `low_thresh`

    Usage:
        filt = HysteresisFilter(low_thresh=0.35, high_thresh=0.55)
        is_speaking = filt.update(speaking_probability)
    """

    def __init__(
        self,
        low_thresh:  float = 0.30,
        high_thresh: float = 0.60,
        initial:     bool  = False,
    ):
        """
        Args:
            low_thresh  : Input must drop below this to switch OFF.
            high_thresh : Input must rise above this to switch ON.
            initial     : Starting state.
        """
        assert low_thresh < high_thresh, "low_thresh must be < high_thresh"
        self.low_thresh  = low_thresh
        self.high_thresh = high_thresh
        self._state      = initial

    def update(self, value: float) -> bool:
        """
        Update state with hysteresis.

        Args:
            value : Continuous input signal ∈ [0, 1].

        Returns:
            Current boolean state after applying hysteresis.
        """
        if not self._state and value >= self.high_thresh:
            self._state = True
        elif self._state and value <= self.low_thresh:
            self._state = False
        return self._state

    def reset(self, state: bool = False) -> None:
        self._state = state

    @property
    def state(self) -> bool:
        return self._state


# ── Stateful: MultiSignalFuser ────────────────────────────────────────────────

class MultiSignalFuser:
    """
    Maintains a rolling fused estimate from multiple named signal streams.

    Designed for use in state/social_state.py to produce per-dimension
    estimates that adapt to which sensors are currently reliable.

    Usage:
        fuser = MultiSignalFuser(dampener=TemporalDampener(0.3, 0.08))
        fuser.push("eye",  value=0.8, weight=0.4, confidence=0.9)
        fuser.push("head", value=0.6, weight=0.3, confidence=0.7)
        fuser.push("gaze", value=0.7, weight=0.3, confidence=0.85)
        result_value, result_conf = fuser.fuse()
    """

    def __init__(self, dampener: TemporalDampener | None = None):
        self._signals:  dict[str, tuple[float, float, float]] = {}
        self._dampener = dampener

    def push(
        self,
        name:       str,
        value:      float,
        weight:     float,
        confidence: float,
    ) -> None:
        """
        Register or update a named signal.

        Args:
            name       : Signal identifier (e.g. "eye", "head").
            value      : Signal value ∈ [0, 1].
            weight     : Relative importance weight > 0.
            confidence : Sensor confidence ∈ [0, 1].
        """
        self._signals[name] = (value, weight, confidence)

    def fuse(self) -> tuple[float, float]:
        """
        Fuse all registered signals and clear the buffer.

        Returns:
            (fused_value, fused_confidence) — both ∈ [0, 1].
            Returns (0.0, 0.0) when no signals have been pushed.
        """
        if not self._signals:
            return 0.0, 0.0
        raw_val, conf = fuse_signals(list(self._signals.values()))
        if self._dampener is not None:
            raw_val = self._dampener.update(raw_val)
        self._signals.clear()
        return float(raw_val), float(conf)

    def reset(self) -> None:
        self._signals.clear()
        if self._dampener is not None:
            self._dampener.reset()
