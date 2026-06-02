"""
state/uncertainty.py

Uncertainty-aware estimation for all emotional/social dimensions.

Uncertainty sources:
  1. Sensor uncertainty  — low landmark confidence, bad lighting, occlusion
  2. Temporal variance   — dimension oscillating rapidly = uncertain
  3. Signal disagreement — geometric signal vs GRU signal diverge = uncertain
  4. Affective ambiguity — multiple plausible emotional states (e.g., anger vs fear)

Output per dimension:
  confidence   : [0, 1]   how reliable this estimate is
  uncertainty  : [0, 1]   complement of confidence (kept separate for clarity)
  stability    : [0, 1]   temporal consistency over rolling window
  ci_low       : float    lower bound of 68% confidence interval
  ci_high      : float    upper bound of 68% confidence interval

Design: stateful (rolling buffers). One UncertaintyTracker per face.
"""

from collections import deque
import numpy as np
from geometry.math_utils import clamp

_BUFFER_LEN = 60   # ~2 s at 30 fps


class UncertaintyTracker:
    """
    Per-face uncertainty tracker for all emotional dimensions.

    Usage:
        tracker = UncertaintyTracker()
        unc     = tracker.update(base_state, affective_emb, gru_state, env_conf)
    """

    def __init__(self, buffer_len: int = _BUFFER_LEN):
        self._bufs: dict[str, deque] = {}
        self._buf_len = buffer_len

    def _get_buf(self, name: str) -> deque:
        if name not in self._bufs:
            self._bufs[name] = deque(maxlen=self._buf_len)
        return self._bufs[name]

    def update(
        self,
        base_state:    dict,
        affective_emb: dict,
        gru_state:     dict,
        sensor_conf:   float = 1.0,
    ) -> dict:
        """
        Compute per-dimension uncertainty estimates.

        Args:
            base_state    : dict from state/social_state (has value + confidence per dim)
            affective_emb : dict from features/affective_embedding
            gru_state     : dict from temporal/affective_gru
            sensor_conf   : overall sensor confidence (from meta.overall_confidence)

        Returns:
            {
              "dimensions": {
                "valence":  {"confidence": f, "uncertainty": f,
                             "stability": f, "ci_low": f, "ci_high": f},
                ...
              },
              "global_uncertainty"   : float  # overall pipeline uncertainty
              "affective_ambiguity"  : float  # multiple plausible emotional states
              "signal_disagreement"  : float  # geometric vs GRU disagreement
              "temporal_instability" : float  # how rapidly all signals are changing
            }
        """
        dim_names = [
            "engagement", "attention", "stress_hints", "fatigue",
            "confidence_level", "discomfort", "valence", "arousal",
        ]

        # ── Per-dimension uncertainty ─────────────────────────────────────────
        dim_results = {}
        stabilities = []

        for name in dim_names:
            dim = base_state.get(name, {})
            if not isinstance(dim, dict):
                dim = {"value": float(dim), "confidence": 0.5}

            val  = float(dim.get("value",      0.0))
            conf = float(dim.get("confidence", 0.5))

            buf = self._get_buf(name)
            buf.append(val)

            arr       = np.array(buf, dtype=np.float32)
            stability = self._stability(arr)
            stabilities.append(stability)

            # Combined confidence: sensor × base_confidence × stability
            combined_conf = clamp(
                sensor_conf * conf * (0.5 + 0.5 * stability),
                0.05, 1.0
            )
            uncertainty   = 1.0 - combined_conf

            # 68% confidence interval (mean ± 1 std of rolling buffer)
            mu  = float(arr.mean()) if len(arr) > 0 else val
            std = float(arr.std())  if len(arr) > 1 else 0.1
            ci_low  = clamp(mu - std, 0.0, 1.0)
            ci_high = clamp(mu + std, 0.0, 1.0)

            dim_results[name] = {
                "confidence": round(combined_conf, 3),
                "uncertainty":round(uncertainty,   3),
                "stability":  round(stability,     3),
                "ci_low":     round(ci_low,        3),
                "ci_high":    round(ci_high,       3),
            }

        # ── Signal disagreement: geometry vs GRU ─────────────────────────────
        geom_val  = (affective_emb.get("valence_geom",  0.0) + 1.0) / 2.0  # map to [0,1]
        gru_val   = (gru_state.get("gru_valence",  0.0) + 1.0) / 2.0
        geom_aro  = affective_emb.get("arousal_geom", 0.0)
        gru_aro   = gru_state.get("gru_arousal",  0.0)

        val_disagree = abs(geom_val - gru_val)
        aro_disagree = abs(geom_aro - gru_aro)
        signal_disagree = clamp((val_disagree + aro_disagree) / 2.0, 0.0, 1.0)

        # ── Affective ambiguity ───────────────────────────────────────────────
        # When both valence and arousal are near 0.5, state is ambiguous
        geom_v_dist = abs(affective_emb.get("valence_geom", 0.0))  # 0 = ambiguous
        geom_a_dist = abs(geom_aro - 0.5) * 2.0                   # 0 = ambiguous
        affective_ambiguity = clamp(
            1.0 - (geom_v_dist * 0.5 + geom_a_dist * 0.5),
            0.0, 1.0
        )

        # ── Temporal instability ──────────────────────────────────────────────
        temporal_instability = clamp(
            1.0 - float(np.mean(stabilities)),
            0.0, 1.0
        )

        # ── Global uncertainty ────────────────────────────────────────────────
        global_unc = clamp(
            0.40 * (1.0 - sensor_conf)
          + 0.25 * temporal_instability
          + 0.20 * signal_disagree
          + 0.15 * affective_ambiguity,
            0.0, 1.0
        )

        return {
            "dimensions":           dim_results,
            "global_uncertainty":   round(global_unc,            3),
            "affective_ambiguity":  round(affective_ambiguity,   3),
            "signal_disagreement":  round(signal_disagree,       3),
            "temporal_instability": round(temporal_instability,  3),
        }

    def reset(self) -> None:
        self._bufs.clear()

    @staticmethod
    def _stability(arr: np.ndarray) -> float:
        """Temporal stability: 1 - normalized std of rolling buffer."""
        if len(arr) < 3:
            return 1.0
        std = float(arr.std())
        return clamp(1.0 - std / 0.25, 0.0, 1.0)
