"""
outputs/vision_output.py

Production vision output — schema vision_v1.

Single JSON object per emission, designed for:
  - Fusion layer (combines with audio + memory)
  - Short-term memory updates
  - Downstream logging

Does NOT include LLM context text, audio handoff, or memory blocks —
those belong to the fusion layer.
"""

import json
import time
import uuid
from geometry.math_utils import clamp

SCHEMA = "vision_v1"
SCHEMA_VERSION = "1.0"


def new_session_id() -> str:
    return f"ses_{uuid.uuid4().hex[:12]}"


def _dim_val(state: dict, key: str, default: float = 0.0) -> float:
    v = state.get(key)
    if isinstance(v, dict):
        return float(v.get("value", default))
    if isinstance(v, (int, float)):
        return float(v)
    return default


def _dim_conf(state: dict, key: str, default: float = 0.0) -> float:
    v = state.get(key)
    if isinstance(v, dict):
        return float(v.get("confidence", default))
    return default


def _round3(v) -> float:
    return round(float(v), 3)


def format_vision_v1(
    state:            dict,
    session_id:       str,
    emit_reason:      str = "",
    timestamp_ms:     float | None = None,
    baseline_active:  bool = False,
    baseline_snapshot: dict | None = None,
    pretty:           bool = False,
) -> str:
    """
    Serialize merged pipeline state to vision_v1 JSON string.

    Args:
        state             : full merged state dict from main loop
        session_id        : current session identifier
        emit_reason       : why this frame was emitted
        timestamp_ms      : emission timestamp
        baseline_active   : whether personal baseline calibration is active
        baseline_snapshot : optional baseline debug snapshot
        pretty            : indent=2 for human reading

    Returns:
        JSON string (one object per line when pretty=False).
    """
    ts = timestamp_ms if timestamp_ms is not None else time.time() * 1000.0
    face_present = bool(state.get("face_present", False))
    if not baseline_active:
        baseline_active = bool(state.get("baseline_active", False))

    unc = state.get("uncertainty", {})
    if isinstance(unc, dict) and "dimensions" in unc:
        unc_dims = unc.get("dimensions", {})
    else:
        unc_dims = {}

    gru_val = float(state.get("gru_valence", 0.0))
    gru_aro = float(state.get("gru_arousal", 0.0))
    gru_dom = float(state.get("gru_dominance", 0.0))

    dyn = state if isinstance(state, dict) else {}
    body = state.get("body", {}) or {}
    hands = state.get("hands", {}) or {}
    head_t_meta = state.get("meta", {}) if isinstance(state.get("meta"), dict) else {}

    output = {
        "schema":           SCHEMA,
        "schema_version":   SCHEMA_VERSION,
        "ts_ms":            round(ts, 1),
        "session_id":       session_id,
        "face_present":     face_present,
        "emit_reason":      emit_reason,
        "baseline_active":  baseline_active,

        "support": {
            "urgency":                state.get("support_urgency", "none"),
            "distress_level":         _round3(state.get("distress_level", 0.0)),
            "distress_type":          state.get("distress_type", "none"),
            "openness_to_support":    _round3(state.get("openness_to_support", 0.0)),
            "genuine_smile":          bool(state.get("genuine_smile", False)),
            "genuine_smile_score":    _round3(state.get("genuine_smile_score", 0.0)),
            "emotional_suppression":  _round3(state.get("emotional_suppression", 0.0)),
            "overwhelm_probability":  _round3(state.get("overwhelm_probability", 0.0)),
            "comfort_seeking":        _round3(state.get("comfort_seeking", 0.0)),
            "emotional_withdrawal":   _round3(state.get("emotional_withdrawal", 0.0)),
        },

        "affect": {
            "valence":           _round3(gru_val),
            "arousal":           _round3(gru_aro),
            "dominance":         _round3(gru_dom),
            "quadrant":          state.get("gru_affect_quadrant",
                                 state.get("current_quadrant", "HVLA")),
            "quadrant_label":    state.get("quadrant_label", "calm/content"),
            "coherence":         _round3(state.get("temporal_coherence", 0.0)),
            "momentum":          _round3(state.get("emotional_momentum", 0.0)),
            "valence_velocity":  _round3(state.get("valence_velocity", 0.0)),
            "arousal_velocity":  _round3(state.get("arousal_velocity", 0.0)),
        },

        "social": {
            "engagement":               _round3(_dim_val(state, "engagement")),
            "attention_score":          _round3(_dim_val(state, "attention")),
            "attention_direction":      (
                state.get("attention", {}).get("direction", "unknown")
                if isinstance(state.get("attention"), dict)
                else head_t_meta.get("head_direction", "unknown")
            ),
            "eye_contact_rate":         _round3(state.get("eye_contact_rate", 0.0)),
            "rapport":                  _round3(state.get("rapport_signal", 0.0)),
            "conversational_readiness": _round3(state.get("conversational_readiness", 0.0)),
            "social_comfort":           _round3(state.get("social_comfort", 0.0)),
            "social_openness":          _round3(state.get("social_openness",
                                              _dim_val(state, "social_openness_estimate"))),
            "responsiveness":           _round3(state.get("social_responsiveness", 0.0)),
            "behavioral_synchrony":     _round3(state.get("behavioral_synchrony", 0.0)),
        },

        "dynamics": {
            "stress_escalating":     bool(state.get("stress_escalating", False)),
            "stress_de_escalating":  bool(state.get("stress_de_escalating", False)),
            "engagement_shift":      state.get("engagement_shift", "stable"),
            "transition_occurred":   bool(state.get("transition_occurred", False)),
            "transition_from":       state.get("transition_from", ""),
            "transition_to":         state.get("transition_to", ""),
            "quadrant_duration_ms":  round(float(state.get("quadrant_duration_ms", 0.0)), 1),
            "state_persistence":     _round3(state.get("state_persistence", 0.0)),
            "recent_transitions":    int(state.get("recent_transitions", 0)),
            "engagement_trend":      _round3(state.get("engagement_trend", 0.0)),
            "stress_trend":          _round3(state.get("stress_trend", 0.0)),
        },

        "behavioral": {
            "speaking":                 bool(state.get("speaking", False)),
            "eye_contact":              bool(state.get("eye_contact", False)),
            "yawning":                  bool(state.get("yawning", False)),
            "nodding":                  bool(state.get("nod_detected", False)),
            "head_shaking":             bool(state.get("shake_detected", False)),
            "fidgeting":                _round3(state.get("fidget_probability", 0.0)),
            "gesture_frequency":        _round3(
                state.get("gesture_frequency",
                          (state.get("hands", {}) or {}).get("gesture_frequency", 0.0)
                          if isinstance(state.get("hands"), dict) else 0.0)
            ),
            "hand_near_face":           bool((hands or {}).get("hand_near_face", False)),
            "crossed_arms":             _round3((hands or {}).get("crossed_arms_probability", 0.0)),
            "leaning_direction":        (body or {}).get("leaning_direction", "center"),
            "leaning_intensity":        _round3((body or {}).get("leaning_intensity", 0.0)),
            "prolonged_stillness":      bool(state.get("prolonged_stillness", False)),
            "prolonged_downward_gaze":  bool(state.get("prolonged_downward_gaze", False)),
        },

        "uncertainty": {
            "global":             _round3(state.get("global_uncertainty", 0.0)),
            "sensor_confidence":  _round3(
                head_t_meta.get("overall_confidence",
                                state.get("system_confidence", 0.5))
            ),
            "signal_disagreement": _round3(state.get("signal_disagreement", 0.0)),
            "affective_ambiguity": _round3(state.get("affective_ambiguity", 0.0)),
            "dimensions": _build_uncertainty_dims(state, unc_dims),
        },
    }

    if baseline_snapshot is not None:
        output["baseline"] = baseline_snapshot

    return json.dumps(output, indent=2 if pretty else None, ensure_ascii=False)


def parse_vision_v1(json_str: str) -> dict:
    return json.loads(json_str)


def _build_uncertainty_dims(state: dict, unc_dims: dict) -> dict:
    out = {}
    for out_key, src_key in (
        ("valence", "valence"),
        ("arousal", "arousal"),
        ("engagement", "engagement"),
    ):
        u = unc_dims.get(src_key, {})
        val = _dim_val(state, src_key)
        conf = float(u.get("confidence", _dim_conf(state, src_key, 0.5)))
        out[out_key] = {
            "confidence": _round3(conf),
            "ci_low":     _round3(float(u.get("ci_low",  clamp(val - 0.1, 0.0, 1.0)))),
            "ci_high":    _round3(float(u.get("ci_high", clamp(val + 0.1, 0.0, 1.0)))),
        }
    d_val = float(state.get("distress_level", 0.0))
    u = unc_dims.get("stress_hints", {})
    out["distress"] = {
        "confidence": _round3(float(u.get("confidence", _dim_conf(state, "stress_hints", 0.5)))),
        "ci_low":     _round3(clamp(d_val - 0.1, 0.0, 1.0)),
        "ci_high":    _round3(clamp(d_val + 0.1, 0.0, 1.0)),
    }
    return out
