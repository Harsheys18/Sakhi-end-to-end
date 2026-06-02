"""
outputs/json_output.py

Serialises the complete social state (base + derived) to LLM-ready JSON.

Covers every field in the original feature spec including the derived,
emotional, temporal, and LLM-final dimensions.
"""

import json
import time
from geometry.math_utils import clamp

_LOW_CONF = 0.35

_LEVEL_LABELS = [
    (0.80, "high"), (0.60, "moderate-high"),
    (0.40, "moderate"), (0.20, "low-moderate"), (0.00, "low"),
]

def _level(v: float) -> str:
    for thresh, label in _LEVEL_LABELS:
        if v >= thresh:
            return label
    return "low"

def _valence_label(v: float) -> str:
    if v > 0.65: return "positive"
    if v < 0.35: return "negative"
    return "neutral"

def _describe(name: str, dim: dict) -> str:
    val  = dim.get("value", 0.0)
    conf = dim.get("confidence", 0.0)
    unc  = " (uncertain)" if conf < _LOW_CONF else ""
    mapping = {
        "engagement":             f"{_level(val)} engagement{unc}",
        "attention":              f"{_level(val)} attention, looking {dim.get('direction','?')}{unc}",
        "stress_hints":           f"{_level(val)} stress indicators{unc}",
        "fatigue":                f"{_level(val)} fatigue{unc}",
        "confidence_level":       f"{_level(val)} behavioral confidence{unc}",
        "discomfort":             f"{_level(val)} discomfort{unc}",
        "valence":                f"{_valence_label(val)} valence{unc}",
        "arousal":                f"{_level(val)} arousal{unc}",
        "interaction_willingness":f"{_level(val)} interaction willingness{unc}",
        "comfort_estimate":       f"{_level(val)} comfort{unc}",
        "curiosity_estimate":     f"{_level(val)} curiosity{unc}",
        "cognitive_load_estimate":f"{_level(val)} cognitive load{unc}",
        "social_openness_estimate":f"{_level(val)} social openness{unc}",
        "dominance":              f"{_level(val)} dominance{unc}",
        "emotional_intensity":    f"{_level(val)} emotional intensity{unc}",
        "emotional_stability":    f"{_level(val)} emotional stability{unc}",
        "uncertainty_estimate":   f"{_level(val)} uncertainty{unc}",
    }
    return mapping.get(name, f"{name}: {_level(val)}{unc}")


def _build_narrative(state: dict) -> str:
    if not state.get("face_present", False):
        return "No face detected."

    parts = []

    attn = state.get("attention",  {})
    eng  = state.get("engagement", {})
    if attn.get("confidence", 0) >= _LOW_CONF:
        parts.append(
            f"The person shows {_describe('engagement', eng)}, "
            f"with {_describe('attention', attn)}."
        )

    valence = state.get("valence",  {})
    arousal = state.get("arousal",  {})
    comfort = state.get("comfort_estimate", {})
    if valence.get("confidence", 0) >= _LOW_CONF:
        parts.append(
            f"Affect: {_describe('valence', valence)}, "
            f"{_describe('arousal', arousal)}, "
            f"{_describe('comfort_estimate', comfort)}."
        )

    stress  = state.get("stress_hints",          {})
    fatigue = state.get("fatigue",               {})
    cog     = state.get("cognitive_load_estimate",{})
    if stress.get("value", 0) > 0.40 or fatigue.get("value", 0) > 0.40:
        parts.append(
            f"Notable: {_describe('stress_hints', stress)}, "
            f"{_describe('fatigue', fatigue)}, "
            f"{_describe('cognitive_load_estimate', cog)}."
        )

    cur  = state.get("curiosity_estimate",        {})
    dom  = state.get("dominance",                 {})
    unc  = state.get("uncertainty_estimate",      {})
    open_= state.get("social_openness_estimate",  {})
    parts.append(
        f"Social: {_describe('curiosity_estimate', cur)}, "
        f"{_describe('dominance', dom)}, "
        f"{_describe('social_openness_estimate', open_)}."
    )

    iw = state.get("interaction_willingness", {})
    if iw.get("confidence", 0) >= _LOW_CONF:
        parts.append(f"Interaction willingness: {_describe('interaction_willingness', iw)}.")

    flags = []
    if state.get("speaking"):               flags.append("speaking")
    if state.get("eye_contact"):            flags.append("eye contact")
    if state.get("sustained_attention"):    flags.append("sustained attention")
    if state.get("sustained_disengagement"):flags.append("sustained disengagement")
    if state.get("prolonged_downward_gaze"):flags.append("prolonged downward gaze")
    if state.get("prolonged_stillness"):    flags.append("prolonged stillness")
    if flags:
        parts.append("Flags: " + ", ".join(flags) + ".")

    return " ".join(parts)


def format_social_state(
    state:        dict,
    timestamp_ms: float | None = None,
    pretty:       bool = False,
) -> str:
    """
    Serialise the full social state to LLM-ready JSON.

    Args:
        state        : merged dict (base + derived from state/derived_state.py)
        timestamp_ms : current time in ms
        pretty       : indent=2 for readability

    Returns:
        JSON string.
    """
    ts = timestamp_ms if timestamp_ms is not None else time.time() * 1000.0

    # Core dimensions
    core_dims = [
        "engagement", "attention", "stress_hints", "fatigue",
        "confidence_level", "discomfort", "valence", "arousal",
        "interaction_willingness",
    ]
    # Extended derived dimensions
    derived_dims = [
        "comfort_estimate", "curiosity_estimate", "cognitive_load_estimate",
        "social_openness_estimate", "emotional_stability", "uncertainty_estimate",
        "dominance", "emotional_intensity", "emotion_transition_rate",
    ]

    dimensions = {}
    for name in core_dims + derived_dims:
        dim = state.get(name, {"value": 0.0, "confidence": 0.0})
        if not isinstance(dim, dict):
            dim = {"value": float(dim), "confidence": 0.0}
        entry = {
            "value":       round(float(dim.get("value",      0.0)), 3),
            "confidence":  round(float(dim.get("confidence", 0.0)), 3),
            "description": _describe(name, dim),
        }
        if name == "attention":
            entry["direction"] = dim.get("direction", "unknown")
        dimensions[name] = entry

    output = {
        "timestamp_ms":  round(ts, 1),
        "face_present":  state.get("face_present", False),
        "narrative":     _build_narrative(state),

        "dimensions":    dimensions,

        "behavioral": {
            "speaking":                state.get("speaking",                False),
            "eye_contact":             state.get("eye_contact",             False),
            "sustained_attention":     state.get("sustained_attention",     False),
            "sustained_disengagement": state.get("sustained_disengagement", False),
            "prolonged_downward_gaze": state.get("prolonged_downward_gaze", False),
            "prolonged_stillness":     state.get("prolonged_stillness",     False),
        },

        "temporal": {
            "engagement_trend":         round(state.get("engagement_trend",  0.0), 5),
            "attention_trend":          round(state.get("attention_trend",   0.0), 5),
            "gaze_trend":               round(state.get("gaze_trend",        0.0), 5),
            "movement_trend":           round(state.get("movement_trend",    0.0), 5),
            "attention_stability":      round(state.get("attention_stability", 1.0), 3),
            "interaction_stability":    round(state.get("interaction_stability",1.0), 3),
        },

        "movement": {
            "movement_energy":          round(state.get("movement_energy",      0.0), 3),
            "movement_variability":     round(state.get("movement_variability", 0.0), 3),
            "face_stability":           round(state.get("face_stability",       1.0), 3),
            "restlessness_score":       round(state.get("restlessness_score",   0.0), 3),
        },

        "llm_visual": {
            "visual_engagement":        round(state.get("engagement",       {}).get("value", 0.0), 3),
            "visual_attention":         round(state.get("attention",        {}).get("value", 0.0), 3),
            "visual_energy":            round(state.get("visual_energy",        0.0), 3),
            "visual_fatigue":           round(state.get("fatigue",          {}).get("value", 0.0), 3),
            "visual_stress_hints":      round(state.get("stress_hints",     {}).get("value", 0.0), 3),
            "visual_confidence":        round(state.get("confidence_level", {}).get("value", 0.0), 3),
            "visual_social_openness":   round(state.get("visual_social_openness", 0.0), 3),
            "visual_discomfort":        round(state.get("discomfort",       {}).get("value", 0.0), 3),
            "visual_interest":          round(state.get("visual_interest",      0.0), 3),
            "visual_emotional_valence": round(state.get("valence",          {}).get("value", 0.5), 3),
            "visual_emotional_arousal": round(state.get("arousal",          {}).get("value", 0.0), 3),
            "system_confidence":        round(state.get("system_confidence",    0.0), 3),
        },

        "meta": state.get("meta", {}),
    }

    return json.dumps(output, indent=2 if pretty else None)


def parse_social_state(json_str: str) -> dict:
    return json.loads(json_str)
