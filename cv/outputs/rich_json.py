"""
outputs/rich_json.py

Rich affective JSON output for the upgraded social perception system.

Produces the full affective state as LLM-ready JSON including:
  - All social state dimensions with uncertainty intervals
  - Affective embedding (continuous 48-dim vector)
  - GRU temporal emotional state
  - Emotional dynamics (transitions, momentum, escalation)
  - Social cognition signals
  - Uncertainty metadata

The output is designed to give an LLM everything it needs to understand:
  - What emotional state the person is in RIGHT NOW
  - How confident we are in that estimate
  - How the state is CHANGING over time
  - What the social interaction quality is

Usage:
    json_str = format_rich_state(rich_state, timestamp_ms=..., pretty=True)
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
    for t, l in _LEVEL_LABELS:
        if v >= t: return l
    return "low"

def _valence_label(v: float) -> str:
    # v is in [-1, 1]
    if v > 0.3:   return "positive"
    if v < -0.3:  return "negative"
    return "neutral"


def _build_rich_narrative(state: dict) -> str:
    """Natural-language summary integrating all new signals."""
    if not state.get("face_present", False):
        return "No face detected. Visual social signals unavailable."

    parts = []

    # Core state
    eng  = state.get("engagement",  {}).get("value", 0.0)
    attn = state.get("attention",   {}).get("value", 0.0)
    val  = state.get("valence",     {}).get("value", 0.5)
    aro  = state.get("arousal",     {}).get("value", 0.0)

    # Affective embedding
    geom_val = state.get("valence_geom",  0.0)
    geom_aro = state.get("arousal_geom",  0.0)
    quadrant = state.get("gru_affect_quadrant", state.get("affect_quadrant", ""))
    quad_label = state.get("quadrant_label", "")

    # Dynamics
    stress_esc  = state.get("stress_escalating",   False)
    stress_de   = state.get("stress_de_escalating",False)
    eng_shift   = state.get("engagement_shift",    "stable")
    transition  = state.get("transition_occurred", False)

    # Social
    rapport     = state.get("rapport_signal",           0.0)
    social_com  = state.get("social_comfort",           0.0)
    conv_ready  = state.get("conversational_readiness", 0.0)

    # Uncertainty
    global_unc  = state.get("global_uncertainty", 0.0)
    unc_note    = " (uncertain)" if global_unc > 0.5 else ""

    parts.append(
        f"The person is in a {quad_label if quad_label else _level(eng) + ' engagement'} "
        f"state{unc_note}."
    )

    if geom_val != 0 or geom_aro != 0:
        parts.append(
            f"Affect appears {_valence_label(geom_val)} "
            f"(valence={geom_val:+.2f}) with {_level(geom_aro)} arousal."
        )

    if transition:
        parts.append(
            f"Emotional state just shifted from "
            f"{state.get('transition_from','')} → {state.get('transition_to','')}."
        )

    if stress_esc:
        parts.append("Stress indicators are escalating.")
    elif stress_de:
        parts.append("Stress indicators are de-escalating.")

    if eng_shift != "stable":
        parts.append(f"Engagement is {eng_shift}.")

    parts.append(
        f"Social: rapport={_level(rapport)}, "
        f"conversational readiness={_level(conv_ready)}, "
        f"social comfort={_level(social_com)}."
    )

    flags = []
    if state.get("speaking"):               flags.append("speaking")
    if state.get("eye_contact"):            flags.append("eye contact")
    if state.get("sustained_attention"):    flags.append("sustained attention")
    if state.get("fidget_probability", 0) > 0.5: flags.append("fidgeting")
    if flags:
        parts.append("Flags: " + ", ".join(flags) + ".")

    return " ".join(parts)


def format_rich_state(
    state:        dict,
    timestamp_ms: float | None = None,
    pretty:       bool = False,
) -> str:
    """
    Serialise the full rich affective state to JSON.

    Args:
        state        : merged dict containing all module outputs
        timestamp_ms : current time ms
        pretty       : indent=2

    Returns:
        JSON string ready for LLM prompt injection.
    """
    ts = timestamp_ms if timestamp_ms is not None else time.time() * 1000.0

    # ── Core social dimensions with uncertainty ────────────────────────────────
    core_dims = [
        "engagement", "attention", "stress_hints", "fatigue",
        "confidence_level", "discomfort", "valence", "arousal",
        "interaction_willingness",
    ]
    derived_dims = [
        "comfort_estimate", "curiosity_estimate", "cognitive_load_estimate",
        "social_openness_estimate", "emotional_stability", "uncertainty_estimate",
        "dominance", "emotional_intensity",
    ]

    unc_data = state.get("uncertainty", {}).get("dimensions", {})

    dimensions = {}
    for name in core_dims + derived_dims:
        dim = state.get(name, {"value": 0.0, "confidence": 0.0})
        if not isinstance(dim, dict):
            dim = {"value": float(dim), "confidence": 0.5}

        val  = round(float(dim.get("value",      0.0)), 3)
        conf = round(float(dim.get("confidence", 0.0)), 3)
        unc  = unc_data.get(name, {})

        entry = {
            "value":      val,
            "confidence": conf,
            "uncertainty":round(float(unc.get("uncertainty", 1.0 - conf)), 3),
            "stability":  round(float(unc.get("stability",   1.0)),         3),
            "ci_low":     round(float(unc.get("ci_low",      max(0,val-0.1))),3),
            "ci_high":    round(float(unc.get("ci_high",     min(1,val+0.1))),3),
        }
        if name == "attention":
            entry["direction"] = state.get("attention", {}).get("direction", "unknown") \
                if isinstance(state.get("attention"), dict) else "unknown"
        dimensions[name] = entry

    # ── Affective embedding ────────────────────────────────────────────────────
    affective = {
        "embedding":          state.get("embedding",          []),
        "valence_geom":       round(float(state.get("valence_geom",    0.0)), 4),
        "arousal_geom":       round(float(state.get("arousal_geom",    0.0)), 4),
        "dominance_geom":     round(float(state.get("dominance_geom",  0.0)), 4),
        "affect_quadrant":    state.get("affect_quadrant",    "HVLA"),
        "expression_conf":    round(float(state.get("expression_conf", 0.0)), 3),
        "gru_valence":        round(float(state.get("gru_valence",     0.0)), 4),
        "gru_arousal":        round(float(state.get("gru_arousal",     0.0)), 4),
        "gru_dominance":      round(float(state.get("gru_dominance",   0.0)), 4),
        "gru_affect_quadrant":state.get("gru_affect_quadrant","HVLA"),
        "temporal_coherence": round(float(state.get("temporal_coherence", 0.0)), 4),
        "emotional_momentum": round(float(state.get("emotional_momentum", 0.0)), 4),
    }

    # ── Emotional dynamics ────────────────────────────────────────────────────
    dynamics = {
        "current_quadrant":      state.get("current_quadrant",      "HVLA"),
        "quadrant_label":        state.get("quadrant_label",         "calm/content"),
        "quadrant_duration_ms":  round(float(state.get("quadrant_duration_ms", 0.0)), 1),
        "transition_occurred":   state.get("transition_occurred",    False),
        "transition_from":       state.get("transition_from",        ""),
        "transition_to":         state.get("transition_to",          ""),
        "stress_escalating":     state.get("stress_escalating",      False),
        "stress_de_escalating":  state.get("stress_de_escalating",   False),
        "stress_trend":          round(float(state.get("stress_trend",       0.0)), 5),
        "engagement_shift":      state.get("engagement_shift",       "stable"),
        "engagement_trend":      round(float(state.get("engagement_trend",   0.0)), 5),
        "affective_inertia":     round(float(state.get("affective_inertia",  0.0)), 4),
        "valence_velocity":      round(float(state.get("valence_velocity",   0.0)), 5),
        "state_persistence":     round(float(state.get("state_persistence",  0.0)), 3),
        "recent_transitions":    state.get("recent_transitions",     0),
    }

    # ── Social cognition ──────────────────────────────────────────────────────
    social = {
        "social_responsiveness":    round(float(state.get("social_responsiveness",    0.0)), 3),
        "interaction_reciprocity":  round(float(state.get("interaction_reciprocity",  0.0)), 3),
        "conversational_readiness": round(float(state.get("conversational_readiness", 0.0)), 3),
        "social_comfort":           round(float(state.get("social_comfort",           0.0)), 3),
        "social_openness":          round(float(state.get("social_openness",          0.0)), 3),
        "rapport_signal":           round(float(state.get("rapport_signal",           0.0)), 3),
        "social_attention_quality": round(float(state.get("social_attention_quality", 0.0)), 3),
        "proxemic_comfort":         round(float(state.get("proxemic_comfort",         0.0)), 3),
        "behavioral_synchrony":     round(float(state.get("behavioral_synchrony",     0.0)), 3),
    }

    # ── Uncertainty metadata ──────────────────────────────────────────────────
    uncertainty_meta = {
        "global_uncertainty":   round(float(state.get("global_uncertainty",   0.0)), 3),
        "affective_ambiguity":  round(float(state.get("affective_ambiguity",  0.0)), 3),
        "signal_disagreement":  round(float(state.get("signal_disagreement",  0.0)), 3),
        "temporal_instability": round(float(state.get("temporal_instability", 0.0)), 3),
    }

    # ── LLM-optimised flat view ───────────────────────────────────────────────
    llm_visual = {
        "visual_engagement":         round(float(state.get("engagement",  {}).get("value", 0.0) if isinstance(state.get("engagement"), dict) else 0.0), 3),
        "visual_attention":          round(float(state.get("attention",   {}).get("value", 0.0) if isinstance(state.get("attention"), dict) else 0.0), 3),
        "visual_energy":             round(float(state.get("visual_energy",        0.0)), 3),
        "visual_fatigue":            round(float(state.get("fatigue",     {}).get("value", 0.0) if isinstance(state.get("fatigue"), dict) else 0.0), 3),
        "visual_stress_hints":       round(float(state.get("stress_hints",{}).get("value", 0.0) if isinstance(state.get("stress_hints"), dict) else 0.0), 3),
        "visual_confidence":         round(float(state.get("confidence_level",{}).get("value", 0.0) if isinstance(state.get("confidence_level"), dict) else 0.0), 3),
        "visual_social_openness":    round(float(state.get("visual_social_openness", 0.0)), 3),
        "visual_discomfort":         round(float(state.get("discomfort",  {}).get("value", 0.0) if isinstance(state.get("discomfort"), dict) else 0.0), 3),
        "visual_interest":           round(float(state.get("visual_interest",  0.0)), 3),
        "visual_emotional_valence":  round((float(state.get("gru_valence", 0.0)) + 1.0) / 2.0, 3),  # GRU-smoothed, mapped [0,1]
        "visual_emotional_arousal":  round(float(state.get("gru_arousal", 0.0)), 3),
        "system_confidence":         round(float(state.get("system_confidence", 0.0)), 3),
        # New rich fields for LLM
        "social_comfort":            round(float(state.get("social_comfort",           0.0)), 3),
        "rapport_signal":            round(float(state.get("rapport_signal",           0.0)), 3),
        "conversational_readiness":  round(float(state.get("conversational_readiness", 0.0)), 3),
        "affect_quadrant":           state.get("gru_affect_quadrant", "HVLA"),
        "stress_escalating":         state.get("stress_escalating",   False),
        "engagement_shift":          state.get("engagement_shift",    "stable"),
        "temporal_coherence":        round(float(state.get("temporal_coherence", 0.0)), 3),
        "global_uncertainty":        round(float(state.get("global_uncertainty", 0.0)), 3),
    }

    output = {
        "timestamp_ms":  round(ts, 1),
        "face_present":  state.get("face_present", False),
        "narrative":     _build_rich_narrative(state),
        "dimensions":    dimensions,
        "affective":     affective,
        "dynamics":      dynamics,
        "social":        social,
        "uncertainty":   uncertainty_meta,
        "llm_visual":    llm_visual,
        "behavioral": {
            "speaking":                state.get("speaking",                False),
            "eye_contact":             state.get("eye_contact",             False),
            "sustained_attention":     state.get("sustained_attention",     False),
            "sustained_disengagement": state.get("sustained_disengagement", False),
            "prolonged_downward_gaze": state.get("prolonged_downward_gaze", False),
            "fidget_probability":      round(float(state.get("fidget_probability", 0.0)), 3),
        },
        "meta": state.get("meta", {}),
    }

    return json.dumps(output, indent=2 if pretty else None)


def parse_rich_state(json_str: str) -> dict:
    return json.loads(json_str)
