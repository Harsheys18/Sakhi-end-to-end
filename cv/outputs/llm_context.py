"""
outputs/llm_context.py

Compact LLM context generator for emotionally supportive robot.

v2 problem: 277-line JSON sent every frame. LLM receives too much noise,
can't quickly identify what matters, wastes context window.

v3 solution: Two output modes:

  1. COMPACT CONTEXT (default) — ~15-30 lines of structured natural language
     Designed to be injected into the robot's LLM system prompt or user turn.
     Human-readable, interpretable, LLM-native.

  2. STRUCTURED DICT — minimal JSON for programmatic use / audio team fusion.
     Only the fields that matter for support decisions.

The compact context is organized as:

  [EMOTIONAL STATE]
  [BEHAVIORAL SIGNALS]
  [SUPPORT ASSESSMENT]
  [CHANGED SINCE LAST]  ← only when something shifted
  [UNCERTAINTY]

Design: pure functions. No state.
"""

import json
from geometry.math_utils import clamp

# Compact level descriptions
def _pct(v: float) -> str:
    """Turn [0,1] float into a compact human label."""
    if v >= 0.80: return "high"
    if v >= 0.55: return "moderate"
    if v >= 0.30: return "low"
    return "minimal"

def _sign(v: float, label_pos: str, label_neg: str, threshold: float = 0.15) -> str:
    if v >  threshold: return label_pos
    if v < -threshold: return label_neg
    return "neutral"

def _urgency_emoji(u: str) -> str:
    return {"immediate": "🔴", "soon": "🟡", "monitor": "🔵", "none": "⚪"}.get(u, "⚪")


def format_llm_context(
    state:         dict,
    changed_dims:  list[tuple[str, float]] | None = None,
    emit_reason:   str = "",
    mode:          str = "compact",   # "compact" | "dict" | "both"
    include_audio_handoff: bool = True,
) -> str:
    """
    Generate LLM-ready context string.

    Args:
        state          : full merged state dict
        changed_dims   : list of (dim, delta) from EmissionController
        emit_reason    : why this emission was triggered (for LLM awareness)
        mode           : output format
        include_audio_handoff : include fusion block for audio team

    Returns:
        String ready to inject into LLM prompt.
    """
    if mode == "dict":
        return json.dumps(_build_structured_dict(state, changed_dims), indent=2)
    if mode == "both":
        ctx  = _build_compact_context(state, changed_dims, emit_reason)
        ctx += "\n\n--- STRUCTURED ---\n"
        ctx += json.dumps(_build_structured_dict(state, changed_dims))
        return ctx
    return _build_compact_context(state, changed_dims, emit_reason, include_audio_handoff)


def _build_compact_context(
    state: dict,
    changed_dims: list | None,
    emit_reason:  str,
    include_audio_handoff: bool = True,
) -> str:
    """Build the compact natural-language context block."""

    if not state.get("face_present", False):
        return "[VISUAL CONTEXT]\nStatus: No face detected. Person may have left or camera obstructed."

    # Pull all signals
    distress    = float(state.get("distress_level",        0.0))
    dist_type   = state.get("distress_type",               "none")
    urgency     = state.get("support_urgency",             "none")
    genuine_sm  = bool(state.get("genuine_smile",          False))
    genuine_sc  = float(state.get("genuine_smile_score",   0.0))
    suppression = float(state.get("emotional_suppression", 0.0))
    overwhelm   = float(state.get("overwhelm_probability", 0.0))
    comfort_sk  = float(state.get("comfort_seeking",       0.0))
    withdrawal  = float(state.get("emotional_withdrawal",  0.0))
    openness    = float(state.get("openness_to_support",   0.0))

    gru_val     = float(state.get("gru_valence",       0.0))
    gru_aro     = float(state.get("gru_arousal",       0.0))
    coherence   = float(state.get("temporal_coherence",0.0))
    quadrant    = state.get("gru_affect_quadrant",     "HVLA")
    quad_label  = state.get("quadrant_label",          "calm")

    engage      = _get_dim_val(state, "engagement")
    attn        = _get_dim_val(state, "attention")
    fatigue     = _get_dim_val(state, "fatigue")
    stress      = _get_dim_val(state, "stress_hints")
    discomfort  = _get_dim_val(state, "discomfort")

    rapport     = float(state.get("rapport_signal",           0.0))
    conv_ready  = float(state.get("conversational_readiness", 0.0))
    soc_comfort = float(state.get("social_comfort",           0.0))
    responsiv   = float(state.get("social_responsiveness",    0.0))

    ec_rate     = float(state.get("eye_contact_rate",         state.get("eye_contact", 0.0) if isinstance(state.get("eye_contact"), float) else 0.0))
    speaking    = bool(state.get("speaking",                   False))
    eye_contact = bool(state.get("eye_contact",                False))
    blink_rate  = float(state.get("blink_rate",               0.0) if not isinstance(state.get("blink_rate"), dict) else 0.0)

    stress_esc  = bool(state.get("stress_escalating",         False))
    eng_shift   = state.get("engagement_shift",               "stable")
    transition  = bool(state.get("transition_occurred",       False))
    trans_from  = state.get("transition_from",                "")
    trans_to    = state.get("transition_to",                  "")
    persist_ms  = float(state.get("quadrant_duration_ms",     0.0))
    recent_trans= int(state.get("recent_transitions",         0))

    global_unc  = float(state.get("global_uncertainty",       0.0))
    ambiguity   = float(state.get("affective_ambiguity",      0.0))

    head_dir    = state.get("meta", {}).get("head_direction",  "unknown")
    gaze_pat    = state.get("meta", {}).get("gaze_pattern",   "unknown")
    fidget      = float(state.get("fidget_probability", 0.0))

    # ── Build context block ───────────────────────────────────────────────────

    lines = ["[VISUAL EMOTIONAL CONTEXT]"]

    # ── Trigger reason (concise) ──────────────────────────────────────────────
    if emit_reason and emit_reason not in ("first_emission", "heartbeat"):
        lines.append(f"Trigger: {emit_reason}")

    # ── Support assessment ────────────────────────────────────────────────────
    emo  = _urgency_emoji(urgency)
    lines.append(f"")
    lines.append(f"SUPPORT ASSESSMENT {emo}")
    lines.append(f"  Urgency:         {urgency.upper()}")
    lines.append(f"  Distress level:  {distress:.2f} ({_pct(distress)})")
    if dist_type != "none":
        lines.append(f"  Distress type:   {dist_type}")
    lines.append(f"  Openness:        {openness:.2f} ({_pct(openness)}) — {'receptive' if openness > 0.5 else 'not receptive'}")
    if suppression > 0.25:
        lines.append(f"  ⚠ Emotional suppression detected ({suppression:.2f}) — visible distress may be masked")
    if overwhelm > 0.45:
        lines.append(f"  ⚠ Overwhelm signals ({overwhelm:.2f}) — avoid information overload")
    if comfort_sk > 0.35:
        lines.append(f"  ↑ Comfort-seeking behavior ({comfort_sk:.2f})")
    if withdrawal > 0.45:
        lines.append(f"  ↓ Emotional withdrawal ({withdrawal:.2f}) — gentle approach needed")

    # ── Emotional state ───────────────────────────────────────────────────────
    lines.append(f"")
    lines.append(f"EMOTIONAL STATE")
    lines.append(f"  Affect quadrant: {quadrant} ({quad_label})")
    lines.append(f"  Valence:         {_sign(gru_val,'positive','negative')} ({gru_val:+.2f})")
    lines.append(f"  Arousal:         {_pct(gru_aro)} ({gru_aro:.2f})")
    lines.append(f"  Signal coherence:{coherence:.2f} ({'stable read' if coherence > 0.6 else 'fluctuating'})")
    if persist_ms > 2000:
        lines.append(f"  In this state:   {persist_ms/1000:.1f}s")
    if recent_trans > 1:
        lines.append(f"  Emotional volatility: {recent_trans} transitions in 30s")

    # ── Behavioral signals ────────────────────────────────────────────────────
    lines.append(f"")
    lines.append(f"BEHAVIORAL SIGNALS")
    lines.append(f"  Engagement:      {_pct(engage)}")
    lines.append(f"  Eye contact:     {'yes' if eye_contact else 'no'} (rate: {ec_rate:.0%})")
    lines.append(f"  Head direction:  {head_dir}")
    lines.append(f"  Gaze pattern:    {gaze_pat}")
    if genuine_sm:
        lines.append(f"  Smile:           genuine (Duchenne {genuine_sc:.2f})")
    elif state.get("speaking"):
        lines.append(f"  Mouth:           speaking")
    if fatigue > 0.40:
        lines.append(f"  Fatigue:         {_pct(fatigue)}")
    if blink_rate > 0:
        lines.append(f"  Blink rate:      {blink_rate:.0f}/min {'(elevated)' if blink_rate > 22 else ''}")
    if fidget > 0.40:
        lines.append(f"  Fidgeting:       {_pct(fidget)}")

    # ── Social interaction quality ────────────────────────────────────────────
    lines.append(f"")
    lines.append(f"SOCIAL INTERACTION")
    lines.append(f"  Rapport:               {_pct(rapport)}")
    lines.append(f"  Conversational ready:  {_pct(conv_ready)}")
    lines.append(f"  Social comfort:        {_pct(soc_comfort)}")
    lines.append(f"  Responsiveness:        {_pct(responsiv)}")

    # ── What changed ─────────────────────────────────────────────────────────
    if changed_dims:
        lines.append(f"")
        lines.append(f"RECENT CHANGES")
        for dim, delta in changed_dims[:4]:   # top 4
            dir_arrow = "↑" if delta > 0 else "↓"
            lines.append(f"  {dir_arrow} {dim}: Δ{delta:.2f}")
        if transition:
            lines.append(f"  → Affect shifted: {trans_from} → {trans_to}")
        if stress_esc:
            lines.append(f"  ↑ Stress escalating")
        if eng_shift != "stable":
            lines.append(f"  {'↑' if eng_shift=='rising' else '↓'} Engagement {eng_shift}")

    # ── Uncertainty ───────────────────────────────────────────────────────────
    if global_unc > 0.35 or ambiguity > 0.50:
        lines.append(f"")
        lines.append(f"SIGNAL QUALITY")
        if global_unc > 0.35:
            lines.append(f"  Uncertainty: {global_unc:.2f} — treat estimates as approximate")
        if ambiguity > 0.50:
            lines.append(f"  Affective ambiguity: {ambiguity:.2f} — emotional state is mixed/unclear")

    # ── Audio team handoff ────────────────────────────────────────────────────
    if include_audio_handoff:
        lines.append(f"")
        lines.append(f"MULTIMODAL HANDOFF [for audio team fusion]")
        lines.append(f"  visual_distress:       {distress:.3f}")
        lines.append(f"  visual_valence:        {gru_val:.3f}")
        lines.append(f"  visual_arousal:        {gru_aro:.3f}")
        lines.append(f"  visual_openness:       {openness:.3f}")
        lines.append(f"  visual_suppression:    {suppression:.3f}")
        lines.append(f"  visual_withdrawal:     {withdrawal:.3f}")
        lines.append(f"  visual_comfort_seeking:{comfort_sk:.3f}")
        lines.append(f"  visual_signal_conf:    {1.0 - global_unc:.3f}")

    return "\n".join(lines)


def _build_structured_dict(state: dict, changed_dims: list | None) -> dict:
    """
    Minimal JSON dict for programmatic use and audio team fusion.
    Only the fields a robot needs to make support decisions.
    """
    return {
        "face_present":          state.get("face_present", False),
        "support": {
            "urgency":           state.get("support_urgency",           "none"),
            "distress_level":    round(float(state.get("distress_level",        0.0)), 3),
            "distress_type":     state.get("distress_type",             "none"),
            "openness":          round(float(state.get("openness_to_support",   0.0)), 3),
            "genuine_smile":     bool(state.get("genuine_smile",        False)),
            "suppression":       round(float(state.get("emotional_suppression", 0.0)), 3),
            "overwhelm":         round(float(state.get("overwhelm_probability", 0.0)), 3),
            "comfort_seeking":   round(float(state.get("comfort_seeking",       0.0)), 3),
            "withdrawal":        round(float(state.get("emotional_withdrawal",  0.0)), 3),
        },
        "affect": {
            "valence":           round(float(state.get("gru_valence",    0.0)), 3),
            "arousal":           round(float(state.get("gru_arousal",    0.0)), 3),
            "quadrant":          state.get("gru_affect_quadrant",        "HVLA"),
            "coherence":         round(float(state.get("temporal_coherence", 0.0)), 3),
        },
        "social": {
            "rapport":           round(float(state.get("rapport_signal",           0.0)), 3),
            "conv_ready":        round(float(state.get("conversational_readiness", 0.0)), 3),
            "social_comfort":    round(float(state.get("social_comfort",           0.0)), 3),
            "eye_contact_rate":  round(float(state.get("eye_contact_rate",         0.0) if not isinstance(state.get("eye_contact_rate"), dict) else 0.0), 3),
        },
        "dynamics": {
            "stress_escalating": bool(state.get("stress_escalating",    False)),
            "engagement_shift":  state.get("engagement_shift",          "stable"),
            "transition":        bool(state.get("transition_occurred",  False)),
        },
        "uncertainty":           round(float(state.get("global_uncertainty", 0.0)), 3),
        "changed_dims":          [{"dim": d, "delta": round(v, 3)} for d, v in (changed_dims or [])[:5]],
        # Fusion block for audio team
        "multimodal_handoff": {
            "visual_distress":        round(float(state.get("distress_level",       0.0)), 3),
            "visual_valence":         round(float(state.get("gru_valence",          0.0)), 3),
            "visual_arousal":         round(float(state.get("gru_arousal",          0.0)), 3),
            "visual_openness":        round(float(state.get("openness_to_support",  0.0)), 3),
            "visual_suppression":     round(float(state.get("emotional_suppression",0.0)), 3),
            "visual_withdrawal":      round(float(state.get("emotional_withdrawal", 0.0)), 3),
            "visual_comfort_seeking": round(float(state.get("comfort_seeking",      0.0)), 3),
            "visual_signal_conf":     round(1.0 - float(state.get("global_uncertainty", 0.0)), 3),
        },
    }


def _get_dim_val(state: dict, dim: str) -> float:
    v = state.get(dim)
    if isinstance(v, dict):
        return float(v.get("value", 0.0))
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0
