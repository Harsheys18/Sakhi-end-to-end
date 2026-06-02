"""
state/derived_state.py

Extended derived social-state dimensions, now incorporating body pose,
hand signals, and fidget probability.
"""

import numpy as np
from geometry.math_utils import clamp, linear_map


def compute_derived_state(
    base_state:  dict,
    social_temp: dict,
    brow_f:      dict,
    aus:         dict,
    gaze_t:      dict,
    head_t:      dict,
    face_t:      dict,
    eye_t:       dict,
    body_feats:  dict | None = None,
    body_t:      dict | None = None,
    hand_feats:  dict | None = None,
    hand_t:      dict | None = None,
) -> dict:
    """
    Compute all extended derived dimensions.

    New args vs previous version:
        body_feats : dict from features/body_pose
        body_t     : dict from temporal/body_temporal
        hand_feats : dict from features/hand_features
        hand_t     : dict from temporal/hand_temporal
    """
    body_feats = body_feats or {}
    body_t     = body_t     or {}
    hand_feats = hand_feats or {}
    hand_t     = hand_t     or {}

    # ── Pull base signals ─────────────────────────────────────────────────────
    discomfort       = base_state.get("discomfort",       {}).get("value", 0.0)
    engagement       = base_state.get("engagement",       {}).get("value", 0.0)
    confidence_level = base_state.get("confidence_level", {}).get("value", 0.0)
    valence          = base_state.get("valence",          {}).get("value", 0.5)
    arousal          = base_state.get("arousal",          {}).get("value", 0.0)
    stress           = base_state.get("stress_hints",     {}).get("value", 0.0)
    attention        = base_state.get("attention",        {}).get("value", 0.0)
    iw               = base_state.get("interaction_willingness", {}).get("value", 0.0)
    overall_conf     = base_state.get("meta", {}).get("overall_confidence", 0.0)

    brow_raise  = brow_f.get("brow_raise",      0.0)
    brow_asym   = brow_f.get("brow_asymmetry",  0.0)
    brow_tension= brow_f.get("brow_tension",    0.0)
    expr_inten  = aus.get("expression_intensity", 0.0)
    au12        = aus.get("AU12", 0.0)

    gaze_vol    = gaze_t.get("gaze_volatility",   0.0)
    ec_rate     = gaze_t.get("eye_contact_rate",  0.0)
    fix_active  = gaze_t.get("fixation_active",   False)
    head_stab   = head_t.get("head_stability",    1.0)
    head_pitch  = abs(head_t.get("pitch",         0.0))
    attn_dir    = head_t.get("attention_direction","unknown")
    nod         = head_t.get("nod_detected",      False)

    blink_rate  = eye_t.get("blink_rate",    0.0)
    ear_mean    = eye_t.get("ear_mean",      0.28)

    etr         = social_temp.get("emotion_transition_rate", 0.0)
    emo_stab    = social_temp.get("emotional_stability",     1.0)

    # Body signals
    spine_angle = body_feats.get("spine_angle",       0.0)
    neck_angle  = body_feats.get("neck_angle",        0.0)
    shoulder_op = body_feats.get("shoulder_openness", 0.0)
    body_sym    = body_feats.get("body_symmetry",     0.0)
    lean_intens = body_feats.get("leaning_intensity", 0.0)
    pose_active = body_feats.get("pose_model_active", False)

    fidget      = body_t.get("fidget_probability",  0.0)
    pose_stab   = body_t.get("pose_stability",      1.0)
    move_energy = body_t.get("movement_energy",     0.0)
    move_var    = body_t.get("movement_variability",0.0)
    restless    = body_t.get("restlessness_score",  0.0)

    # Hand signals
    crossed     = hand_feats.get("crossed_arms_probability", 0.0)
    self_touch  = hand_feats.get("self_touch_behavior",      0.0)
    hand_face   = hand_feats.get("hand_near_face",           False)
    gest_freq   = hand_t.get("gesture_frequency",            0.0)
    face_freq   = hand_t.get("hand_to_face_frequency",       0.0)

    # ── Derived dimensions ────────────────────────────────────────────────────

    comfort = clamp(
        0.45 * (1.0 - discomfort)
      + 0.20 * (valence - 0.5) * 2.0
      + 0.15 * ec_rate
      + 0.10 * (1.0 - stress)
      + 0.10 * (1.0 - crossed) * 0.5
    , 0.0, 1.0)

    head_tilt  = clamp(abs(head_t.get("roll", 0.0)) / 20.0, 0.0, 1.0)
    curiosity  = clamp(
        0.30 * brow_raise
      + 0.20 * clamp(gaze_vol * 1.5, 0.0, 1.0)
      + 0.15 * float(fix_active)
      + 0.15 * head_tilt
      + 0.10 * float(nod)
      + 0.10 * clamp(gest_freq / 10.0, 0.0, 1.0)
    , 0.0, 1.0)
    if attn_dir == "away":
        curiosity *= 0.5

    resting_blink = 15.0
    blink_dev  = clamp(abs(blink_rate - resting_blink) / 10.0, 0.0, 1.0)
    cog_load   = clamp(
        0.25 * brow_tension
      + 0.20 * blink_dev
      + 0.20 * gaze_vol
      + 0.15 * (1.0 - expr_inten)
      + 0.10 * stress
      + 0.10 * fidget
    , 0.0, 1.0)

    social_open = clamp(
        0.25 * ec_rate
      + 0.20 * au12
      + 0.15 * (1.0 - discomfort)
      + 0.15 * engagement
      + 0.15 * shoulder_op if pose_active else 0.10
      + 0.10 * (1.0 - crossed)
    , 0.0, 1.0)

    uncertainty = clamp(
        0.25 * gaze_vol
      + 0.20 * brow_asym
      + 0.20 * (1.0 - head_stab)
      + 0.15 * brow_raise * 0.5
      + 0.20 * fidget
    , 0.0, 1.0)

    head_up   = clamp(1.0 - head_pitch / 30.0, 0.0, 1.0)
    dominance = clamp(
        0.25 * confidence_level
      + 0.25 * ec_rate
      + 0.20 * head_up
      + 0.15 * (1.0 - brow_tension)
      + 0.15 * (1.0 - lean_intens) * 0.5
    , 0.0, 1.0)

    emo_intensity = clamp((expr_inten + arousal) / 2.0, 0.0, 1.0)

    # Visual energy: arousal + body movement
    norm_move  = clamp(move_energy / 0.3, 0.0, 1.0) if pose_active else \
                 clamp(face_t.get("movement_energy", 0.0) / 10.0, 0.0, 1.0)
    visual_energy = clamp(arousal * 0.55 + norm_move * 0.30 + fidget * 0.15, 0.0, 1.0)

    def _dim(val, conf=None):
        return {"value": round(float(val), 3), "confidence": round(float(conf or overall_conf), 3)}

    result = {
        "comfort_estimate":          _dim(comfort),
        "curiosity_estimate":        _dim(curiosity),
        "cognitive_load_estimate":   _dim(cog_load),
        "social_openness_estimate":  _dim(social_open),
        "emotional_stability":       _dim(emo_stab),
        "uncertainty_estimate":      _dim(uncertainty),
        "dominance":                 _dim(dominance),
        "emotional_intensity":       _dim(emo_intensity),
        "emotion_transition_rate":   _dim(etr),

        # Temporal pass-throughs
        "engagement_trend":              social_temp.get("engagement_trend",           0.0),
        "attention_trend":               social_temp.get("attention_trend",            0.0),
        "gaze_trend":                    social_temp.get("gaze_trend",                 0.0),
        "sustained_attention":           social_temp.get("sustained_attention",        False),
        "sustained_attention_ms":        social_temp.get("sustained_attention_ms",     0.0),
        "sustained_disengagement":       social_temp.get("sustained_disengagement",    False),
        "sustained_disengagement_ms":    social_temp.get("sustained_disengagement_ms", 0.0),
        "attention_stability":           social_temp.get("attention_stability",        1.0),
        "interaction_stability":         social_temp.get("interaction_stability",      1.0),
        "prolonged_downward_gaze":       social_temp.get("prolonged_downward_gaze",    False),

        # Face movement (from face_temporal)
        "face_movement_energy":          face_t.get("movement_energy",      0.0),
        "face_movement_variability":     face_t.get("movement_variability", 0.0),
        "movement_trend":                face_t.get("movement_trend",       0.0),
        "face_stability":                face_t.get("face_stability",       1.0),
        "prolonged_stillness":           body_t.get("prolonged_stillness",
                                         face_t.get("prolonged_stillness", False)),
        "restlessness_score":            restless if pose_active else face_t.get("restlessness_score", 0.0),

        # Body
        "fidget_probability":            round(fidget,    3),
        "pose_stability":                round(pose_stab, 3),
        "movement_energy":               round(move_energy if pose_active
                                              else face_t.get("movement_energy", 0.0), 4),
        "movement_variability":          round(move_var if pose_active
                                              else face_t.get("movement_variability", 0.0), 3),

        # Hand
        "gesture_frequency":             round(gest_freq,  2),
        "hand_to_face_frequency":        round(face_freq,  2),

        # LLM-final
        "visual_energy":                 round(visual_energy, 3),
        "visual_social_openness":        round(social_open,   3),
        "visual_interest":               round(curiosity,     3),
        "system_confidence":             round(overall_conf,  3),
    }
    return result
