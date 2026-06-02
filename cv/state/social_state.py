"""
state/social_state.py

Social state estimator — the final fusion layer.

Fuses all temporal feature signals into a structured social state estimate
that is ready to be serialized to JSON and consumed by an LLM.

Dimensions estimated:
  engagement              : sustained, active attention toward the interaction
  attention               : directional focus (on camera / on robot)
  stress_hints            : physiological / behavioral tension signals
  fatigue                 : tiredness indicators (blink rate, eye closure, yawning)
  confidence_level        : postural / behavioral self-assurance signals
  discomfort              : avoidance, gaze aversion, closed posture signals
  valence                 : positive (pleasant) vs negative (unpleasant) affect
  arousal                 : activation level (calm ↔ excited/alert)
  interaction_willingness : whether the person is ready/open to engage

Each dimension:
  - value      : float [0, 1]
  - confidence : float [0, 1]  ← how reliable this estimate is right now

The fusion uses weighted linear combinations and heuristic rules.
No machine learning. No magic constants that can't be understood.

INPUT:
  eye_t    : dict from temporal/eye_temporal.BlinkTracker.update()
  head_t   : dict from temporal/head_pose_temporal.HeadPoseTracker.update()
  gaze_t   : dict from temporal/gaze_temporal.GazeTracker.update()
  mouth_t  : dict from temporal/mouth_temporal.MouthTracker.update()
  brow_f   : dict from features/brow_features.extract_brow_features()
  aus      : dict from features/action_units.estimate_action_units()
  face_f   : dict from features/face_features.extract_face_features()

OUTPUT:
  dict — see estimate_social_state() docstring.
"""

import numpy as np
from geometry.math_utils import clamp, linear_map


def estimate_social_state(
    eye_t:    dict,
    head_t:   dict,
    gaze_t:   dict,
    mouth_t:  dict,
    brow_f:   dict,
    aus:      dict,
    face_f:   dict,
    baseline: "PersonalBaselineTracker | None" = None,
) -> dict:
    """
    Fuse all temporal feature streams into a social state estimate.

    Args:
        eye_t   : BlinkTracker output
        head_t  : HeadPoseTracker output
        gaze_t  : GazeTracker output
        mouth_t : MouthTracker output
        brow_f  : brow_features output (per-frame; low latency)
        aus     : action_units output
        face_f  : face_features output

    Returns:
        {
          "face_present"       : bool
          "engagement"         : {"value": float, "confidence": float}
          "attention"          : {"value": float, "direction": str, "confidence": float}
          "stress_hints"       : {"value": float, "confidence": float}
          "fatigue"            : {"value": float, "confidence": float}
          "confidence_level"   : {"value": float, "confidence": float}
          "discomfort"         : {"value": float, "confidence": float}
          "valence"            : {"value": float, "confidence": float}
          "arousal"            : {"value": float, "confidence": float}
          "interaction_willingness": {"value": float, "confidence": float}
          "speaking"           : bool
          "eye_contact"        : bool
          "meta"               : {
              "overall_confidence": float,
              "blink_rate"        : float,
              "head_direction"    : str,
              "gaze_pattern"      : str,
          }
        }
    """
    face_present = face_f.get("face_presence", False)

    if not face_present:
        return _absent_state()

    # ── Pull signal values ────────────────────────────────────────────────────

    # Eye / blink
    blink_rate          = eye_t.get("blink_rate",          0.0)
    avg_ear             = eye_t.get("avg_EAR",             0.0)
    blink_trend         = eye_t.get("blink_trend",         0.0)
    eye_open_dur        = eye_t.get("eye_open_duration_ms",0.0)
    eye_conf            = eye_t.get("confidence",          0.0)

    # Head pose
    head_attn           = head_t.get("attention_score",    0.0)
    head_dir            = head_t.get("attention_direction","unknown")
    head_stability      = head_t.get("head_stability",     1.0)
    nod_detected        = head_t.get("nod_detected",       False)
    head_yaw            = abs(head_t.get("yaw",            0.0))
    head_pitch          = abs(head_t.get("pitch",          0.0))
    head_conf           = head_t.get("confidence",         0.0)

    # Gaze
    gaze_attn           = gaze_t.get("eye_contact_score",    0.0)
    gaze_ec_rate        = gaze_t.get("eye_contact_rate",     0.0)
    gaze_volatility     = gaze_t.get("gaze_volatility",      0.0)
    gaze_pattern        = gaze_t.get("gaze_pattern",         "unknown")
    fix_active          = gaze_t.get("fixation_active",      False)
    gaze_conf           = gaze_t.get("confidence",           0.0)

    # Mouth
    smile               = mouth_t.get("smile_intensity",  0.0)
    speaking            = mouth_t.get("speaking_activity", 0.0) > 0.4
    yawning             = mouth_t.get("yawning",           False)
    tension             = mouth_t.get("lip_tension",       0.0)
    tension_trend       = mouth_t.get("tension_trend",     0.0)
    mouth_conf          = mouth_t.get("confidence",        0.0)

    # Brow
    brow_raise          = brow_f.get("brow_raise",         0.0)
    brow_lower          = brow_f.get("brow_lower",         0.0)
    brow_tension_b      = brow_f.get("brow_tension",       0.0)
    brow_asym           = brow_f.get("brow_asymmetry",     0.0)
    brow_conf           = brow_f.get("confidence",         0.0)

    # Action units
    facial_tension      = aus.get("facial_tension",        0.0)
    expr_intensity      = aus.get("expression_intensity",  0.0)
    au4                 = aus.get("AU4",  0.0)   # brow lower
    au12                = aus.get("AU12", 0.0)   # smile
    au15                = aus.get("AU15", 0.0)   # lip depress
    au26                = aus.get("AU26", 0.0)   # jaw drop

    # Face geometry
    face_dist           = face_f.get("face_distance_cm",   60.0)

    # ── Overall sensor confidence ─────────────────────────────────────────────
    overall_conf = float(np.mean([
        eye_conf, head_conf, gaze_conf, mouth_conf, brow_conf
    ]))

    # ══════════════════════════════════════════════════════════════════════════
    # ENGAGEMENT
    # High engagement: forward head, eye contact, fixation, nodding, not yawning
    # ══════════════════════════════════════════════════════════════════════════
    engagement_raw = (
        0.30 * head_attn
      + 0.25 * gaze_attn
      + 0.15 * gaze_ec_rate
      + 0.15 * (1.0 - gaze_volatility)
      + 0.10 * float(fix_active)
      + 0.05 * float(not yawning)
    )
    # Penalty: yawning or head turned far away
    if yawning:
        engagement_raw *= 0.5
    if head_dir == "away":
        engagement_raw *= 0.6
    engagement = clamp(engagement_raw, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # ATTENTION
    # Combined directional focus score
    # ══════════════════════════════════════════════════════════════════════════
    attention_val = (
        0.50 * head_attn
      + 0.30 * gaze_attn
      + 0.20 * gaze_ec_rate
    )
    attention_val = clamp(attention_val, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # STRESS HINTS
    # Brow tension, lip tension, blink rate elevation, gaze instability
    # ══════════════════════════════════════════════════════════════════════════
    if baseline is not None and baseline.is_active():
        blink_stress = baseline.elevated(blink_rate, "blink_rate")
        brow_stress  = baseline.elevated(brow_tension_b, "brow_tension")
        face_stress  = baseline.elevated(facial_tension, "facial_tension")
        lip_stress   = baseline.elevated(tension, "lip_tension")
        gaze_stress  = baseline.elevated(gaze_volatility, "gaze_volatility")
        au4_stress   = baseline.elevated(au4, "AU4")
    else:
        blink_stress = clamp(linear_map(blink_rate, 15.0, 30.0), 0.0, 1.0)
        brow_stress  = brow_tension_b
        face_stress  = facial_tension
        lip_stress   = tension
        gaze_stress  = clamp(gaze_volatility, 0.0, 1.0)
        au4_stress   = au4

    stress_raw = (
        0.25 * brow_stress
      + 0.20 * face_stress
      + 0.15 * lip_stress
      + 0.15 * blink_stress
      + 0.10 * gaze_stress
      + 0.10 * clamp(tension_trend * 50.0, 0.0, 1.0)
      + 0.05 * brow_asym
    )
    stress = clamp(stress_raw, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # FATIGUE
    # Low blink rate (>25 bpm or <8 bpm can both indicate fatigue),
    # prolonged eye closure, yawning, low arousal expression
    # ══════════════════════════════════════════════════════════════════════════
    if baseline is not None and baseline.is_active():
        blink_fatigue = baseline.blink_rate_abnormal(blink_rate)
        lid_droop     = baseline.ear_droop(avg_ear)
    else:
        blink_fatigue = 0.0
        if blink_rate > 25.0:
            blink_fatigue = clamp(linear_map(blink_rate, 25.0, 40.0), 0.0, 1.0)
        elif 0 < blink_rate < 8.0:
            blink_fatigue = clamp(linear_map(blink_rate, 8.0, 2.0), 0.0, 1.0)
        lid_droop = clamp(linear_map(avg_ear, 0.28, 0.18), 0.0, 1.0) if avg_ear > 0 else 0.0

    fatigue_raw = (
        0.30 * float(yawning)
      + 0.25 * lid_droop
      + 0.20 * blink_fatigue
      + 0.15 * (1.0 - head_stability)
      + 0.10 * clamp(blink_trend / 5.0, 0.0, 1.0)   # rising blink rate trend
    )
    fatigue = clamp(fatigue_raw, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # CONFIDENCE LEVEL (behavioral)
    # Open posture signals, direct gaze, head raised, no tension
    # ══════════════════════════════════════════════════════════════════════════
    # Lower head pitch = head up = more confident
    head_up   = clamp(1.0 - head_pitch / 30.0, 0.0, 1.0)
    conf_lvl_raw = (
        0.30 * gaze_ec_rate
      + 0.25 * head_up
      + 0.20 * (1.0 - facial_tension)
      + 0.15 * head_stability
      + 0.10 * (1.0 - brow_tension_b)
    )
    confidence_level = clamp(conf_lvl_raw, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # DISCOMFORT
    # Gaze aversion, brow furrowing, lip tension, shifted posture
    # ══════════════════════════════════════════════════════════════════════════
    gaze_aversion = clamp(1.0 - gaze_attn, 0.0, 1.0)
    if baseline is not None and baseline.is_active():
        brow_disc = baseline.elevated(brow_tension_b, "brow_tension")
        au4_disc  = baseline.elevated(au4, "AU4")
        lip_disc  = baseline.elevated(tension, "lip_tension")
    else:
        brow_disc = brow_tension_b
        au4_disc  = au4
        lip_disc  = tension

    discomfort_raw = (
        0.25 * gaze_aversion * float(gaze_pattern == "avoidant")
      + 0.20 * brow_disc
      + 0.20 * au4_disc
      + 0.15 * lip_disc
      + 0.10 * au15
      + 0.10 * (1.0 - head_attn)
    )
    discomfort = clamp(discomfort_raw, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # VALENCE  (positive ↔ negative affect)
    # Smile = high valence; tension + brow lower = low valence
    # Scaled to [0, 1] where 0.5 = neutral
    # ══════════════════════════════════════════════════════════════════════════
    positive_signals = (
        0.50 * smile
      + 0.30 * float(not yawning) * 0.3
      + 0.20 * (1.0 - discomfort)
    )
    negative_signals = (
        0.40 * au15
      + 0.35 * brow_tension_b
      + 0.25 * discomfort
    )
    valence = clamp(0.5 + (positive_signals - negative_signals) * 0.5, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # AROUSAL  (calm ↔ activated/excited)
    # High arousal: wide eyes, expression intensity, rapid gaze, head movement
    # ══════════════════════════════════════════════════════════════════════════
    # Eye openness above resting (wide eyes → alert/excited)
    eye_wide = clamp(linear_map(avg_ear, 0.28, 0.40), 0.0, 1.0) if avg_ear > 0 else 0.0

    arousal_raw = (
        0.25 * eye_wide
      + 0.25 * expr_intensity
      + 0.20 * gaze_volatility
      + 0.15 * (1.0 - head_stability)
      + 0.15 * brow_raise
    )
    # Yawning suppresses arousal
    if yawning:
        arousal_raw *= 0.5
    arousal = clamp(arousal_raw, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # INTERACTION WILLINGNESS
    # Ready and open to interact: engaged, not stressed, making eye contact
    # ══════════════════════════════════════════════════════════════════════════
    iw_raw = (
        0.35 * engagement
      + 0.25 * (1.0 - discomfort)
      + 0.20 * gaze_ec_rate
      + 0.10 * head_attn
      + 0.10 * (1.0 - stress * 0.5)   # stress partially suppresses willingness
    )
    # Speaking (actively talking) → high willingness
    if speaking:
        iw_raw = clamp(iw_raw + 0.15, 0.0, 1.0)
    interaction_willingness = clamp(iw_raw, 0.0, 1.0)

    # ── Estimate confidence per dimension ────────────────────────────────────
    # Each dimension uses a subset of sensors; weight their confidences.
    def _dim_conf(*vals: float) -> float:
        return round(float(np.mean(vals)), 3)

    return {
        "face_present": True,

        "engagement": {
            "value":      round(engagement,    3),
            "confidence": _dim_conf(head_conf, gaze_conf, eye_conf),
        },
        "attention": {
            "value":      round(attention_val, 3),
            "direction":  head_dir,
            "confidence": _dim_conf(head_conf, gaze_conf),
        },
        "stress_hints": {
            "value":      round(stress,        3),
            "confidence": _dim_conf(eye_conf, brow_conf, mouth_conf),
        },
        "fatigue": {
            "value":      round(fatigue,       3),
            "confidence": _dim_conf(eye_conf, mouth_conf),
        },
        "confidence_level": {
            "value":      round(confidence_level, 3),
            "confidence": _dim_conf(gaze_conf, head_conf, brow_conf),
        },
        "discomfort": {
            "value":      round(discomfort,    3),
            "confidence": _dim_conf(gaze_conf, brow_conf, mouth_conf),
        },
        "valence": {
            "value":      round(valence,       3),
            "confidence": _dim_conf(mouth_conf, brow_conf),
        },
        "arousal": {
            "value":      round(arousal,       3),
            "confidence": _dim_conf(eye_conf, head_conf, gaze_conf),
        },
        "interaction_willingness": {
            "value":      round(interaction_willingness, 3),
            "confidence": _dim_conf(gaze_conf, head_conf, mouth_conf),
        },

        "speaking":    speaking,
        "eye_contact": gaze_ec_rate > 0.5,

        "meta": {
            "overall_confidence": round(overall_conf, 3),
            "blink_rate":         round(blink_rate,   1),
            "head_direction":     head_dir,
            "gaze_pattern":       gaze_pattern,
        },
    }


def _absent_state() -> dict:
    """Return a zero-confidence state when no face is detected."""
    dim = {"value": 0.0, "confidence": 0.0}
    return {
        "face_present":           False,
        "engagement":             dict(dim),
        "attention":              {**dim, "direction": "unknown"},
        "stress_hints":           dict(dim),
        "fatigue":                dict(dim),
        "confidence_level":       dict(dim),
        "discomfort":             dict(dim),
        "valence":                {**{"value": 0.5, "confidence": 0.0}},
        "arousal":                dict(dim),
        "interaction_willingness": dict(dim),
        "speaking":               False,
        "eye_contact":            False,
        "meta": {
            "overall_confidence": 0.0,
            "blink_rate":         0.0,
            "head_direction":     "unknown",
            "gaze_pattern":       "unknown",
        },
    }
