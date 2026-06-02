"""
state/support_signals.py

Emotional support-specific signal estimation.

These are the signals an emotionally supportive robot needs most:
  distress_level         : composite 0-1 — how distressed is this person RIGHT NOW
  genuine_smile          : bool + score — Duchenne vs social/forced smile
  emotional_suppression  : person hiding distress behind forced positive affect
  overwhelm_probability  : acute high-arousal distress (panic, flooding)
  comfort_seeking        : behavioral cues person wants/needs comfort
  emotional_withdrawal   : person shutting down, disengaging
  openness_to_support    : is the person receptive to engagement right now

Scientific basis:
  - Genuine smile: Ekman's Duchenne marker (AU6 + AU12, Frank et al. 1993)
  - Suppression: high AU15/AU4/tension + forced AU12 (Gross & Levenson 1993)
  - Overwhelm: high arousal + negative valence + loss of regulation signals
  - Withdrawal: gaze aversion + low expression + stillness (Coyne et al. 1990)
  - Comfort-seeking: self-touch, leaning in, gaze seeking (Bowlby 1988)
  - Openness: reciprocity + eye contact + forward lean (Mehrabian 1972)

Design: pure function — no state needed. Call once per frame.
"""

import numpy as np
from geometry.math_utils import clamp


def estimate_support_signals(
    aus:         dict,
    eye_t:       dict,
    head_t:      dict,
    gaze_t:      dict,
    mouth_t:     dict,
    brow_f:      dict,
    face_f:      dict,
    affective:   dict,
    gru_state:   dict,
    body_feats:  dict,
    hand_feats:  dict,
    face_temp:   dict,
    dynamics:    dict,
) -> dict:
    """
    Estimate support-specific signals from all available feature streams.

    Returns:
        {
          "distress_level"        : float  [0,1]
          "distress_type"         : str    "overwhelm"|"sadness"|"anxiety"|"anger"|"none"
          "genuine_smile"         : bool
          "genuine_smile_score"   : float  [0,1]
          "emotional_suppression" : float  [0,1]
          "overwhelm_probability" : float  [0,1]
          "comfort_seeking"       : float  [0,1]
          "emotional_withdrawal"  : float  [0,1]
          "openness_to_support"   : float  [0,1]
          "support_urgency"       : str    "immediate"|"soon"|"monitor"|"none"
        }
    """
    # ── Pull key signals ──────────────────────────────────────────────────────
    au4   = float(aus.get("AU4",  0.0))  # brow lower
    au6   = float(aus.get("AU6",  0.0))  # cheek raiser (Duchenne)
    au7   = float(aus.get("AU7",  0.0))  # lid tightener
    au12  = float(aus.get("AU12", 0.0))  # lip corner puller
    au15  = float(aus.get("AU15", 0.0))  # lip corner depressor
    au17  = float(aus.get("AU17", 0.0))  # chin raiser
    au23  = float(aus.get("AU23", 0.0))  # lip tightener
    au24  = float(aus.get("AU24", 0.0))  # lip pressor
    au43  = float(aus.get("AU43", 0.0))  # eye closure
    facial_tension = float(aus.get("facial_tension", 0.0))
    expr_intensity = float(aus.get("expression_intensity", 0.0))

    blink_rate  = float(eye_t.get("blink_rate",    0.0))
    blink_trend = float(eye_t.get("blink_trend",   0.0))
    avg_ear     = float(eye_t.get("avg_EAR",       0.28))
    ear_std     = float(eye_t.get("ear_std",       0.0))

    head_pitch  = float(head_t.get("pitch",        0.0))
    head_stab   = float(head_t.get("head_stability",1.0))
    attn_dir    = head_t.get("attention_direction","unknown")
    nod         = bool(head_t.get("nod_detected",  False))

    ec_rate     = float(gaze_t.get("eye_contact_rate",  0.0))
    gaze_vol    = float(gaze_t.get("gaze_volatility",   0.0))
    gaze_pat    = gaze_t.get("gaze_pattern", "unknown")

    speaking    = float(mouth_t.get("speaking_activity", 0.0))
    yawning     = bool(mouth_t.get("yawning",            False))
    lip_tens    = float(mouth_t.get("lip_tension",       0.0))
    smile       = float(mouth_t.get("smile_intensity",   0.0))

    brow_lower  = float(brow_f.get("brow_lower",    0.0))
    brow_tension= float(brow_f.get("brow_tension",  0.0))
    brow_asym   = float(brow_f.get("brow_asymmetry",0.0))

    face_dist   = float(face_f.get("face_distance_cm",  60.0))
    frame_disp  = float(face_f.get("frame_displacement", 0.0))

    valence_geom= float(affective.get("valence_geom",  0.0))
    arousal_geom= float(affective.get("arousal_geom",  0.0))
    gru_valence = float(gru_state.get("gru_valence",   0.0))
    gru_arousal = float(gru_state.get("gru_arousal",   0.0))
    momentum    = float(gru_state.get("emotional_momentum", 0.0))

    lean_dir    = body_feats.get("leaning_direction", "center")
    lean_int    = float(body_feats.get("leaning_intensity", 0.0))

    self_touch  = float(hand_feats.get("self_touch_behavior", 0.0))
    hand_face   = bool(hand_feats.get("hand_near_face", False))

    restless    = float(face_temp.get("restlessness_score", 0.0))
    still       = bool(face_temp.get("prolonged_stillness", False))
    fidget      = float(face_temp.get("fidget_probability", 0.0) or 0.0)

    stress_esc  = bool(dynamics.get("stress_escalating",    False))
    trans_occ   = bool(dynamics.get("transition_occurred",  False))

    # ── 1. Genuine smile (Duchenne marker) ────────────────────────────────────
    # Genuine: AU6 (cheek raiser) co-occurs strongly with AU12 (lip corner)
    # Forced:  AU12 alone, or AU12 + tension signals
    duchenne_score = au6 * au12                  # 0 if either absent
    forced_smile   = au12 * (1.0 - au6) * (facial_tension * 0.5 + brow_lower * 0.5)

    genuine_smile       = duchenne_score > 0.15
    genuine_smile_score = clamp(duchenne_score * 2.0, 0.0, 1.0)

    # ── 2. Emotional suppression ──────────────────────────────────────────────
    # Masking negative affect with positive display:
    # High tension/AU4/AU15 BUT also showing AU12 → suppression
    negative_substrate = clamp((au4 + au15 + brow_tension + lip_tens) / 4.0, 0.0, 1.0)
    positive_display   = au12 * (1.0 - au6)    # smile without genuine cheek raise
    suppression = clamp(negative_substrate * positive_display * 2.5, 0.0, 1.0)

    # ── 3. Overwhelm probability ──────────────────────────────────────────────
    # Acute flooding: high arousal + negative valence + loss of regulation
    # Markers: AU7 high + gaze volatile + blink irregular + rapid momentum
    negative_high_arousal = clamp(
        (-gru_valence + 1.0) / 2.0 * gru_arousal, 0.0, 1.0
    )
    dysregulation = clamp(
        0.30 * gaze_vol
      + 0.25 * momentum
      + 0.20 * restless
      + 0.15 * clamp(blink_rate / 30.0 - 0.5, 0.0, 1.0)   # blink rate > 15
      + 0.10 * ear_std / 0.05   # EAR instability
    , 0.0, 1.0)
    overwhelm = clamp(negative_high_arousal * 0.6 + dysregulation * 0.4, 0.0, 1.0)

    # ── 4. Emotional withdrawal ───────────────────────────────────────────────
    # Shutting down: low expression + gaze avoidant + still + head down
    head_down = clamp(head_pitch / 20.0, 0.0, 1.0)        # positive pitch = looking down
    withdrawal = clamp(
        0.30 * float(gaze_pat == "avoidant")
      + 0.20 * (1.0 - ec_rate)
      + 0.20 * (1.0 - expr_intensity)                      # flat affect
      + 0.15 * head_down
      + 0.10 * float(still)
      + 0.05 * (1.0 - float(speaking))
    , 0.0, 1.0)

    # ── 5. Comfort-seeking ────────────────────────────────────────────────────
    # Person behaviorally signaling need for comfort:
    # Self-touch, leaning in, seeking eye contact after aversion, fidgeting
    lean_in  = float(lean_dir == "forward") * lean_int
    comfort_seeking = clamp(
        0.30 * self_touch
      + 0.25 * float(hand_face)
      + 0.20 * fidget
      + 0.15 * lean_in
      + 0.10 * (1.0 - ec_rate) * overwhelm   # avoiding gaze when distressed
    , 0.0, 1.0)

    # ── 6. Distress level (composite) ─────────────────────────────────────────
    # Core robot signal: how much does this person need support right now?
    blink_stress = clamp((blink_rate - 15.0) / 15.0, 0.0, 1.0)
    distress_raw = clamp(
        0.30 * ((-gru_valence + 1.0) / 2.0)   # negative valence
      + 0.20 * facial_tension
      + 0.15 * brow_tension
      + 0.10 * blink_stress
      + 0.10 * suppression
      + 0.10 * float(stress_esc)
      + 0.05 * overwhelm
    , 0.0, 1.0)

    # ── 7. Distress type ──────────────────────────────────────────────────────
    # Classify which type — robot needs different responses for each
    if distress_raw < 0.15:
        distress_type = "none"
    elif overwhelm > 0.55:
        distress_type = "overwhelm"
    elif gru_arousal > 0.6 and au23 > 0.3 and au4 > 0.3:
        distress_type = "anger"
    elif gru_arousal > 0.55 and gaze_vol > 0.4:
        distress_type = "anxiety"
    elif withdrawal > 0.5 and gru_valence < -0.2:
        distress_type = "sadness"
    else:
        distress_type = "anxiety"   # default uncertain distress type

    # ── 8. Openness to support ────────────────────────────────────────────────
    # Will this person accept/benefit from engagement right now?
    # Low when: overwhelming, withdrawn, or turning away
    # High when: seeking, facing forward, moderate distress
    openness = clamp(
        0.35 * ec_rate
      + 0.20 * (1.0 - withdrawal)
      + 0.20 * float(not overwhelm > 0.7)   # not flooded
      + 0.15 * (1.0 - float(attn_dir == "away"))
      + 0.10 * float(nod)                    # nodding = responsive
    , 0.0, 1.0)
    # If person is comfort-seeking AND distressed → high openness
    if comfort_seeking > 0.4 and distress_raw > 0.3:
        openness = clamp(openness + 0.2, 0.0, 1.0)

    # ── 9. Support urgency ────────────────────────────────────────────────────
    if distress_raw > 0.70 or overwhelm > 0.70 or stress_esc and distress_raw > 0.50:
        urgency = "immediate"
    elif distress_raw > 0.45 or (stress_esc and distress_raw > 0.30):
        urgency = "soon"
    elif distress_raw > 0.20:
        urgency = "monitor"
    else:
        urgency = "none"

    return {
        "distress_level"       : round(distress_raw,          3),
        "distress_type"        : distress_type,
        "genuine_smile"        : genuine_smile,
        "genuine_smile_score"  : round(genuine_smile_score,   3),
        "emotional_suppression": round(suppression,           3),
        "overwhelm_probability": round(overwhelm,             3),
        "comfort_seeking"      : round(comfort_seeking,       3),
        "emotional_withdrawal" : round(withdrawal,            3),
        "openness_to_support"  : round(openness,              3),
        "support_urgency"      : urgency,
    }
