"""
features/affective_embedding.py

Continuous affective embedding from facial action units and geometry.

Produces a dense 48-dimensional affective feature vector that represents
the face's expressive state in a continuous space — NOT a discrete label.

The embedding is built from:
  - AU intensities (AU4, AU6, AU7, AU12, AU15, AU23, AU24, AU26, AU43)
  - Geometric ratios (EAR, MAR, brow-eye distance, lip corner angle)
  - Head pose angles (yaw, pitch, roll normalized)
  - Temporal smoothed versions of the above

This gives a rich continuous representation that can be:
  - Fed into the GRU for temporal modeling
  - Used directly for valence/arousal estimation
  - Tracked over time for emotional trajectory

Valence/arousal are estimated via a linear projection trained on
the circumplex model of affect (Russell, 1980):
  - Valence: positive correlation with AU12, AU6; negative with AU4, AU15
  - Arousal: positive with AU7, AU26, AU23; negative with low EAR

This is analytically derived, not learned — scientifically grounded,
requires no training data, and works in realtime.

INPUT:
  aus      : dict from features/action_units
  eye_f    : dict from features/eye_features
  mouth_f  : dict from features/mouth_features
  brow_f   : dict from features/brow_features
  head_t   : dict from temporal/head_pose_temporal

OUTPUT:
  dict — see extract_affective_embedding() docstring.
"""

import numpy as np
from geometry.math_utils import clamp, linear_map

# ── Circumplex valence/arousal projection weights ─────────────────────────────
# Based on published AU-emotion-dimension mappings
# (Ekman 1978, Russell 1980, Mehrabian 1996, Posner 2005)
#
# Valence weights per AU (positive = increases valence)
_VALENCE_W = {
    "AU12":  0.45,   # Lip Corner Puller → happiness → positive
    "AU6":   0.30,   # Cheek Raiser (Duchenne) → genuine positive
    "AU4":  -0.30,   # Brow Lowerer → anger/sadness → negative
    "AU15": -0.35,   # Lip Corner Depressor → sadness → negative
    "AU17": -0.15,   # Chin Raiser → sadness → slightly negative
    "AU7":  -0.10,   # Lid Tightener → threat signal → slightly negative
    "AU23": -0.10,   # Lip Tightener → anger → slightly negative
    "AU26":  0.05,   # Jaw Drop → surprise (neutral valence)
}

# Arousal weights per AU (positive = increases arousal/activation)
_AROUSAL_W = {
    "AU7":   0.40,   # Lid Tightener → vigilance/threat → high arousal
    "AU26":  0.35,   # Jaw Drop → surprise → high arousal
    "AU4":   0.20,   # Brow Lowerer → concentrated effort → moderate arousal
    "AU23":  0.25,   # Lip Tightener → tension → moderate arousal
    "AU43":  0.30,   # Eye Closure → fatigue → low arousal (inverted below)
    "AU12":  0.15,   # Happiness → moderate-high arousal
    "AU6":   0.10,   # Cheek Raiser → pleasant activation
}


def extract_affective_embedding(
    aus:    dict,
    eye_f:  dict,
    mouth_f:dict,
    brow_f: dict,
    head_t: dict,
) -> dict:
    """
    Extract continuous affective embedding.

    Args:
        aus     : dict from features/action_units.estimate_action_units()
        eye_f   : dict from features/eye_features.extract_eye_features()
        mouth_f : dict from features/mouth_features.extract_mouth_features()
        brow_f  : dict from features/brow_features.extract_brow_features()
        head_t  : dict from temporal/head_pose_temporal.HeadPoseTracker.update()

    Returns:
        {
          "embedding"       : list[float]  # 48-dim vector
          "valence_geom"    : float        # [-1, 1] geometry-based valence
          "arousal_geom"    : float        # [0, 1]  geometry-based arousal
          "dominance_geom"  : float        # [0, 1]  geometry-based dominance
          "affect_quadrant" : str          # "HVHA"|"HVLA"|"LVHA"|"LVLA"
          "expression_conf" : float        # how expressive the face is [0,1]
        }
    """
    # ── Extract raw AU values ─────────────────────────────────────────────────
    au_vals = {
        "AU4":  float(aus.get("AU4",  0.0)),
        "AU6":  float(aus.get("AU6",  0.0)),
        "AU7":  float(aus.get("AU7",  0.0)),
        "AU12": float(aus.get("AU12", 0.0)),
        "AU15": float(aus.get("AU15", 0.0)),
        "AU17": float(aus.get("AU17", 0.0)),
        "AU23": float(aus.get("AU23", 0.0)),
        "AU24": float(aus.get("AU24", 0.0)),
        "AU26": float(aus.get("AU26", 0.0)),
        "AU43": float(aus.get("AU43", 0.0)),
    }

    # ── Geometric features ────────────────────────────────────────────────────
    avg_ear     = float(eye_f.get("avg_EAR",        0.28))
    mar         = float(mouth_f.get("MAR",          0.12))
    smile       = float(mouth_f.get("smile_intensity", 0.0))
    lip_tension = float(mouth_f.get("lip_tension",  0.0))
    jaw_drop    = float(mouth_f.get("jaw_drop",     0.0))
    brow_raise  = float(brow_f.get("brow_raise",    0.0))
    brow_lower  = float(brow_f.get("brow_lower",    0.0))
    brow_tension= float(brow_f.get("brow_tension",  0.0))
    brow_asym   = float(brow_f.get("brow_asymmetry",0.0))

    yaw   = float(head_t.get("yaw",   0.0)) / 45.0   # normalize to [-1,1]
    pitch = float(head_t.get("pitch", 0.0)) / 30.0
    roll  = float(head_t.get("roll",  0.0)) / 30.0
    head_stab = float(head_t.get("head_stability", 1.0))

    # ── Valence estimation (circumplex projection) ────────────────────────────
    valence_raw = 0.0
    for au, w in _VALENCE_W.items():
        valence_raw += au_vals.get(au, 0.0) * w

    # Geometric corrections
    valence_raw += smile * 0.20               # smile adds valence directly
    valence_raw -= lip_tension * 0.15         # tension reduces valence
    valence_raw -= clamp(abs(yaw) - 0.2, 0.0, 0.8) * 0.10  # turning away = lower valence

    # Map to [-1, 1]
    valence_geom = clamp(valence_raw, -1.0, 1.0)

    # ── Arousal estimation ────────────────────────────────────────────────────
    arousal_raw = 0.0
    for au, w in _AROUSAL_W.items():
        if au == "AU43":
            # Eye closure reduces arousal
            arousal_raw -= au_vals.get(au, 0.0) * w
        else:
            arousal_raw += au_vals.get(au, 0.0) * w

    # EAR: wide eyes = high arousal, drooping = low
    ear_arousal = linear_map(avg_ear, 0.20, 0.38)   # 0=drowsy, 1=wide-eyed
    arousal_raw += (ear_arousal - 0.5) * 0.25

    arousal_geom = clamp(arousal_raw, 0.0, 1.0)

    # ── Dominance estimation ──────────────────────────────────────────────────
    # High dominance: head up, direct gaze, relaxed brows, open shoulders
    head_up      = clamp(1.0 - abs(float(head_t.get("pitch", 0.0))) / 30.0, 0.0, 1.0)
    dominance_geom = clamp(
        0.35 * head_up
      + 0.25 * (1.0 - brow_tension)
      + 0.20 * head_stab
      + 0.20 * (1.0 - brow_lower)
    , 0.0, 1.0)

    # ── Affect quadrant ───────────────────────────────────────────────────────
    # Valence > 0 = High Valence (HV), < 0 = Low Valence (LV)
    # Arousal > 0.45 = High Arousal (HA), < 0.45 = Low Arousal (LA)
    v_label = "HV" if valence_geom > 0 else "LV"
    a_label = "HA" if arousal_geom > 0.45 else "LA"
    quadrant = f"{v_label}{a_label}"

    # ── Expression confidence ──────────────────────────────────────────────────
    # How expressive is the face right now?
    expression_conf = clamp(
        float(aus.get("expression_intensity", 0.0)) * 0.6
      + abs(valence_geom) * 0.4
    , 0.0, 1.0)

    # ── Build 48-dim embedding ────────────────────────────────────────────────
    # Organized as named groups for interpretability:
    # [0:10]  AU intensities
    # [10:20] geometric features
    # [20:23] head pose normalized
    # [23:26] valence/arousal/dominance
    # [26:30] temporal stability features
    # [30:38] AU pair interactions (products)
    # [38:48] nonlinear transformations

    au_vec = [
        au_vals["AU4"], au_vals["AU6"], au_vals["AU7"],
        au_vals["AU12"], au_vals["AU15"], au_vals["AU17"],
        au_vals["AU23"], au_vals["AU24"], au_vals["AU26"], au_vals["AU43"],
    ]

    geom_vec = [
        avg_ear, mar, smile, lip_tension, jaw_drop,
        brow_raise, brow_lower, brow_tension, brow_asym,
        float(aus.get("facial_tension", 0.0)),
    ]

    pose_vec = [
        clamp(yaw,   -1.0, 1.0),
        clamp(pitch, -1.0, 1.0),
        clamp(roll,  -1.0, 1.0),
    ]

    affect_vec = [
        (valence_geom + 1.0) / 2.0,   # map to [0,1] for uniformity
        arousal_geom,
        dominance_geom,
    ]

    temporal_vec = [
        head_stab,
        float(head_t.get("attention_score",    0.0)),
        float(head_t.get("nod_detected",       False)),
        float(head_t.get("shake_detected",     False)),
    ]

    # AU pair interactions — captures co-occurring AUs
    # e.g., AU6 * AU12 = Duchenne smile (genuine happiness signal)
    pairs = [
        au_vals["AU6"]  * au_vals["AU12"],   # Duchenne smile
        au_vals["AU4"]  * au_vals["AU7"],    # threat/anger
        au_vals["AU4"]  * au_vals["AU15"],   # sadness
        au_vals["AU23"] * au_vals["AU7"],    # anger
        au_vals["AU26"] * au_vals["AU7"],    # surprise-fear
        au_vals["AU12"] * (1 - au_vals["AU4"]),  # pure smile (no brow lower)
        brow_raise * jaw_drop,               # surprise
        lip_tension * brow_tension,          # stress compound
    ]

    # Nonlinear: sigmoid-transformed affect signals for richer representation
    def sig(x, k=5.0): return float(1.0 / (1.0 + np.exp(-k * x)))

    nonlinear = [
        sig(valence_geom),
        sig(arousal_geom - 0.5),
        sig(dominance_geom - 0.5),
        np.sqrt(max(au_vals["AU12"], 0)),        # compressed smile
        np.sqrt(max(au_vals["AU4"],  0)),        # compressed brow lower
        np.sqrt(max(arousal_geom,    0)),        # compressed arousal
        float(np.tanh(valence_geom * 2.0)),      # saturated valence
        expression_conf,
        clamp(1.0 - au_vals["AU43"], 0.0, 1.0), # eye openness complement
        clamp(avg_ear / 0.35,        0.0, 1.0), # normalized EAR
    ]

    embedding = (
        au_vec +         # 10
        geom_vec +       # 10
        pose_vec +       # 3
        affect_vec +     # 3
        temporal_vec +   # 4
        pairs +          # 8
        nonlinear        # 10
    )  # total = 48

    return {
        "embedding":      [round(float(x), 5) for x in embedding],
        "valence_geom":   round(valence_geom,   4),
        "arousal_geom":   round(arousal_geom,   4),
        "dominance_geom": round(dominance_geom, 4),
        "affect_quadrant":quadrant,
        "expression_conf":round(expression_conf,3),
    }
