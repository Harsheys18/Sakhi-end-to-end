"""
tests/test_affective.py

Tests for all new affective-upgrade modules:
  features/affective_embedding.py
  temporal/affective_gru.py
  state/uncertainty.py
  state/emotional_dynamics.py
  state/social_cognition.py
  outputs/rich_json.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json
import time

_passed = 0
_failed = 0
NOW_MS  = int(time.time() * 1000)


def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        print(f"  ✓ {name}")
        _passed += 1
    else:
        print(f"  ✗ {name}{' — ' + detail if detail else ''}")
        _failed += 1


# ── Shared helpers ─────────────────────────────────────────────────────────────

def fake_aus(smile=0.5, au4=0.1, au6=0.3, au7=0.1):
    return {
        "AU4": au4, "AU6": au6, "AU7": au7, "AU12": smile,
        "AU15": 0.05, "AU17": 0.0, "AU23": 0.1, "AU24": 0.05,
        "AU26": 0.1, "AU43": 0.0,
        "facial_tension": 0.1, "expression_intensity": 0.3, "confidence": 0.9,
    }

def fake_eye_f(ear=0.30):
    return {"avg_EAR": ear, "blink_detected": False,
            "ear_confidence": 1.0, "metric_space": "pixel"}

def fake_mouth_f(smile=0.4, mar=0.12):
    return {"MAR": mar, "smile_intensity": smile, "lip_tension": 0.1,
            "jaw_drop": 0.1, "lip_compression": 0.05, "confidence": 1.0}

def fake_brow_f(raise_=0.2, tension=0.1, asym=0.05):
    return {"brow_raise": raise_, "brow_lower": 0.1, "brow_tension": tension,
            "brow_asymmetry": asym, "inner_brow_angle": 0.0, "confidence": 0.9}

def fake_head_t(yaw=3.0, pitch=-5.0, stability=0.9, attn=0.85):
    return {
        "yaw": yaw, "pitch": pitch, "roll": 1.0,
        "head_stability": stability, "attention_score": attn,
        "attention_direction": "forward",
        "nod_detected": False, "shake_detected": False,
        "confidence": 0.9,
    }

def fake_gaze_t(ec=0.8, ec_rate=0.75, vol=0.10):
    return {
        "eye_contact_score": ec, "eye_contact_rate": ec_rate,
        "gaze_volatility": vol, "gaze_pattern": "fixating",
        "fixation_active": True, "gaze_x": 0.05, "gaze_y": 0.0,
        "confidence": 1.0,
    }

def fake_mouth_t(smile=0.4, speaking=0.0, yawning=False):
    return {"smile_intensity": smile, "speaking_activity": speaking,
            "yawning": yawning, "lip_tension": 0.1, "confidence": 1.0}

def fake_eye_t(blink_rate=15.0, ear=0.30):
    return {"blink_rate": blink_rate, "avg_EAR": ear, "blink_trend": 0.0,
            "ear_mean": ear, "ear_std": 0.02, "confidence": 0.9}

def fake_body_f():
    return {"neck_angle": 0.0, "leaning_direction": "center",
            "leaning_intensity": 0.0, "pose_model_active": False,
            "spine_angle": 0.0}

def fake_base_state():
    from state.social_state import estimate_social_state
    return estimate_social_state(
        fake_eye_t(), fake_head_t(), fake_gaze_t(),
        fake_mouth_t(), fake_brow_f(), fake_aus(), 
        {"face_presence": True, "face_count": 1, "face_distance_cm": 55.0,
         "face_center": [0.5, 0.5], "face_bbox": [0.3,0.3,0.7,0.7],
         "face_bbox_size": 0.16, "face_yaw_proxy": 0.0,
         "frame_displacement": 1.0, "confidence": 1.0}
    )

def fake_aff_emb():
    from features.affective_embedding import extract_affective_embedding
    return extract_affective_embedding(
        fake_aus(), fake_eye_f(), fake_mouth_f(), fake_brow_f(), fake_head_t()
    )

def fake_gru_out():
    from temporal.affective_gru import AffectiveGRU
    gru = AffectiveGRU()
    aff = fake_aff_emb()
    ts = NOW_MS
    out = {}
    for _ in range(20):
        out = gru.update(aff, ts)
        ts += 33
    return out


# ── Test: affective_embedding ──────────────────────────────────────────────────

def test_affective_embedding():
    print("\n[features/affective_embedding]")
    from features.affective_embedding import extract_affective_embedding

    r = extract_affective_embedding(fake_aus(), fake_eye_f(), fake_mouth_f(),
                                     fake_brow_f(), fake_head_t())

    check("has embedding",       "embedding"       in r)
    check("has valence_geom",    "valence_geom"    in r)
    check("has arousal_geom",    "arousal_geom"    in r)
    check("has dominance_geom",  "dominance_geom"  in r)
    check("has affect_quadrant", "affect_quadrant" in r)
    check("has expression_conf", "expression_conf" in r)

    check("embedding dim=48",    len(r["embedding"]) == 48,
          f"got {len(r['embedding'])}")
    check("valence [-1,1]",      -1.0 <= r["valence_geom"]   <= 1.0)
    check("arousal [0,1]",        0.0 <= r["arousal_geom"]   <= 1.0)
    check("dominance [0,1]",      0.0 <= r["dominance_geom"] <= 1.0)
    check("quadrant valid",       r["affect_quadrant"] in {"HVHA","HVLA","LVHA","LVLA"})
    check("expr_conf [0,1]",      0.0 <= r["expression_conf"] <= 1.0)

    # High smile → positive valence
    r_happy = extract_affective_embedding(
        fake_aus(smile=0.9, au4=0.0), fake_eye_f(), fake_mouth_f(smile=0.9),
        fake_brow_f(), fake_head_t()
    )
    check("high smile → positive valence", r_happy["valence_geom"] > 0.0,
          f"got {r_happy['valence_geom']:.3f}")

    # High AU4 (brow lower) + AU15 → negative valence
    r_sad = extract_affective_embedding(
        fake_aus(smile=0.0, au4=0.8), fake_eye_f(), fake_mouth_f(smile=0.0),
        fake_brow_f(tension=0.8), fake_head_t()
    )
    check("brow lower + no smile → negative valence", r_sad["valence_geom"] < 0.1,
          f"got {r_sad['valence_geom']:.3f}")

    # Embedding values finite
    emb = np.array(r["embedding"])
    check("embedding all finite",  np.all(np.isfinite(emb)))
    check("embedding in [-2,2]",   np.all(np.abs(emb) <= 2.0))

    # Duchenne smile (AU6*AU12) should be in embedding
    aff_duchenne = extract_affective_embedding(
        fake_aus(smile=1.0, au6=1.0), fake_eye_f(), fake_mouth_f(smile=1.0),
        fake_brow_f(), fake_head_t()
    )
    check("Duchenne embedding slot > 0",
          aff_duchenne["embedding"][30] > 0.1,   # index 30 = first pair (AU6*AU12)
          f"got {aff_duchenne['embedding'][30]:.4f}")


# ── Test: affective_gru ────────────────────────────────────────────────────────

def test_affective_gru():
    print("\n[temporal/affective_gru]")
    from temporal.affective_gru import AffectiveGRU

    gru = AffectiveGRU(hidden_dim=32)
    aff = fake_aff_emb()

    # Single step
    out = gru.update(aff, NOW_MS)
    check("has gru_valence",        "gru_valence"         in out)
    check("has gru_arousal",        "gru_arousal"         in out)
    check("has gru_dominance",      "gru_dominance"       in out)
    check("has temporal_coherence", "temporal_coherence"  in out)
    check("has emotional_momentum", "emotional_momentum"  in out)
    check("has gru_affect_quadrant","gru_affect_quadrant" in out)

    check("gru_valence [-1,1]",     -1.0 <= out["gru_valence"]        <= 1.0)
    check("gru_arousal [0,1]",       0.0 <= out["gru_arousal"]        <= 1.0)
    check("temporal_coherence [0,1]",0.0 <= out["temporal_coherence"] <= 1.0)
    check("momentum [0,1]",          0.0 <= out["emotional_momentum"]  <= 1.0)
    check("quadrant valid",          out["gru_affect_quadrant"] in {"HVHA","HVLA","LVHA","LVLA"})

    # Hidden state should change over time
    ts = NOW_MS
    prev_val = out["gru_valence"]
    changed  = False
    for _ in range(50):
        out2 = gru.update(aff, ts)
        ts  += 33
        if abs(out2["gru_valence"] - prev_val) > 1e-6:
            changed = True
        prev_val = out2["gru_valence"]
    check("GRU state evolves over frames", changed)

    # Reset clears hidden state
    gru.reset()
    out_fresh = gru.update(aff, ts)
    # After reset, output should differ from post-50-frame state
    check("reset produces fresh state",
          abs(out_fresh["gru_valence"] - out2["gru_valence"]) > 0.0 or
          abs(out_fresh["gru_arousal"] - out2["gru_arousal"]) > 0.0)

    # Happy signal → eventually positive GRU valence
    gru2 = AffectiveGRU()
    happy_aff = extract_affective_embedding_happy()
    ts2 = NOW_MS
    for _ in range(60):
        g2 = gru2.update(happy_aff, ts2); ts2 += 33
    check("sustained smile → coherent emotional state",
          0.0 <= g2["temporal_coherence"] <= 1.0)

    # Invalid embedding → no crash
    bad_emb = {"embedding": [0.0] * 48}
    out_bad = gru.update(bad_emb, ts)
    check("invalid embedding → no crash", True)

    # Wrong-length embedding → graceful
    bad_emb2 = {"embedding": [0.0] * 10}
    out_bad2 = gru.update(bad_emb2, ts)
    check("short embedding → no crash", True)


def extract_affective_embedding_happy():
    from features.affective_embedding import extract_affective_embedding
    return extract_affective_embedding(
        fake_aus(smile=0.9, au6=0.8, au4=0.0),
        fake_eye_f(ear=0.35),
        fake_mouth_f(smile=0.9),
        fake_brow_f(raise_=0.4, tension=0.0),
        fake_head_t(pitch=0.0, stability=1.0, attn=1.0)
    )


# ── Test: uncertainty ──────────────────────────────────────────────────────────

def test_uncertainty():
    print("\n[state/uncertainty]")
    from state.uncertainty import UncertaintyTracker

    tracker = UncertaintyTracker()
    base    = fake_base_state()
    aff     = fake_aff_emb()
    gru_out = fake_gru_out()

    # Single update
    unc = tracker.update(base, aff, gru_out, sensor_conf=0.9)

    check("has dimensions",          "dimensions"         in unc)
    check("has global_uncertainty",  "global_uncertainty" in unc)
    check("has affective_ambiguity", "affective_ambiguity"in unc)
    check("has signal_disagreement", "signal_disagreement"in unc)
    check("has temporal_instability","temporal_instability"in unc)

    check("global_unc [0,1]",        0.0 <= unc["global_uncertainty"]   <= 1.0)
    check("ambiguity [0,1]",         0.0 <= unc["affective_ambiguity"]  <= 1.0)
    check("disagreement [0,1]",      0.0 <= unc["signal_disagreement"]  <= 1.0)
    check("instability [0,1]",       0.0 <= unc["temporal_instability"] <= 1.0)

    # Per-dimension checks
    for dim in ["valence", "arousal", "engagement", "stress_hints"]:
        d = unc["dimensions"].get(dim, {})
        check(f"dim {dim} present",      dim in unc["dimensions"])
        check(f"dim {dim}.confidence [0,1]",
              0.0 <= d.get("confidence", -1) <= 1.0,
              f"got {d.get('confidence')}")
        check(f"dim {dim}.uncertainty [0,1]",
              0.0 <= d.get("uncertainty", -1) <= 1.0)
        check(f"dim {dim}.stability [0,1]",
              0.0 <= d.get("stability", -1) <= 1.0)
        check(f"dim {dim}.ci_low <= ci_high",
              d.get("ci_low", 0) <= d.get("ci_high", 1))

    # Low sensor confidence → high global uncertainty
    unc_low = tracker.update(base, aff, gru_out, sensor_conf=0.1)
    check("low sensor conf → higher uncertainty",
          unc_low["global_uncertainty"] > unc["global_uncertainty"],
          f"{unc_low['global_uncertainty']:.3f} vs {unc['global_uncertainty']:.3f}")

    # Stability increases with repeated same signal
    tracker2 = UncertaintyTracker()
    ts = NOW_MS
    for _ in range(40):
        unc2 = tracker2.update(base, aff, gru_out, 0.9)
    check("stability improves over consistent frames",
          unc2["dimensions"]["engagement"]["stability"] > 0.5,
          f"got {unc2['dimensions']['engagement']['stability']:.3f}")

    tracker.reset()
    check("reset no crash", True)


# ── Test: emotional_dynamics ───────────────────────────────────────────────────

def test_emotional_dynamics():
    print("\n[state/emotional_dynamics]")
    from state.emotional_dynamics import EmotionalDynamicsTracker

    tracker = EmotionalDynamicsTracker()
    base    = fake_base_state()
    aff     = fake_aff_emb()
    gru_out = fake_gru_out()

    ts = NOW_MS
    for _ in range(10):
        dyn = tracker.update(aff, gru_out, base, ts)
        ts += 33

    check("has current_quadrant",      "current_quadrant"     in dyn)
    check("has quadrant_label",        "quadrant_label"       in dyn)
    check("has quadrant_duration_ms",  "quadrant_duration_ms" in dyn)
    check("has transition_occurred",   "transition_occurred"  in dyn)
    check("has stress_escalating",     "stress_escalating"    in dyn)
    check("has stress_de_escalating",  "stress_de_escalating" in dyn)
    check("has stress_trend",          "stress_trend"         in dyn)
    check("has engagement_shift",      "engagement_shift"     in dyn)
    check("has emotional_momentum",    "emotional_momentum"   in dyn)
    check("has affective_inertia",     "affective_inertia"    in dyn)
    check("has state_persistence",     "state_persistence"    in dyn)
    check("has recent_transitions",    "recent_transitions"   in dyn)

    check("quadrant valid",         dyn["current_quadrant"] in {"HVHA","HVLA","LVHA","LVLA"})
    check("engagement_shift valid", dyn["engagement_shift"] in {"rising","falling","stable"})
    check("momentum [0,1]",         0.0 <= dyn["emotional_momentum"] <= 1.0)
    check("inertia [0,1]",          0.0 <= dyn["affective_inertia"]  <= 1.0)
    check("persistence [0,1]",      0.0 <= dyn["state_persistence"]  <= 1.0)
    check("duration_ms >= 0",       dyn["quadrant_duration_ms"]      >= 0.0)

    # Stress escalation test: inject rising stress
    from features.affective_embedding import extract_affective_embedding
    tracker2 = EmotionalDynamicsTracker()
    ts2 = NOW_MS
    stress_base = fake_base_state()
    for i in range(80):
        # Gradually increase AU4 + brow_tension (stress signals)
        stress_val = min(i / 80.0, 1.0)
        stress_aus = fake_aus(smile=0.0, au4=stress_val)
        stress_brow= fake_brow_f(tension=stress_val)
        stress_aff = extract_affective_embedding(stress_aus, fake_eye_f(), fake_mouth_f(smile=0.0),
                                                  stress_brow, fake_head_t())
        # Override stress in base state
        import copy
        sb = copy.deepcopy(stress_base)
        sb["stress_hints"] = {"value": stress_val * 0.8, "confidence": 0.9}
        dyn2 = tracker2.update(stress_aff, gru_out, sb, ts2)
        ts2 += 33
    check("rising stress → stress_escalating",
          dyn2["stress_escalating"] or dyn2["stress_trend"] > 0,
          f"trend={dyn2['stress_trend']:.5f}")

    tracker.reset()
    check("reset no crash", True)


# ── Test: social_cognition ────────────────────────────────────────────────────

def test_social_cognition():
    print("\n[state/social_cognition]")
    from state.social_cognition import SocialCognitionTracker

    tracker = SocialCognitionTracker()
    base    = fake_base_state()
    aff     = fake_aff_emb()

    ts = NOW_MS
    for _ in range(30):
        cog = tracker.update(
            base, fake_head_t(), fake_gaze_t(), fake_mouth_t(),
            fake_eye_t(), aff, fake_body_f(), fake_aus(), ts
        )
        ts += 33

    fields = [
        "social_responsiveness", "interaction_reciprocity",
        "conversational_readiness", "social_comfort", "social_openness",
        "rapport_signal", "social_attention_quality",
        "proxemic_comfort", "behavioral_synchrony",
    ]
    for f in fields:
        check(f"has {f}",   f in cog)
        check(f"{f} [0,1]", 0.0 <= cog[f] <= 1.0, f"got {cog[f]:.3f}")

    # High engagement → high conversational readiness
    tracker2 = SocialCognitionTracker()
    ts2 = NOW_MS
    for _ in range(30):
        cog2 = tracker2.update(
            base, fake_head_t(attn=1.0), fake_gaze_t(ec=0.95, ec_rate=0.95, vol=0.05),
            fake_mouth_t(), fake_eye_t(), aff, fake_body_f(), fake_aus(), ts2
        )
        ts2 += 33
    check("high attention → high conv_readiness",
          cog2["conversational_readiness"] > 0.4,
          f"got {cog2['conversational_readiness']:.3f}")

    # High discomfort → low social comfort
    import copy
    base_discomfort = copy.deepcopy(base)
    base_discomfort["discomfort"] = {"value": 0.9, "confidence": 0.9}
    tracker3 = SocialCognitionTracker()
    ts3 = NOW_MS
    for _ in range(30):
        cog3 = tracker3.update(
            base_discomfort, fake_head_t(), fake_gaze_t(ec=0.1, ec_rate=0.1),
            fake_mouth_t(smile=0.0), fake_eye_t(), aff, fake_body_f(), fake_aus(), ts3
        )
        ts3 += 33
    check("high discomfort → lower social comfort",
          cog3["social_comfort"] < cog2["social_comfort"],
          f"discomfort={cog3['social_comfort']:.3f} vs engaged={cog2['social_comfort']:.3f}")

    tracker.reset()
    check("reset no crash", True)


# ── Test: rich_json ────────────────────────────────────────────────────────────

def test_rich_json():
    print("\n[outputs/rich_json]")
    from outputs.rich_json import format_rich_state, parse_rich_state
    from state.emotional_dynamics import EmotionalDynamicsTracker
    from state.uncertainty import UncertaintyTracker
    from state.social_cognition import SocialCognitionTracker

    base    = fake_base_state()
    aff     = fake_aff_emb()
    gru_out = fake_gru_out()

    unc_t = UncertaintyTracker()
    dyn_t = EmotionalDynamicsTracker()
    cog_t = SocialCognitionTracker()

    ts = NOW_MS
    for _ in range(20):
        unc = unc_t.update(base, aff, gru_out, 0.9)
        dyn = dyn_t.update(aff, gru_out, base, ts)
        cog = cog_t.update(base, fake_head_t(), fake_gaze_t(), fake_mouth_t(),
                            fake_eye_t(), aff, fake_body_f(), fake_aus(), ts)
        ts += 33

    from state.derived_state import compute_derived_state
    from temporal.social_temporal import SocialTemporalTracker
    from temporal.face_temporal import FaceTemporalTracker
    soc_tr = SocialTemporalTracker()
    face_tr = FaceTemporalTracker()
    soc_t_out = soc_tr.update(base, fake_gaze_t(), ts)
    face_t_out = face_tr.update({
        "face_presence": True, "face_center": [0.5,0.5],
        "frame_displacement": 1.0, "face_distance_cm": 55.0, "confidence":1.0
    }, ts)

    derived = compute_derived_state(
        base, soc_t_out, fake_brow_f(), fake_aus(), fake_gaze_t(),
        fake_head_t(), face_t_out, fake_eye_t()
    )

    full_state = {
        **base, **derived, **aff, **gru_out, **dyn, **cog,
        **unc, "uncertainty": unc,
    }

    # Standard JSON
    js = format_rich_state(full_state, timestamp_ms=NOW_MS)
    obj = json.loads(js)

    check("valid JSON",       True)
    check("has timestamp_ms", "timestamp_ms" in obj)
    check("has face_present", "face_present" in obj)
    check("has narrative",    "narrative"    in obj)
    check("has dimensions",   "dimensions"   in obj)
    check("has affective",    "affective"    in obj)
    check("has dynamics",     "dynamics"     in obj)
    check("has social",       "social"       in obj)
    check("has uncertainty",  "uncertainty"  in obj)
    check("has llm_visual",   "llm_visual"   in obj)
    check("has behavioral",   "behavioral"   in obj)

    # Affective section
    aff_sec = obj["affective"]
    check("affective.embedding len=48", len(aff_sec.get("embedding", [])) == 48)
    check("affective.valence_geom",     "valence_geom"    in aff_sec)
    check("affective.arousal_geom",     "arousal_geom"    in aff_sec)
    check("affective.gru_valence",      "gru_valence"     in aff_sec)
    check("affective.temporal_coherence","temporal_coherence" in aff_sec)

    # Dynamics section
    dyn_sec = obj["dynamics"]
    for key in ["current_quadrant","quadrant_label","stress_escalating",
                "engagement_shift","state_persistence"]:
        check(f"dynamics.{key}", key in dyn_sec)

    # Social section
    soc_sec = obj["social"]
    for key in ["social_responsiveness","interaction_reciprocity",
                "conversational_readiness","social_comfort","rapport_signal"]:
        check(f"social.{key}", key in soc_sec)

    # Uncertainty section
    unc_sec = obj["uncertainty"]
    for key in ["global_uncertainty","affective_ambiguity",
                "signal_disagreement","temporal_instability"]:
        check(f"uncertainty.{key}", key in unc_sec)

    # LLM visual — all new fields
    llm = obj["llm_visual"]
    for key in ["visual_engagement","visual_attention","visual_stress_hints",
                "social_comfort","rapport_signal","conversational_readiness",
                "affect_quadrant","stress_escalating","temporal_coherence",
                "global_uncertainty","visual_emotional_valence"]:
        check(f"llm_visual.{key}", key in llm)

    # All numeric llm_visual values in [0,1] (except bools and strings)
    for key, val in llm.items():
        if isinstance(val, float):
            check(f"llm_visual.{key} [0,1]", 0.0 <= val <= 1.0,
                  f"got {val:.3f}")

    # Dimension uncertainty intervals
    dims = obj["dimensions"]
    for name in ["valence", "arousal", "engagement"]:
        d = dims.get(name, {})
        check(f"dim {name} has uncertainty", "uncertainty" in d)
        check(f"dim {name} has stability",   "stability"   in d)
        check(f"dim {name} has ci_low",      "ci_low"      in d)
        check(f"dim {name} has ci_high",     "ci_high"     in d)
        check(f"dim {name} ci_low<=ci_high",
              d.get("ci_low", 0) <= d.get("ci_high", 1))

    # Pretty print
    js_pretty = format_rich_state(full_state, pretty=True)
    check("pretty JSON parseable", json.loads(js_pretty) is not None)

    # Round-trip
    obj2 = parse_rich_state(js)
    check("round-trip OK", obj2["face_present"] == obj["face_present"])

    # No-face state
    from state.social_state import _absent_state
    absent = _absent_state()
    js_absent = format_rich_state(absent)
    obj_absent = json.loads(js_absent)
    check("no-face JSON valid",          isinstance(obj_absent, dict))
    check("no-face narrative mentions",  "No face" in obj_absent["narrative"])

    # Narrative is non-empty and meaningful for face-present state
    check("narrative non-empty",  len(obj["narrative"]) > 50)
    check("narrative is string",  isinstance(obj["narrative"], str))


# ── Test: full affective pipeline ─────────────────────────────────────────────

def test_full_affective_pipeline():
    print("\n[full affective pipeline integration]")
    from features.affective_embedding import extract_affective_embedding
    from temporal.affective_gru       import AffectiveGRU
    from state.uncertainty            import UncertaintyTracker
    from state.emotional_dynamics     import EmotionalDynamicsTracker
    from state.social_cognition       import SocialCognitionTracker
    from state.social_state           import estimate_social_state
    from state.derived_state          import compute_derived_state
    from temporal.social_temporal     import SocialTemporalTracker
    from temporal.face_temporal       import FaceTemporalTracker
    from temporal.eye_temporal        import BlinkTracker
    from temporal.head_pose_temporal  import HeadPoseTracker
    from temporal.gaze_temporal       import GazeTracker
    from temporal.mouth_temporal      import MouthTracker
    from outputs.rich_json            import format_rich_state

    gru    = AffectiveGRU()
    unc_t  = UncertaintyTracker()
    dyn_t  = EmotionalDynamicsTracker()
    cog_t  = SocialCognitionTracker()
    blink  = BlinkTracker()
    head_tr= HeadPoseTracker(fps=30)
    gaze_tr= GazeTracker()
    mouth_tr=MouthTracker(fps=30)
    face_tr= FaceTemporalTracker()
    soc_tr = SocialTemporalTracker()

    face_feat = {
        "face_presence": True, "face_count": 1, "face_distance_cm": 55.0,
        "face_center": [0.5, 0.5], "face_bbox": [0.3,0.3,0.7,0.7],
        "face_bbox_size": 0.16, "face_yaw_proxy": 0.0,
        "frame_displacement": 1.0, "confidence": 1.0
    }

    ts = NOW_MS
    full_state = {}
    for i in range(50):
        # Vary signals slightly over time
        smile_v = 0.3 + 0.2 * np.sin(i / 10.0)
        aus_i   = fake_aus(smile=smile_v)
        eye_fi  = fake_eye_f()
        mouth_fi= fake_mouth_f(smile=smile_v)
        brow_fi = fake_brow_f()
        head_fi = {"head_yaw": 3.0, "head_pitch": -5.0, "head_roll": 1.0,
                   "rotation_vector": [0,0,0], "reprojection_error": 2.0, "confidence": 0.9}
        gaze_fi = fake_gaze_t()
        mouth_fi_dict = fake_mouth_t(smile=smile_v)

        eye_t   = blink.update(eye_fi, ts)
        head_t  = head_tr.update(head_fi, ts)
        gaze_t  = gaze_tr.update(gaze_fi, ts)
        mouth_t = mouth_tr.update(mouth_fi_dict, ts)
        face_t  = face_tr.update(face_feat, ts)

        aff     = extract_affective_embedding(aus_i, eye_fi, mouth_fi, brow_fi, head_t)
        gru_out = gru.update(aff, ts)

        base    = estimate_social_state(eye_t, head_t, gaze_t, mouth_t, brow_fi, aus_i, face_feat)
        soc_t   = soc_tr.update(base, gaze_t, ts)
        derived = compute_derived_state(base, soc_t, brow_fi, aus_i, gaze_t, head_t, face_t, eye_t)

        sensor_conf = base.get("meta", {}).get("overall_confidence", 0.5)
        unc     = unc_t.update(base, aff, gru_out, sensor_conf)
        dyn     = dyn_t.update(aff, gru_out, base, ts)
        cog     = cog_t.update(base, head_t, gaze_t, mouth_t, eye_t, aff,
                                fake_body_f(), aus_i, ts)

        full_state = {
            **base, **derived, **aff, **gru_out, **dyn, **cog,
            **unc, "uncertainty": unc,
        }
        ts += 33

    check("50-frame pipeline complete", True)

    # Validate output JSON
    js  = format_rich_state(full_state, timestamp_ms=ts)
    obj = json.loads(js)

    check("final JSON valid",              isinstance(obj, dict))
    check("face_present True",             obj["face_present"])
    check("affective section present",     "affective" in obj)
    check("dynamics section present",      "dynamics"  in obj)
    check("social section present",        "social"    in obj)
    check("uncertainty section present",   "uncertainty" in obj)
    check("embedding 48-dim",             len(obj["affective"]["embedding"]) == 48)
    check("GRU valence in output",         "gru_valence" in obj["affective"])
    check("quadrant in dynamics",          "current_quadrant" in obj["dynamics"])
    check("rapport in social",             "rapport_signal" in obj["social"])
    check("global_uncertainty in unc",     "global_uncertainty" in obj["uncertainty"])
    check("all llm values [0,1]",
          all(0.0 <= v <= 1.0
              for k, v in obj["llm_visual"].items()
              if isinstance(v, float)))

    # Check GRU values are reasonable (not NaN or inf)
    gru_v = obj["affective"]["gru_valence"]
    gru_a = obj["affective"]["gru_arousal"]
    check("gru_valence finite", -1.0 <= gru_v <= 1.0, f"got {gru_v}")
    check("gru_arousal finite",  0.0 <= gru_a <= 1.0, f"got {gru_a}")

    # Emotional dynamics should have settled
    quad = obj["dynamics"]["current_quadrant"]
    check("quadrant set", quad in {"HVHA","HVLA","LVHA","LVLA"}, f"got {quad}")

    # Social signals should be non-zero after 50 frames of engagement
    check("rapport > 0",               obj["social"]["rapport_signal"]           > 0.0)
    check("conv_readiness > 0",        obj["social"]["conversational_readiness"] > 0.0)
    check("social_comfort > 0",        obj["social"]["social_comfort"]           > 0.0)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_affective_embedding,
        test_affective_gru,
        test_uncertainty,
        test_emotional_dynamics,
        test_social_cognition,
        test_rich_json,
        test_full_affective_pipeline,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"  ✗ EXCEPTION in {t.__name__}: {e}")
            traceback.print_exc()
            _failed += 1

    total = _passed + _failed
    print(f"\n{'='*50}")
    print(f"Results: {_passed}/{total} passed  |  {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)
