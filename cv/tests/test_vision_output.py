"""tests/test_vision_output.py — vision_v1 JSON and baseline tracker."""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_passed = 0
_failed = 0
NOW_MS = int(time.time() * 1000)


def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        print(f"  ✓ {name}")
        _passed += 1
    else:
        print(f"  ✗ {name}{' — ' + detail if detail else ''}")
        _failed += 1


def test_baseline_tracker():
    print("\n[state/baseline_tracker]")
    from state.baseline_tracker import PersonalBaselineTracker

    bt = PersonalBaselineTracker(activation_ms=1000.0)
    ts = NOW_MS
    for i in range(120):
        bt.update({
            "blink_rate": 14.0 + (i % 3),
            "avg_EAR": 0.27,
            "brow_tension": 0.38,
            "facial_tension": 0.25,
            "lip_tension": 0.08,
            "gaze_volatility": 0.15,
            "AU4": 0.20,
            "expression_intensity": 0.18,
        }, ts)
        ts += 33

    check("not active before activation_ms", not bt.is_active())
    check("elevated works", bt.elevated(0.55, "brow_tension") >= 0.0)
    snap = bt.export_snapshot()
    check("snapshot has signals", "signals" in snap)
    check("brow rest tracked", "brow_tension" in snap["signals"])

    # High-resting-brow person at their own resting level (not elevated vs self)
    bt2 = PersonalBaselineTracker(activation_ms=500.0)
    ts2 = NOW_MS
    for _ in range(350):
        bt2.update({
            "blink_rate": 13.0, "avg_EAR": 0.26, "brow_tension": 0.45,
            "facial_tension": 0.22, "lip_tension": 0.07, "gaze_volatility": 0.12,
            "AU4": 0.18, "expression_intensity": 0.15,
        }, ts2)
        ts2 += 33
    check("active after face time + samples", bt2.is_active())

    from state.social_state import estimate_social_state
    eye_t = {"blink_rate": 13.0, "avg_EAR": 0.26, "blink_trend": 0.0, "confidence": 0.9}
    head_t = {"attention_score": 0.85, "attention_direction": "forward",
              "head_stability": 0.9, "yaw": 2, "pitch": -3, "confidence": 0.9}
    gaze_t = {"eye_contact_score": 0.8, "eye_contact_rate": 0.75,
              "gaze_volatility": 0.12, "gaze_pattern": "fixating",
              "fixation_active": True, "confidence": 1.0}
    mouth_t = {"smile_intensity": 0.3, "speaking_activity": 0.0, "yawning": False,
               "lip_tension": 0.07, "tension_trend": 0.0, "confidence": 1.0}
    brow_f = {"brow_raise": 0.1, "brow_lower": 0.2, "brow_tension": 0.45,
              "brow_asymmetry": 0.05, "confidence": 0.9}
    aus = {"AU4": 0.18, "AU12": 0.2, "AU15": 0.05, "facial_tension": 0.22,
           "expression_intensity": 0.15, "confidence": 0.9}
    face_f = {"face_presence": True, "face_distance_cm": 55.0, "confidence": 1.0}

    without = estimate_social_state(eye_t, head_t, gaze_t, mouth_t, brow_f, aus, face_f)
    with_bl = estimate_social_state(
        eye_t, head_t, gaze_t, mouth_t, brow_f, aus, face_f, baseline=bt2,
    )
    check("baseline lowers stress for high-resting brow",
          with_bl["stress_hints"]["value"] < without["stress_hints"]["value"],
          f"{with_bl['stress_hints']['value']:.3f} vs {without['stress_hints']['value']:.3f}")


def test_vision_v1_schema():
    print("\n[outputs/vision_output]")
    from outputs.vision_output import format_vision_v1, parse_vision_v1, SCHEMA

    state = {
        "face_present": True,
        "support_urgency": "monitor",
        "distress_level": 0.35,
        "distress_type": "anxiety",
        "openness_to_support": 0.55,
        "genuine_smile": False,
        "genuine_smile_score": 0.1,
        "emotional_suppression": 0.2,
        "overwhelm_probability": 0.15,
        "comfort_seeking": 0.1,
        "emotional_withdrawal": 0.2,
        "gru_valence": -0.12,
        "gru_arousal": 0.4,
        "gru_dominance": 0.5,
        "gru_affect_quadrant": "LVLA",
        "quadrant_label": "sad/bored",
        "temporal_coherence": 0.7,
        "emotional_momentum": 0.1,
        "valence_velocity": -0.002,
        "arousal_velocity": 0.001,
        "engagement": {"value": 0.65, "confidence": 0.85},
        "attention": {"value": 0.6, "direction": "forward", "confidence": 0.8},
        "eye_contact_rate": 0.7,
        "rapport_signal": 0.45,
        "conversational_readiness": 0.5,
        "social_comfort": 0.55,
        "social_openness": 0.5,
        "social_responsiveness": 0.4,
        "behavioral_synchrony": 0.45,
        "stress_escalating": False,
        "stress_de_escalating": False,
        "engagement_shift": "stable",
        "transition_occurred": False,
        "quadrant_duration_ms": 3000,
        "state_persistence": 0.4,
        "recent_transitions": 0,
        "engagement_trend": 0.0,
        "stress_trend": 0.001,
        "speaking": False,
        "eye_contact": True,
        "fidget_probability": 0.1,
        "global_uncertainty": 0.25,
        "signal_disagreement": 0.15,
        "affective_ambiguity": 0.3,
        "meta": {"overall_confidence": 0.82, "head_direction": "forward"},
        "uncertainty": {"dimensions": {}},
        "baseline_active": True,
    }

    js = format_vision_v1(state, session_id="ses_test123", emit_reason="heartbeat",
                          timestamp_ms=NOW_MS, baseline_active=True)
    obj = parse_vision_v1(js)
    check("schema field", obj.get("schema") == SCHEMA)
    check("session_id", obj.get("session_id") == "ses_test123")
    check("baseline_active", obj.get("baseline_active") is True)
    check("support section", "support" in obj and "urgency" in obj["support"])
    check("affect section", "affect" in obj and "valence" in obj["affect"])
    check("social section", "social" in obj)
    check("dynamics section", "dynamics" in obj)
    check("behavioral section", "behavioral" in obj)
    check("uncertainty section", "uncertainty" in obj)
    check("valid json one line", "\n" not in js or js.count("{") > 0)
    check("distress in support", obj["support"]["distress_level"] == 0.35)

    pretty = format_vision_v1(state, session_id="ses_x", pretty=True)
    check("pretty has newlines", "\n" in pretty)


if __name__ == "__main__":
    test_baseline_tracker()
    test_vision_v1_schema()
    total = _passed + _failed
    print(f"\n{'='*50}")
    print(f"Results: {_passed}/{total} passed  |  {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)
