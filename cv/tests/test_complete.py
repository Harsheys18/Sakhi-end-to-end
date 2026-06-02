"""
tests/test_complete.py

Tests for all modules added in the completion pass:
  temporal/face_temporal.py
  temporal/social_temporal.py
  features/gaze_aversion.py
  features/environment.py
  features/body_pose.py
  features/hand_features.py
  state/derived_state.py
  outputs/json_output.py  (extended fields)
  Full pipeline integration (all modules together)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time

_passed = 0
_failed = 0

def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        print(f"  ✓ {name}")
        _passed += 1
    else:
        print(f"  ✗ {name}{' — ' + detail if detail else ''}")
        _failed += 1

NOW = int(time.time() * 1000)

# ── Shared fake state dicts ────────────────────────────────────────────────────

def fake_eye_t(blink_rate=15.0, avg_ear=0.28, blink_trend=0.0):
    return {"blink_rate": blink_rate, "avg_EAR": avg_ear, "blink_trend": blink_trend,
            "eye_open_duration_ms": 800.0, "ear_mean": avg_ear, "ear_std": 0.02,
            "confidence": 0.9}

def fake_head_t(yaw=3.0, pitch=-5.0, direction="forward", stability=0.9):
    return {"attention_score": 0.85, "attention_direction": direction,
            "head_stability": stability, "nod_detected": False,
            "yaw": yaw, "pitch": pitch, "roll": 2.0,
            "yaw_velocity": 0.5, "pitch_velocity": 0.3, "confidence": 0.9}

def fake_gaze_t(ec_score=0.80, ec_rate=0.75, volatility=0.10, pattern="fixating", gaze_y=0.0):
    return {"eye_contact_score": ec_score, "eye_contact_rate": ec_rate,
            "gaze_volatility": volatility, "gaze_pattern": pattern,
            "fixation_active": True, "gaze_x": 0.05, "gaze_y": gaze_y,
            "gaze_direction": "center", "confidence": 1.0}

def fake_mouth_t(smile=0.4, speaking=0.0, yawning=False, tension=0.1):
    return {"smile_intensity": smile, "speaking_activity": speaking, "yawning": yawning,
            "lip_tension": tension, "tension_trend": 0.0, "lip_compression": 0.05,
            "MAR": 0.12, "confidence": 1.0}

def fake_brow(raise_=0.2, lower=0.1, tension=0.1, asym=0.05):
    return {"brow_raise": raise_, "brow_lower": lower, "brow_tension": tension,
            "brow_asymmetry": asym, "inner_brow_angle": 2.0, "confidence": 0.9}

def fake_aus(smile=0.4, tension=0.1, intensity=0.2):
    return {"AU4": 0.1, "AU6": 0.3, "AU7": 0.1, "AU12": smile,
            "AU15": 0.05, "AU23": 0.1, "AU24": 0.05, "AU26": 0.1, "AU43": 0.0,
            "facial_tension": tension, "expression_intensity": intensity, "confidence": 0.9}

def fake_face_f(cx=0.5, cy=0.5, disp=1.0):
    return {"face_presence": True, "face_count": 1, "face_distance_cm": 55.0,
            "face_center": [cx, cy], "face_bbox": [0.3, 0.3, 0.7, 0.7],
            "face_bbox_size": 0.16, "face_yaw_proxy": 0.0,
            "frame_displacement": disp, "confidence": 1.0}

def fake_face_t():
    return {"face_stability": 0.9, "movement_energy": 1.5, "movement_variability": 0.1,
            "movement_trend": 0.0, "prolonged_stillness": False,
            "stillness_duration_ms": 0.0, "restlessness_score": 0.1}

def fake_social_state():
    from state.social_state import estimate_social_state
    return estimate_social_state(
        fake_eye_t(), fake_head_t(), fake_gaze_t(),
        fake_mouth_t(), fake_brow(), fake_aus(), fake_face_f()
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_face_temporal():
    print("\n[temporal/face_temporal]")
    from temporal.face_temporal import FaceTemporalTracker

    tracker = FaceTemporalTracker()
    ts = NOW
    for i in range(30):
        state = tracker.update(fake_face_f(disp=float(i % 5)), ts)
        ts += 33

    check("has face_stability",      "face_stability"      in state)
    check("has movement_energy",     "movement_energy"     in state)
    check("has movement_variability","movement_variability" in state)
    check("has movement_trend",      "movement_trend"      in state)
    check("has restlessness_score",  "restlessness_score"  in state)
    check("face_stability [0,1]",    0.0 <= state["face_stability"]      <= 1.0)
    check("movement_var [0,1]",      0.0 <= state["movement_variability"] <= 1.0)
    check("restlessness [0,1]",      0.0 <= state["restlessness_score"]   <= 1.0)

    # Prolonged stillness test
    tracker2 = FaceTemporalTracker()
    ts2 = NOW
    for _ in range(100):   # ~3.3 s at 30fps
        s = tracker2.update(fake_face_f(disp=0.2), ts2)  # very still
        ts2 += 33
    check("prolonged_stillness detected", s["prolonged_stillness"],
          f"still_dur={s['stillness_duration_ms']:.0f}ms")


def test_social_temporal():
    print("\n[temporal/social_temporal]")
    from temporal.social_temporal import SocialTemporalTracker

    tracker = SocialTemporalTracker()
    ss = fake_social_state()
    gt = fake_gaze_t()
    ts = NOW
    for _ in range(60):
        state = tracker.update(ss, gt, ts)
        ts += 33

    check("has engagement_trend",        "engagement_trend"   in state)
    check("has attention_trend",         "attention_trend"    in state)
    check("has emotion_transition_rate", "emotion_transition_rate" in state)
    check("has emotional_stability",     "emotional_stability" in state)
    check("has attention_stability",     "attention_stability" in state)
    check("has interaction_stability",   "interaction_stability" in state)
    check("emotional_stability [0,1]",   0.0 <= state["emotional_stability"] <= 1.0)
    check("attention_stability [0,1]",   0.0 <= state["attention_stability"]  <= 1.0)

    # Sustained attention test (high attention for long enough)
    tracker2 = SocialTemporalTracker()
    ts2 = NOW
    for _ in range(100):
        s = tracker2.update(ss, fake_gaze_t(ec_rate=0.9), ts2)
        ts2 += 33
    check("sustained_attention detected", s["sustained_attention"] or s["sustained_attention_ms"] > 0,
          f"ms={s['sustained_attention_ms']:.0f}")

    # Prolonged downward gaze
    tracker3 = SocialTemporalTracker()
    ts3 = NOW
    for _ in range(60):
        s3 = tracker3.update(fake_social_state(), fake_gaze_t(gaze_y=0.5), ts3)
        ts3 += 33
    check("prolonged_downward_gaze detectable",
          s3["prolonged_downward_gaze"] or s3["prolonged_downward_gaze_ms"] > 0)


def test_gaze_aversion():
    print("\n[features/gaze_aversion]")
    from features.gaze_aversion import GazeAversionTracker

    tracker = GazeAversionTracker(window_seconds=10)
    ts = NOW

    # Simulate: 2 s contact, 0.5 s aversion, 2 s contact, 0.5 s aversion
    sequence = (
        [(0.8, 60)] +   # 2s eye contact
        [(0.2, 15)] +   # 0.5s aversion
        [(0.8, 60)] +   # 2s eye contact
        [(0.2, 15)]     # 0.5s aversion
    )
    last = {}
    for ec, frames in sequence:
        for _ in range(frames):
            last = tracker.update({"eye_contact_score": ec}, ts)
            ts += 33

    check("has gaze_aversion_frequency", "gaze_aversion_frequency" in last)
    check("has current_aversion",        "current_aversion"        in last)
    check("aversion_frequency >= 0",     last["gaze_aversion_frequency"] >= 0.0)
    # After two aversions the frequency should be nonzero
    tracker2 = GazeAversionTracker(window_seconds=30)
    ts2 = NOW
    # Force two complete aversion events
    for _ in range(20):   tracker2.update({"eye_contact_score": 0.9}, ts2); ts2 += 33
    for _ in range(15):   tracker2.update({"eye_contact_score": 0.1}, ts2); ts2 += 33
    for _ in range(20):   tracker2.update({"eye_contact_score": 0.9}, ts2); ts2 += 33
    for _ in range(15):   r = tracker2.update({"eye_contact_score": 0.1}, ts2); ts2 += 33
    # End aversion
    for _ in range(5):    r = tracker2.update({"eye_contact_score": 0.9}, ts2); ts2 += 33
    check("frequency > 0 after 2 aversions",
          r["gaze_aversion_frequency"] > 0.0, f"freq={r['gaze_aversion_frequency']:.2f}")


def test_environment():
    print("\n[features/environment]")
    from features.environment import extract_environment_features

    # Dark frame
    dark = np.zeros((480, 640, 3), dtype=np.uint8)
    r = extract_environment_features(dark)
    check("dark → condition 'dark'",  r["lighting_condition"] == "dark")
    check("dark → low score",         r["lighting_score"] < 0.3,
          f"score={r['lighting_score']}")

    # Normal frame (~128 brightness)
    normal = np.full((480, 640, 3), 128, dtype=np.uint8)
    r2 = extract_environment_features(normal)
    check("normal → not dark",        r2["lighting_condition"] in ("normal", "dim"))
    check("normal → score > 0.5",     r2["lighting_score"] > 0.5)
    check("has lighting_uniformity",  "lighting_uniformity" in r2)

    # Bright frame
    bright = np.full((480, 640, 3), 240, dtype=np.uint8)
    r3 = extract_environment_features(bright)
    check("bright → condition 'bright'", r3["lighting_condition"] == "bright")

    # With face_feats
    r4 = extract_environment_features(normal, face_feats={"face_distance_cm": 55.0})
    check("distance passthrough",     r4["distance_from_robot_cm"] == 55.0)

    # Stub fields present
    check("device_usage present",                   "device_usage"                    in r4)
    check("env_distraction_level present",          "environmental_distraction_level" in r4)


def test_body_pose():
    print("\n[features/body_pose]")
    from features.body_pose import extract_body_pose_features

    hf = fake_head_t()
    ff = fake_face_f()

    # No pose model — pass None for world/norm
    r = extract_body_pose_features(pose_world=None, pose_norm=None, face_feats=ff, head_pose=hf)
    check("has neck_angle",          "neck_angle"        in r)
    check("has leaning_direction",   "leaning_direction" in r)
    check("has leaning_intensity",   "leaning_intensity" in r)
    check("pose_model_active False", not r["pose_model_active"])
    check("leaning_intensity [0,1]", 0.0 <= r["leaning_intensity"] <= 1.0)

    # Face to left → leaning left
    ff_left = fake_face_f(cx=0.2)
    r2 = extract_body_pose_features(pose_world=None, pose_norm=None, face_feats=ff_left, head_pose=hf)
    check("cx=0.2 → leaning left",  r2["leaning_direction"] == "left")

    # Face to right
    ff_right = fake_face_f(cx=0.8)
    r3 = extract_body_pose_features(pose_world=None, pose_norm=None, face_feats=ff_right, head_pose=hf)
    check("cx=0.8 → leaning right", r3["leaning_direction"] == "right")


def test_hand_features():
    print("\n[features/hand_features]")
    from features.hand_features import extract_hand_features

    ff = fake_face_f()

    # No model — all None
    r = extract_hand_features(hand_lm_norm=None, hand_lm_world=None,
                               handedness=None, pose_norm=None, face_feats=ff)
    check("has hand_near_face",           "hand_near_face"           in r)
    check("has crossed_arms_probability", "crossed_arms_probability" in r)
    check("pose_model_active False",      not r["pose_model_active"])
    check("hand_model_active False",      not r["hand_model_active"])
    check("hand_near_face False w/o model", not r["hand_near_face"])

    # With fake pose landmarks — hands far from face
    pose_lm = np.zeros((1, 33, 3), dtype=np.float32)
    pose_lm[0, 0]  = [0.5, 0.3, 0.0]   # nose
    pose_lm[0, 15] = [0.1, 0.9, 0.0]   # left wrist (far)
    pose_lm[0, 16] = [0.9, 0.9, 0.0]   # right wrist (far)
    pose_lm[0, 11] = [0.35, 0.5, 0.0]  # left shoulder
    pose_lm[0, 12] = [0.65, 0.5, 0.0]  # right shoulder
    pose_lm[0, 13] = [0.25, 0.7, 0.0]  # left elbow
    pose_lm[0, 14] = [0.75, 0.7, 0.0]  # right elbow
    pose_lm[0, 23] = [0.4,  0.8, 0.0]  # left hip
    pose_lm[0, 24] = [0.6,  0.8, 0.0]  # right hip

    r2 = extract_hand_features(hand_lm_norm=None, hand_lm_world=None,
                                handedness=None, pose_norm=pose_lm, face_feats=ff)
    check("pose active",               r2["pose_model_active"])
    check("wrists far → not near face",not r2["hand_near_face"],
          f"score={r2['hand_to_face_score']:.3f}")
    check("crossed_arms [0,1]",        0.0 <= r2["crossed_arms_probability"] <= 1.0)


def test_derived_state():
    print("\n[state/derived_state]")
    from state.derived_state import compute_derived_state
    from temporal.social_temporal import SocialTemporalTracker

    ss = fake_social_state()
    gt = fake_gaze_t()
    tracker = SocialTemporalTracker()
    ts = NOW
    for _ in range(30):
        soc_t = tracker.update(ss, gt, ts)
        ts += 33

    ft = fake_face_t()
    et = fake_eye_t()

    derived = compute_derived_state(
        ss, soc_t, fake_brow(), fake_aus(), gt, fake_head_t(), ft, et
    )

    expected_dims = [
        "comfort_estimate", "curiosity_estimate", "cognitive_load_estimate",
        "social_openness_estimate", "emotional_stability", "uncertainty_estimate",
        "dominance", "emotional_intensity", "emotion_transition_rate",
        "visual_energy", "visual_social_openness", "visual_interest",
        "system_confidence", "engagement_trend", "attention_trend",
        "face_stability", "movement_energy", "restlessness_score",
    ]
    for key in expected_dims:
        check(f"has {key}", key in derived, f"missing from derived")

    # Dim values
    for name in ["comfort_estimate","curiosity_estimate","cognitive_load_estimate",
                 "social_openness_estimate","dominance","emotional_intensity",
                 "emotional_stability","uncertainty_estimate"]:
        dim = derived.get(name, {})
        v = dim.get("value", -1)
        check(f"{name}.value [0,1]", 0.0 <= v <= 1.0, f"got {v}")

    check("visual_energy [0,1]",          0.0 <= derived["visual_energy"]         <= 1.0)
    check("visual_social_openness [0,1]", 0.0 <= derived["visual_social_openness"] <= 1.0)
    check("system_confidence [0,1]",      0.0 <= derived["system_confidence"]      <= 1.0)


def test_json_output_complete():
    print("\n[outputs/json_output — complete]")
    import json
    from outputs.json_output import format_social_state
    from state.derived_state import compute_derived_state
    from temporal.social_temporal import SocialTemporalTracker
    from temporal.face_temporal import FaceTemporalTracker

    ss  = fake_social_state()
    gt  = fake_gaze_t()

    soc_tracker  = SocialTemporalTracker()
    face_tracker = FaceTemporalTracker()
    ts = NOW
    for _ in range(30):
        soc_t  = soc_tracker.update(ss, gt, ts)
        face_t = face_tracker.update(fake_face_f(), ts)
        ts += 33

    derived = compute_derived_state(
        ss, soc_t, fake_brow(), fake_aus(), gt, fake_head_t(), face_t, fake_eye_t()
    )
    full_state = {**ss, **derived}

    js  = format_social_state(full_state, timestamp_ms=NOW)
    obj = json.loads(js)

    check("has timestamp_ms",    "timestamp_ms"   in obj)
    check("has narrative",       "narrative"      in obj)
    check("has dimensions",      "dimensions"     in obj)
    check("has behavioral",      "behavioral"     in obj)
    check("has temporal",        "temporal"       in obj)
    check("has movement",        "movement"       in obj)
    check("has llm_visual",      "llm_visual"     in obj)

    # LLM-final fields
    llm = obj["llm_visual"]
    for field in ["visual_engagement","visual_attention","visual_energy",
                  "visual_fatigue","visual_stress_hints","visual_confidence",
                  "visual_social_openness","visual_discomfort","visual_interest",
                  "visual_emotional_valence","visual_emotional_arousal","system_confidence"]:
        check(f"llm_visual.{field}", field in llm, f"missing")
        v = llm.get(field, -1)
        check(f"llm.{field} [0,1]", 0.0 <= v <= 1.0, f"got {v}")

    # Extended dimensions in output
    for dim in ["comfort_estimate","curiosity_estimate","cognitive_load_estimate",
                "social_openness_estimate","dominance"]:
        check(f"dim {dim} in output", dim in obj["dimensions"])

    # Temporal fields
    temp = obj["temporal"]
    for f in ["engagement_trend","attention_trend","gaze_trend","movement_trend"]:
        check(f"temporal.{f}", f in temp)


def test_full_integration():
    print("\n[full pipeline integration]")
    from features.gaze_aversion   import GazeAversionTracker
    from features.body_pose       import extract_body_pose_features
    from features.hand_features   import extract_hand_features
    from features.environment     import extract_environment_features
    from temporal.face_temporal   import FaceTemporalTracker
    from temporal.body_temporal   import BodyTemporalTracker
    from temporal.hand_temporal   import HandTemporalTracker
    from temporal.social_temporal import SocialTemporalTracker
    from state.social_state       import estimate_social_state
    from state.derived_state      import compute_derived_state
    from outputs.json_output      import format_social_state
    import json

    from temporal.eye_temporal       import BlinkTracker
    from temporal.head_pose_temporal import HeadPoseTracker
    from temporal.gaze_temporal      import GazeTracker
    from temporal.mouth_temporal     import MouthTracker

    blink    = BlinkTracker()
    head_tr  = HeadPoseTracker(fps=30)
    gaze_tr  = GazeTracker()
    mouth_tr = MouthTracker(fps=30)
    face_tr  = FaceTemporalTracker()
    body_tr  = BodyTemporalTracker(fps=30)
    hand_tr  = HandTemporalTracker()
    social_tr= SocialTemporalTracker()
    aversion = GazeAversionTracker()

    rgb = np.full((480, 640, 3), 120, dtype=np.uint8)

    ts = NOW
    for _ in range(50):
        eye_f   = fake_eye_t()
        head_f  = {"head_yaw": 3.0, "head_pitch": -5.0, "head_roll": 1.0,
                   "rotation_vector": [0,0,0], "reprojection_error": 2.0, "confidence": 0.9}
        gaze_f  = fake_gaze_t()
        mouth_f = fake_mouth_t()
        brow_f  = fake_brow()
        aus_f   = fake_aus()
        face_f  = fake_face_f()

        eye_t   = blink.update(eye_f,     ts)
        head_t  = head_tr.update(head_f,  ts)
        gaze_t  = gaze_tr.update(gaze_f,  ts)
        mouth_t = mouth_tr.update(mouth_f, ts)
        face_t  = face_tr.update(face_f,  ts)

        # New: body/hand/env with no models active (None inputs)
        body_f  = extract_body_pose_features(pose_world=None, pose_norm=None,
                                              face_feats=face_f, head_pose=head_f)
        body_t  = body_tr.update(None, body_f, ts)
        hand_f  = extract_hand_features(hand_lm_norm=None, hand_lm_world=None,
                                         handedness=None, pose_norm=None, face_feats=face_f)
        hand_t  = hand_tr.update(hand_f, ts)
        av_t    = aversion.update(gaze_t, ts)
        env_f   = extract_environment_features(rgb, detections=[], face_feats=face_f)

        base    = estimate_social_state(eye_t, head_t, gaze_t, mouth_t, brow_f, aus_f, face_f)
        soc_t   = social_tr.update(base, gaze_t, ts)
        drv     = compute_derived_state(base, soc_t, brow_f, aus_f, gaze_t, head_t, face_t, eye_t,
                                        body_f, body_t, hand_f, hand_t)

        full = {**base, **drv,
                "gaze_aversion": av_t, "environment": env_f,
                "body": body_f, "hands": hand_f}
        ts += 33

    js  = format_social_state(full, timestamp_ms=ts)
    obj = json.loads(js)
    check("integration: valid JSON",         True)
    check("integration: face_present True",  obj["face_present"])
    check("integration: narrative non-empty",len(obj["narrative"]) > 20)
    check("integration: 9 core dims",
          len([k for k in obj["dimensions"] if k in [
              "engagement","attention","stress_hints","fatigue","confidence_level",
              "discomfort","valence","arousal","interaction_willingness"]]) == 9)
    check("integration: derived dims present",
          all(k in obj["dimensions"] for k in [
              "comfort_estimate","curiosity_estimate","dominance"]))
    check("integration: 12 llm_visual fields", len(obj["llm_visual"]) == 12)
    check("integration: all llm values [0,1]",
          all(0.0 <= v <= 1.0 for v in obj["llm_visual"].values()))


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_face_temporal,
        test_social_temporal,
        test_gaze_aversion,
        test_environment,
        test_body_pose,
        test_hand_features,
        test_derived_state,
        test_json_output_complete,
        test_full_integration,
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
