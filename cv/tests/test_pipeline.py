"""
tests/test_pipeline.py

Unit tests for every module in the social perception pipeline.

Run with:
    python -m pytest tests/test_pipeline.py -v
    # or directly:
    python tests/test_pipeline.py

Tests are pure Python — no camera, no MediaPipe inference needed.
All landmark arrays are synthetically generated.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_landmarks(num_faces=1, num_points=478, seed=42) -> np.ndarray:
    """Deterministic random normalized landmarks."""
    rng = np.random.default_rng(seed)
    lm  = rng.uniform(0.1, 0.9, size=(num_faces, num_points, 2)).astype(np.float32)

    # Plant realistic eye geometry for EAR tests (face 0, left eye)
    from landmarks.indices import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX
    # Left eye: set a ~0.30 EAR in normalized space
    lm[0, LEFT_EYE_EAR_IDX[0]] = [0.30, 0.45]  # outer
    lm[0, LEFT_EYE_EAR_IDX[3]] = [0.40, 0.45]  # inner  → horiz=0.10
    # vertical chords = 0.03 → EAR = 0.06 / 0.10 = 0.30 (approx in norm space)
    lm[0, LEFT_EYE_EAR_IDX[1]] = [0.35, 0.435]
    lm[0, LEFT_EYE_EAR_IDX[4]] = [0.35, 0.465]
    lm[0, LEFT_EYE_EAR_IDX[2]] = [0.35, 0.435]
    lm[0, LEFT_EYE_EAR_IDX[5]] = [0.35, 0.465]

    # Right eye
    lm[0, RIGHT_EYE_EAR_IDX[0]] = [0.60, 0.45]
    lm[0, RIGHT_EYE_EAR_IDX[3]] = [0.70, 0.45]
    lm[0, RIGHT_EYE_EAR_IDX[1]] = [0.65, 0.435]
    lm[0, RIGHT_EYE_EAR_IDX[4]] = [0.65, 0.465]
    lm[0, RIGHT_EYE_EAR_IDX[2]] = [0.65, 0.435]
    lm[0, RIGHT_EYE_EAR_IDX[5]] = [0.65, 0.465]

    return lm


LANDMARKS = make_landmarks()
IMAGE_WH  = (640, 480)
NOW_MS    = int(time.time() * 1000)


# ── Test helpers ──────────────────────────────────────────────────────────────

_passed = 0
_failed = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        print(f"  ✓ {name}")
        _passed += 1
    else:
        print(f"  ✗ {name}{' — ' + detail if detail else ''}")
        _failed += 1


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_math_utils():
    print("\n[geometry/math_utils]")
    from geometry.math_utils import (
        clamp, safe_div, sigmoid, linear_map, euclidean,
        normalize, rolling_slope, zero_crossing_rate, peak_frequency_hz,
    )
    check("clamp low",    clamp(-0.5) == 0.0)
    check("clamp high",   clamp(1.5)  == 1.0)
    check("clamp mid",    clamp(0.4)  == 0.4)
    check("safe_div ok",  abs(safe_div(1.0, 2.0) - 0.5) < 1e-6)
    check("safe_div zero",safe_div(1.0, 0.0) == 0.0)
    check("sigmoid 0",    abs(sigmoid(0.0) - 0.5) < 1e-6)
    check("linear_map",   abs(linear_map(0.5, 0.0, 1.0, 0.0, 100.0) - 50.0) < 1e-4)
    p1 = np.array([0.0, 0.0])
    p2 = np.array([3.0, 4.0])
    check("euclidean",    abs(euclidean(p1, p2) - 5.0) < 1e-5)
    v = np.array([3.0, 4.0])
    u = normalize(v)
    check("normalize len",abs(np.linalg.norm(u) - 1.0) < 1e-6)
    arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    check("rolling_slope pos", rolling_slope(arr) > 0)
    arr2 = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    check("rolling_slope neg", rolling_slope(arr2) < 0)
    sin = np.sin(np.linspace(0, 4 * np.pi, 64)).astype(np.float32)
    f = peak_frequency_hz(sin, fps=32.0)
    check("peak_freq ~1Hz", 0.8 <= f <= 1.2, f"got {f:.3f}")


def test_eye_features():
    print("\n[features/eye_features]")
    from features.eye_features import extract_eye_features, compute_ear
    from landmarks.indices import LEFT_EYE_EAR_IDX

    face = LANDMARKS[0]
    ear  = compute_ear(face, LEFT_EYE_EAR_IDX)
    check("EAR > 0", ear > 0)
    check("EAR < 1", ear < 1.0)

    result = extract_eye_features(LANDMARKS, face_idx=0, image_wh=IMAGE_WH)
    check("has left_EAR",      "left_EAR"       in result)
    check("has blink_detected","blink_detected"  in result)
    check("metric_space=pixel",result["metric_space"] == "pixel")
    check("confidence=1.0",    result["ear_confidence"] == 1.0)

    # No image_wh → normalized
    r2 = extract_eye_features(LANDMARKS, face_idx=0)
    check("normalized fallback", r2["metric_space"] == "normalized")

    # Invalid face_idx
    r3 = extract_eye_features(LANDMARKS, face_idx=99)
    check("invalid idx → confidence 0", r3["ear_confidence"] == 0.0)


def test_blink_tracker():
    print("\n[temporal/eye_temporal]")
    from temporal.eye_temporal import BlinkTracker
    from features.eye_features import extract_eye_features

    tracker = BlinkTracker(window_seconds=10)
    # 5 frames: open, open, blink, open, open
    ears = [0.30, 0.30, 0.08, 0.30, 0.30]

    def fake_eye_feats(avg_ear):
        return {
            "left_EAR": avg_ear, "right_EAR": avg_ear, "avg_EAR": avg_ear,
            "blink_detected": avg_ear < 0.21, "ear_confidence": 1.0, "metric_space": "pixel",
        }

    ts = 1000
    states = []
    for ear in ears:
        state = tracker.update(fake_eye_feats(ear), ts)
        states.append(state)
        ts += 100

    check("has blink_rate",     "blink_rate" in states[-1])
    check("has ear_mean",       "ear_mean"   in states[-1])
    check("blink dur set",      states[-1]["avg_blink_duration_ms"] > 0,
          f"dur={states[-1]['avg_blink_duration_ms']}")
    check("window_seconds=10",  states[-1]["window_seconds"] == 10.0)


def test_face_features():
    print("\n[features/face_features]")
    from features.face_features import extract_face_features

    result = extract_face_features(LANDMARKS, face_idx=0, image_wh=IMAGE_WH)
    check("face_presence True",  result["face_presence"])
    check("face_count 1",        result["face_count"] == 1)
    check("bbox is list",        isinstance(result["face_bbox"], list))
    check("bbox 4 elements",     len(result["face_bbox"]) == 4)
    check("distance > 0",        result["face_distance_cm"] > 0)
    check("center in range",
          0.0 <= result["face_center"][0] <= 1.0 and
          0.0 <= result["face_center"][1] <= 1.0)

    # No face
    r2 = extract_face_features(LANDMARKS, face_idx=5)
    check("no face → not present", not r2["face_presence"])


def test_head_pose():
    print("\n[features/head_pose]")
    from features.head_pose import extract_head_pose

    r = extract_head_pose(LANDMARKS, face_idx=0, image_wh=IMAGE_WH)
    check("has head_yaw",         "head_yaw"   in r)
    check("has head_pitch",       "head_pitch"  in r)
    check("has reprojection_err", "reprojection_error" in r)
    check("confidence [0,1]",     0.0 <= r["confidence"] <= 1.0)

    # No image_wh → empty result
    r2 = extract_head_pose(LANDMARKS, face_idx=0, image_wh=None)
    check("no image_wh → conf=0", r2["confidence"] == 0.0)


def test_gaze_features():
    print("\n[features/gaze_features]")
    from features.gaze_features import extract_gaze_features

    r = extract_gaze_features(LANDMARKS, face_idx=0)
    check("has avg_gaze_x",        "avg_gaze_x"       in r)
    check("has eye_contact_score",  "eye_contact_score" in r)
    check("gaze_direction is str",  isinstance(r["gaze_direction"], str))
    check("gaze_x in [-1,1]",       -1.0 <= r["avg_gaze_x"] <= 1.0)
    check("gaze_y in [-1,1]",       -1.0 <= r["avg_gaze_y"] <= 1.0)
    check("eye_contact [0,1]",      0.0 <= r["eye_contact_score"] <= 1.0)


def test_mouth_features():
    print("\n[features/mouth_features]")
    from features.mouth_features import extract_mouth_features

    r = extract_mouth_features(LANDMARKS, face_idx=0, image_wh=IMAGE_WH)
    check("has MAR",              "MAR"             in r)
    check("has smile_intensity",  "smile_intensity" in r)
    check("has lip_tension",      "lip_tension"     in r)
    check("MAR >= 0",             r["MAR"] >= 0.0)
    check("smile [0,1]",          0.0 <= r["smile_intensity"] <= 1.0)


def test_brow_features():
    print("\n[features/brow_features]")
    from features.brow_features import extract_brow_features

    r = extract_brow_features(LANDMARKS, face_idx=0, image_wh=IMAGE_WH)
    check("has brow_raise",     "brow_raise"       in r)
    check("has brow_tension",   "brow_tension"     in r)
    check("raise [0,1]",        0.0 <= r["brow_raise"]   <= 1.0)
    check("tension [0,1]",      0.0 <= r["brow_tension"] <= 1.0)
    check("asymmetry [0,1]",    0.0 <= r["brow_asymmetry"] <= 1.0)


def test_action_units():
    print("\n[features/action_units]")
    from features.action_units import estimate_action_units
    from features.eye_features   import extract_eye_features
    from features.mouth_features import extract_mouth_features
    from features.brow_features  import extract_brow_features

    eye_f   = extract_eye_features(LANDMARKS,  0, IMAGE_WH)
    mouth_f = extract_mouth_features(LANDMARKS, 0, IMAGE_WH)
    brow_f  = extract_brow_features(LANDMARKS,  0, IMAGE_WH)
    r = estimate_action_units(eye_f, mouth_f, brow_f, LANDMARKS, 0, IMAGE_WH)

    for au in ["AU4","AU6","AU7","AU12","AU15","AU23","AU24","AU26","AU43"]:
        check(f"{au} [0,1]", 0.0 <= r[au] <= 1.0, f"{au}={r[au]}")
    check("facial_tension [0,1]",      0.0 <= r["facial_tension"] <= 1.0)
    check("expression_intensity [0,1]",0.0 <= r["expression_intensity"] <= 1.0)


def test_head_pose_tracker():
    print("\n[temporal/head_pose_temporal]")
    from temporal.head_pose_temporal import HeadPoseTracker

    tracker = HeadPoseTracker(fps=30)
    fake_pose = {
        "head_yaw": 5.0, "head_pitch": -3.0, "head_roll": 0.5,
        "rotation_vector": [0.0, 0.0, 0.0],
        "reprojection_error": 2.0, "confidence": 0.9,
    }
    ts = NOW_MS
    for _ in range(10):
        state = tracker.update(fake_pose, ts)
        ts += 33

    check("has yaw",              "yaw"                 in state)
    check("has attention_score",  "attention_score"     in state)
    check("attention_dir is str", isinstance(state["attention_direction"], str))
    check("stability [0,1]",      0.0 <= state["head_stability"] <= 1.0)
    check("attention [0,1]",      0.0 <= state["attention_score"] <= 1.0)


def test_gaze_tracker():
    print("\n[temporal/gaze_temporal]")
    from temporal.gaze_temporal import GazeTracker

    tracker = GazeTracker()
    fake_gaze = {
        "avg_gaze_x": 0.05, "avg_gaze_y": -0.02,
        "gaze_direction": "center",
        "eye_contact_score": 0.85, "gaze_asymmetry": 0.10,
        "confidence": 1.0,
    }
    ts = NOW_MS
    for _ in range(20):
        state = tracker.update(fake_gaze, ts)
        ts += 33

    check("has gaze_volatility",    "gaze_volatility"    in state)
    check("has fixation_active",    "fixation_active"    in state)
    check("has gaze_pattern",       "gaze_pattern"       in state)
    check("has eye_contact_rate",   "eye_contact_rate"   in state)
    check("ec_rate [0,1]",          0.0 <= state["eye_contact_rate"] <= 1.0)
    check("volatility [0,1]",       0.0 <= state["gaze_volatility"]  <= 1.0)


def test_mouth_tracker():
    print("\n[temporal/mouth_temporal]")
    from temporal.mouth_temporal import MouthTracker

    tracker = MouthTracker(fps=30)
    fake_mouth = {
        "MAR": 0.12, "smile_intensity": 0.4, "lip_tension": 0.1,
        "lip_compression": 0.05, "jaw_drop": 0.1, "mouth_open": False,
        "confidence": 1.0,
    }
    ts = NOW_MS
    for _ in range(30):
        state = tracker.update(fake_mouth, ts)
        ts += 33

    check("has speaking_activity",  "speaking_activity"  in state)
    check("has yawning",            "yawning"            in state)
    check("has tension_trend",      "tension_trend"      in state)
    check("MAR > 0",                state["MAR"] > 0)


def test_social_state():
    print("\n[state/social_state]")
    from state.social_state import estimate_social_state

    def fake_eye_t():
        return {"blink_rate": 15.0, "avg_EAR": 0.28, "blink_trend": 0.1,
                "eye_open_duration_ms": 1000.0, "confidence": 0.9}
    def fake_head_t():
        return {"attention_score": 0.85, "attention_direction": "forward",
                "head_stability": 0.9, "nod_detected": False, "yaw": 3.0,
                "pitch": -5.0, "confidence": 0.9}
    def fake_gaze_t():
        return {"eye_contact_score": 0.80, "eye_contact_rate": 0.75,
                "gaze_volatility": 0.10, "gaze_pattern": "fixating",
                "fixation_active": True, "confidence": 1.0}
    def fake_mouth_t():
        return {"smile_intensity": 0.45, "speaking_activity": 0.0, "yawning": False,
                "lip_tension": 0.1, "tension_trend": 0.0, "confidence": 1.0}
    def fake_brow():
        return {"brow_raise": 0.2, "brow_lower": 0.1, "brow_tension": 0.1,
                "brow_asymmetry": 0.05, "inner_brow_angle": 2.0, "confidence": 0.9}
    def fake_aus():
        return {"AU4": 0.1, "AU6": 0.3, "AU7": 0.1, "AU12": 0.45,
                "AU15": 0.05, "AU23": 0.1, "AU24": 0.05, "AU26": 0.1, "AU43": 0.0,
                "facial_tension": 0.1, "expression_intensity": 0.2, "confidence": 0.9}
    def fake_face_f():
        return {"face_presence": True, "face_count": 1, "face_distance_cm": 55.0,
                "face_center": [0.5, 0.5], "confidence": 1.0}

    state = estimate_social_state(
        fake_eye_t(), fake_head_t(), fake_gaze_t(),
        fake_mouth_t(), fake_brow(), fake_aus(), fake_face_f()
    )

    check("face_present True",     state["face_present"])
    for dim in ["engagement","attention","stress_hints","fatigue",
                "confidence_level","discomfort","valence","arousal",
                "interaction_willingness"]:
        v = state[dim]["value"]
        c = state[dim]["confidence"]
        check(f"{dim}.value [0,1]",     0.0 <= v <= 1.0, f"got {v}")
        check(f"{dim}.confidence [0,1]",0.0 <= c <= 1.0, f"got {c}")

    # Engaged scenario → engagement should be reasonably high
    check("engagement > 0.5", state["engagement"]["value"] > 0.5,
          f"got {state['engagement']['value']:.3f}")
    check("interaction_willingness > 0.4",
          state["interaction_willingness"]["value"] > 0.4,
          f"got {state['interaction_willingness']['value']:.3f}")


def test_json_output():
    print("\n[outputs/json_output]")
    import json
    from outputs.json_output import format_social_state, parse_social_state
    from state.social_state import _absent_state

    state = _absent_state()
    js = format_social_state(state, timestamp_ms=NOW_MS)
    check("valid JSON", True)   # would raise if not
    obj = json.loads(js)

    check("has face_present",   "face_present"  in obj)
    check("has narrative",      "narrative"     in obj)
    check("has dimensions",     "dimensions"    in obj)
    check("has behavioral",     "behavioral"    in obj)
    check("has meta",           "meta"          in obj)
    check("narrative is str",   isinstance(obj["narrative"], str))

    # Pretty print
    js_pretty = format_social_state(state, pretty=True)
    check("pretty JSON parseable", json.loads(js_pretty) is not None)

    # Round-trip
    state2 = parse_social_state(js)
    check("round-trip face_present", state2["face_present"] == False)


def test_timing():
    print("\n[utils/timing]")
    from utils.timing import FPSCounter, RateGate, PipelineTimer

    fps = FPSCounter(window=10)
    for _ in range(5):
        fps.tick()
        time.sleep(0.01)
    check("fps > 0", fps.fps > 0)
    check("frame_time_ms > 0", fps.frame_time_ms > 0)

    gate = RateGate(max_hz=100.0)
    check("gate allow first", gate.allow())
    gate2 = RateGate(max_hz=0.001)
    gate2.allow()
    check("gate block second",not gate2.allow())

    timer = PipelineTimer(enabled=True, window=5)
    timer.start("test")
    time.sleep(0.005)
    timer.end("test")
    report = timer.report()
    check("report has test",  "test" in report)
    check("mean_ms > 0",      report["test"]["mean_ms"] > 0)

    # Disabled timer → no overhead, empty report
    t2 = PipelineTimer(enabled=False)
    t2.start("x"); t2.end("x")
    check("disabled → empty report", t2.report() == {})


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_math_utils,
        test_eye_features,
        test_blink_tracker,
        test_face_features,
        test_head_pose,
        test_gaze_features,
        test_mouth_features,
        test_brow_features,
        test_action_units,
        test_head_pose_tracker,
        test_gaze_tracker,
        test_mouth_tracker,
        test_social_state,
        test_json_output,
        test_timing,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  ✗ EXCEPTION in {t.__name__}: {e}")
            _failed += 1

    total = _passed + _failed
    print(f"\n{'='*50}")
    print(f"Results: {_passed}/{total} passed  |  {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)
