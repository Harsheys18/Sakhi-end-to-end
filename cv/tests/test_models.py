"""
tests/test_models.py

Integration tests using the real MediaPipe models.
Requires: pose_landmarker.task, hand_landmarker.task, efficientdet_lite0.tflite

Tests verify:
  - All three new models load correctly
  - Detectors run on synthetic frames without crashing
  - pose_to_numpy / hand_to_numpy / detections_to_list handle empty results
  - Full pipeline: pose → body_pose → body_temporal → derived_state
  - Full pipeline: hand → hand_features → hand_temporal
  - Full pipeline: object detector → environment features
  - All feature keys present and in valid ranges
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
import mediapipe as mp

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


def make_frame(brightness=120) -> np.ndarray:
    """Synthetic RGB frame."""
    return np.full((480, 640, 3), brightness, dtype=np.uint8)


def make_mp_img(brightness=120):
    from mediapipe_models.loader import make_mp_image
    return make_mp_image(make_frame(brightness))


NOW_MS = int(time.time() * 1000)


# ── Model loading ─────────────────────────────────────────────────────────────

def test_model_loading():
    print("\n[model loading]")
    from mediapipe_models.loader import (
        build_pose_detector, build_hand_detector, build_object_detector
    )
    try:
        pd = build_pose_detector(); pd.close()
        check("pose model loads", True)
    except Exception as e:
        check("pose model loads", False, str(e))

    try:
        hd = build_hand_detector(); hd.close()
        check("hand model loads", True)
    except Exception as e:
        check("hand model loads", False, str(e))

    try:
        od = build_object_detector(); od.close()
        check("object model loads", True)
    except Exception as e:
        check("object model loads", False, str(e))


# ── Pose pipeline ─────────────────────────────────────────────────────────────

def test_pose_pipeline():
    print("\n[pose pipeline]")
    from mediapipe_models.loader import (
        build_pose_detector, pose_to_numpy, pose_world_to_numpy
    )
    from features.body_pose    import extract_body_pose_features
    from temporal.body_temporal import BodyTemporalTracker

    det = build_pose_detector()
    img = make_mp_img()

    # Run on blank frame (no person → None results)
    r = det.detect_for_video(img, NOW_MS)
    norm  = pose_to_numpy(r)
    world = pose_world_to_numpy(r)
    check("blank frame → norm None",  norm  is None)
    check("blank frame → world None", world is None)

    # Feature extraction with no landmarks
    body_f = extract_body_pose_features(pose_world=None, pose_norm=None)
    check("no pose → pose_model_active False", not body_f["pose_model_active"])
    check("no pose → has neck_angle",          "neck_angle" in body_f)
    for k in ["spine_angle","shoulder_openness","body_symmetry",
              "leaning_direction","leaning_intensity"]:
        check(f"no pose → has {k}", k in body_f)

    # Temporal with no pose
    tracker = BodyTemporalTracker(fps=30)
    ts = NOW_MS
    for _ in range(10):
        bt = tracker.update(None, body_f, ts)
        ts += 33
    check("body_t has pose_stability",      "pose_stability"      in bt)
    check("body_t has fidget_probability",  "fidget_probability"  in bt)
    check("body_t has movement_energy",     "movement_energy"     in bt)
    check("body_t has restlessness_score",  "restlessness_score"  in bt)
    check("body_t all [0,1]",
          all(0.0 <= bt[k] <= 1.0 for k in
              ["pose_stability","fidget_probability","restlessness_score","movement_variability"]))

    # Test with synthetic pose landmarks (fabricated)
    fake_world = np.zeros((1, 33, 3), dtype=np.float32)
    # Place shoulders, hips in plausible world positions (meters)
    fake_world[0, 11] = [-0.20, 0.45, 0.0]   # left shoulder
    fake_world[0, 12] = [ 0.20, 0.45, 0.0]   # right shoulder
    fake_world[0, 23] = [-0.15, 0.0,  0.0]   # left hip
    fake_world[0, 24] = [ 0.15, 0.0,  0.0]   # right hip
    fake_world[0,  7] = [-0.08, 0.60, 0.0]   # left ear
    fake_world[0,  8] = [ 0.08, 0.60, 0.0]   # right ear

    body_f2 = extract_body_pose_features(pose_world=fake_world, pose_norm=None)
    check("fake pose → pose_model_active True", body_f2["pose_model_active"])
    check("fake pose → spine_angle >= 0",       body_f2["spine_angle"] >= 0.0)
    check("fake pose → shoulder_openness [0,1]",0.0 <= body_f2["shoulder_openness"] <= 1.0)
    check("fake pose → body_symmetry [0,1]",    0.0 <= body_f2["body_symmetry"]     <= 1.0)

    # Body temporal with fake pose
    tracker2 = BodyTemporalTracker(fps=30)
    ts2 = NOW_MS
    for i in range(20):
        # Slightly vary the pose to simulate movement
        fw = fake_world.copy()
        fw[0, 15] = [0.1 + i*0.002, 0.3, 0.0]   # left wrist moving
        fw[0, 16] = [0.1 + i*0.002, 0.3, 0.0]
        bt2 = tracker2.update(fw, body_f2, ts2)
        ts2 += 33
    check("body_t with pose → movement_energy > 0",
          bt2["movement_energy"] > 0.0, f"energy={bt2['movement_energy']:.5f}")

    det.close()


# ── Hand pipeline ─────────────────────────────────────────────────────────────

def test_hand_pipeline():
    print("\n[hand pipeline]")
    from mediapipe_models.loader import (
        build_hand_detector, hands_to_numpy, hand_world_to_numpy, hand_handedness
    )
    from features.hand_features import extract_hand_features
    from temporal.hand_temporal  import HandTemporalTracker

    det = build_hand_detector()
    img = make_mp_img()

    r       = det.detect_for_video(img, NOW_MS)
    norm    = hands_to_numpy(r)
    world   = hand_world_to_numpy(r)
    handed  = hand_handedness(r)
    check("blank → hand norm None",  norm  is None)
    check("blank → handedness []",   handed == [])

    # Features with no hands
    hand_f = extract_hand_features(hand_lm_norm=None, hand_lm_world=None,
                                    handedness=None, pose_norm=None)
    check("no hand → num_hands_visible 0",     hand_f["num_hands_visible"]    == 0)
    check("no hand → hand_near_face False",    not hand_f["hand_near_face"])
    check("no hand → gesture_primitive 'none'",hand_f["gesture_primitive"]    == "none")
    check("no hand → hand_model_active False", not hand_f["hand_model_active"])

    # Synthetic hand landmarks (open palm)
    fake_hand = np.zeros((1, 21, 3), dtype=np.float32)
    # Wrist at 0.5, 0.7; fingertips extended upward
    fake_hand[0, 0]  = [0.5, 0.7, 0.0]   # wrist
    fake_hand[0, 5]  = [0.4, 0.5, 0.0]   # index MCP
    fake_hand[0, 8]  = [0.4, 0.2, 0.0]   # index tip (far from wrist = extended)
    fake_hand[0, 9]  = [0.5, 0.5, 0.0]   # middle MCP
    fake_hand[0, 12] = [0.5, 0.2, 0.0]   # middle tip (extended)
    fake_hand[0, 13] = [0.6, 0.5, 0.0]   # ring MCP
    fake_hand[0, 16] = [0.6, 0.2, 0.0]   # ring tip (extended)
    fake_hand[0, 17] = [0.65, 0.5, 0.0]  # pinky MCP
    fake_hand[0, 20] = [0.65, 0.2, 0.0]  # pinky tip (extended)
    fake_hand[0, 4]  = [0.35, 0.6, 0.0]  # thumb tip

    hand_f2 = extract_hand_features(hand_lm_norm=fake_hand, hand_lm_world=None,
                                     handedness=["Right"])
    check("fake open hand → hand_model_active", hand_f2["hand_model_active"])
    check("fake open hand → open_palm_right",   hand_f2["open_palm_right"])
    check("fake open hand → gesture 'open'",    hand_f2["gesture_primitive"] == "open",
          f"got {hand_f2['gesture_primitive']}")

    # Temporal tracker
    tracker = HandTemporalTracker()
    ts = NOW_MS
    for _ in range(30):
        ht = tracker.update(hand_f2, ts)
        ts += 100   # 10fps for faster gesture accumulation
    check("hand_t has gesture_frequency",       "gesture_frequency"      in ht)
    check("hand_t has hand_to_face_frequency",  "hand_to_face_frequency" in ht)
    check("hand_t has dominant_gesture",        "dominant_gesture"       in ht)

    det.close()


# ── Object detection pipeline ─────────────────────────────────────────────────

def test_object_pipeline():
    print("\n[object detection pipeline]")
    from mediapipe_models.loader import build_object_detector, detections_to_list
    from features.environment    import extract_environment_features

    det = build_object_detector()
    ts  = NOW_MS

    # Dark frame → no detections (but no crash)
    dark = make_mp_img(brightness=0)
    r    = det.detect_for_video(dark, ts)
    dets = detections_to_list(r)
    check("dark frame → list result", isinstance(dets, list))

    # Normal frame
    ts  += 100
    img  = make_mp_img(brightness=128)
    r2   = det.detect_for_video(img, ts)
    dets2 = detections_to_list(r2)
    check("normal frame → list result", isinstance(dets2, list))
    if dets2:
        d = dets2[0]
        check("detection has label", "label" in d)
        check("detection has score", "score" in d and 0.0 <= d["score"] <= 1.0)
        check("detection has bbox",  "bbox"  in d and len(d["bbox"]) == 4)

    # Environment features with real detections
    rgb = make_frame(128)
    env = extract_environment_features(rgb, detections=dets2,
                                        face_feats={"face_distance_cm": 55.0},
                                        image_wh=(640, 480))
    check("env has lighting_condition",             "lighting_condition"              in env)
    check("env has device_usage",                   "device_usage"                    in env)
    check("env has environmental_distraction_level","environmental_distraction_level" in env)
    check("env has object_detector_active True",    env["object_detector_active"])
    check("env device_usage [0,1]",                 0.0 <= env["device_usage"] <= 1.0)
    check("env distraction [0,1]",
          0.0 <= env["environmental_distraction_level"] <= 1.0)
    check("env distance passthrough",               env["distance_from_robot_cm"] == 55.0)

    # Lighting variation tests
    for brightness, expected in [(5, "dark"), (60, "dim"), (128, "normal"), (240, "bright")]:
        env_b = extract_environment_features(make_frame(brightness), detections=[])
        check(f"brightness={brightness} → {expected}",
              env_b["lighting_condition"] == expected,
              f"got '{env_b['lighting_condition']}'")

    det.close()


# ── Full pipeline with all three models ───────────────────────────────────────

def test_full_model_pipeline():
    print("\n[full model pipeline — all four detectors]")
    from mediapipe_models.loader import (
        build_detector, landmarks_to_numpy,
        build_pose_detector, pose_to_numpy, pose_world_to_numpy,
        build_hand_detector, hands_to_numpy, hand_world_to_numpy, hand_handedness,
        build_object_detector, detections_to_list, make_mp_image,
    )
    from smooth                      import smooth_landmarks
    from features.eye_features       import extract_eye_features
    from features.face_features      import extract_face_features
    from features.head_pose          import extract_head_pose
    from features.gaze_features      import extract_gaze_features
    from features.mouth_features     import extract_mouth_features
    from features.brow_features      import extract_brow_features
    from features.action_units       import estimate_action_units
    from features.body_pose          import extract_body_pose_features
    from features.hand_features      import extract_hand_features
    from features.environment        import extract_environment_features
    from temporal.eye_temporal       import BlinkTracker
    from temporal.head_pose_temporal import HeadPoseTracker
    from temporal.gaze_temporal      import GazeTracker
    from temporal.mouth_temporal     import MouthTracker
    from temporal.face_temporal      import FaceTemporalTracker
    from temporal.body_temporal      import BodyTemporalTracker
    from temporal.hand_temporal      import HandTemporalTracker
    from temporal.social_temporal    import SocialTemporalTracker
    from features.gaze_aversion      import GazeAversionTracker
    from state.social_state          import estimate_social_state
    from state.derived_state         import compute_derived_state
    from outputs.json_output         import format_social_state
    import json

    face_det = build_detector()
    pose_det = build_pose_detector()
    hand_det = build_hand_detector()
    obj_det  = build_object_detector()

    # Trackers
    blink    = BlinkTracker()
    head_tr  = HeadPoseTracker(fps=30)
    gaze_tr  = GazeTracker()
    mouth_tr = MouthTracker(fps=30)
    face_tr  = FaceTemporalTracker()
    body_tr  = BodyTemporalTracker(fps=30)
    hand_tr  = HandTemporalTracker()
    social_tr= SocialTemporalTracker()
    aversion = GazeAversionTracker()

    prev_smoothed = None
    full_state    = {}

    ts = NOW_MS
    for i in range(10):
        rgb = make_frame(100 + i * 5)
        img = make_mp_image(rgb)

        face_r = face_det.detect_for_video(img, ts)
        pose_r = pose_det.detect_for_video(img, ts)
        hand_r = hand_det.detect_for_video(img, ts)
        obj_r  = obj_det.detect_for_video(img, ts)

        curr_np  = landmarks_to_numpy(face_r)
        pose_norm  = pose_to_numpy(pose_r)
        pose_world = pose_world_to_numpy(pose_r)
        hand_norm  = hands_to_numpy(hand_r)
        hand_world = hand_world_to_numpy(hand_r)
        handed     = hand_handedness(hand_r)
        dets       = detections_to_list(obj_r)

        if curr_np is not None:
            smoothed, prev_smoothed = smooth_landmarks(curr_np, prev_smoothed, 0.4)
            w, h = 640, 480
            fi   = 0
            face_f  = extract_face_features( smoothed, fi, image_wh=(w,h))
            eye_f   = extract_eye_features(  smoothed, fi, image_wh=(w,h))
            head_f  = extract_head_pose(     smoothed, fi, image_wh=(w,h))
            gaze_f  = extract_gaze_features( smoothed, fi)
            mouth_f = extract_mouth_features(smoothed, fi, image_wh=(w,h))
            brow_f  = extract_brow_features( smoothed, fi, image_wh=(w,h))
            aus     = estimate_action_units(eye_f, mouth_f, brow_f, smoothed, fi, (w,h))
        else:
            face_f = {"face_presence": False, "face_center": [0.5,0.5],
                      "frame_displacement": 0.0, "face_distance_cm": 0.0, "confidence": 0.0}
            eye_f = head_f = gaze_f = mouth_f = brow_f = aus = {}

        body_f = extract_body_pose_features(pose_world, pose_norm, face_f, head_f)
        hand_f = extract_hand_features(hand_norm, hand_world, handed, pose_norm, face_f)
        env_f  = extract_environment_features(rgb, dets, face_f, (640,480))

        eye_t   = blink.update(eye_f,    ts)
        head_t  = head_tr.update(head_f, ts)
        gaze_t  = gaze_tr.update(gaze_f, ts)
        mouth_t = mouth_tr.update(mouth_f, ts)
        face_t  = face_tr.update(face_f, ts)
        body_t  = body_tr.update(pose_world, body_f, ts)
        hand_t  = hand_tr.update(hand_f, ts)
        av_t    = aversion.update(gaze_t, ts)

        base    = estimate_social_state(eye_t, head_t, gaze_t, mouth_t,
                                        brow_f, aus, face_f)
        soc_t   = social_tr.update(base, gaze_t, ts)
        derived = compute_derived_state(base, soc_t, brow_f, aus,
                                        gaze_t, head_t, face_t, eye_t,
                                        body_f, body_t, hand_f, hand_t)

        full_state = {**base, **derived,
                      "gaze_aversion": av_t,
                      "environment":   env_f,
                      "body":          body_f,
                      "hands":         hand_f}
        ts += 33

    for d in [face_det, pose_det, hand_det, obj_det]:
        d.close()

    check("full pipeline completed 10 frames", True)
    check("full_state face_present key exists", "face_present" in full_state)
    check("env in full_state",  "environment" in full_state)
    check("body in full_state", "body"        in full_state)
    check("hands in full_state","hands"       in full_state)

    js  = format_social_state(full_state, timestamp_ms=ts)
    obj = json.loads(js)
    check("final JSON valid",          isinstance(obj, dict))
    check("JSON has llm_visual",       "llm_visual" in obj)
    check("JSON has temporal section", "temporal"   in obj)
    check("JSON has movement section", "movement"   in obj)
    check("all llm_visual [0,1]",
          all(0.0 <= v <= 1.0 for v in obj["llm_visual"].values()))

    # Feature completeness check
    required_derived = [
        "fidget_probability","gesture_frequency","hand_to_face_frequency",
        "pose_stability","movement_energy","curiosity_estimate",
        "cognitive_load_estimate","dominance","comfort_estimate",
    ]
    for k in required_derived:
        check(f"derived has {k}", k in full_state, f"missing")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_model_loading,
        test_pose_pipeline,
        test_hand_pipeline,
        test_object_pipeline,
        test_full_model_pipeline,
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
