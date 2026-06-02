"""
tests/test_new_modules.py

Tests for: camera/capture.py, mediapipe_models/loader.py,
           fusion/signal_fusion.py, smooth.py, face_mesh.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time

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


# ── smooth.py ──────────────────────────────────────────────────────────────────

def test_smooth():
    print("\n[smooth]")
    from smooth import smooth_landmarks
    curr = np.ones((1, 10, 2), dtype=np.float32) * 0.8
    prev = np.ones((1, 10, 2), dtype=np.float32) * 0.4

    smoothed, prev_out = smooth_landmarks(curr, prev, alpha=0.5)
    check("shape preserved",    smoothed.shape == curr.shape)
    check("EMA value correct",  abs(smoothed[0, 0, 0] - 0.6) < 1e-5,
          f"got {smoothed[0,0,0]:.4f}")
    check("returns copy",       smoothed is not prev_out)

    # First frame — no prev
    s2, p2 = smooth_landmarks(curr, None)
    check("first frame = curr", np.allclose(s2, curr))

    # Shape mismatch resets
    curr2 = np.ones((2, 10, 2), dtype=np.float32)
    s3, _ = smooth_landmarks(curr2, prev)
    check("shape mismatch → passthrough", np.allclose(s3, curr2))


# ── face_mesh.py ───────────────────────────────────────────────────────────────

def test_face_mesh():
    print("\n[face_mesh]")
    from face_mesh import draw_landmarks_on_image
    rgb  = np.zeros((480, 640, 3), dtype=np.uint8)
    rng  = np.random.default_rng(0)
    lm   = rng.uniform(0.1, 0.9, (1, 478, 2)).astype(np.float32)
    out  = draw_landmarks_on_image(rgb, lm)
    check("output shape same",   out.shape == rgb.shape)
    check("output is new array", out is not rgb)
    check("draws lines (not all zero)", out.max() > 0)

    # Multi-face
    lm2 = rng.uniform(0.1, 0.9, (2, 478, 2)).astype(np.float32)
    out2 = draw_landmarks_on_image(rgb, lm2)
    check("multi-face no crash",  out2.shape == rgb.shape)


# ── fusion/signal_fusion.py ────────────────────────────────────────────────────

def test_signal_fusion():
    print("\n[fusion/signal_fusion]")
    from fusion.signal_fusion import (
        weighted_mean, fuse_signals,
        TemporalDampener, HysteresisFilter, MultiSignalFuser
    )

    # weighted_mean
    v = weighted_mean([1.0, 0.0], [0.9, 0.1])
    check("weighted_mean dominant", abs(v - 0.9) < 1e-5, f"got {v:.4f}")
    check("weighted_mean zero-w",   weighted_mean([1.0], [0.0]) == 0.0)

    # fuse_signals
    val, conf = fuse_signals([
        (0.8, 0.5, 0.9),
        (0.2, 0.5, 0.1),   # low confidence — downweighted
    ])
    check("high-conf dominates", val > 0.5, f"val={val:.3f}")
    check("confidence [0,1]",    0.0 <= conf <= 1.0)
    check("empty → (0,0)",       fuse_signals([]) == (0.0, 0.0))

    # TemporalDampener
    d = TemporalDampener(rise_alpha=0.5, fall_alpha=0.1)
    v1 = d.update(1.0)
    check("dampener rises fast",  v1 >= 0.4, f"v1={v1:.3f}")
    v2 = d.update(0.0)
    check("dampener falls slow",  v2 > 0.2,  f"v2={v2:.3f}")
    d.reset()
    check("dampener reset → 0",   d.value == 0.0)

    # HysteresisFilter
    hf = HysteresisFilter(low_thresh=0.3, high_thresh=0.7)
    check("starts False",       not hf.state)
    check("below high → stays", not hf.update(0.5))
    check("above high → True",  hf.update(0.8))
    check("above low → stays",  hf.update(0.5))
    check("below low → False",  not hf.update(0.2))

    # MultiSignalFuser
    mf = MultiSignalFuser(dampener=TemporalDampener(0.5, 0.1))
    mf.push("a", 0.8, 0.6, 0.9)
    mf.push("b", 0.2, 0.4, 0.2)
    fv, fc = mf.fuse()
    check("fuser value [0,1]",  0.0 <= fv <= 1.0, f"fv={fv:.3f}")
    check("fuser conf [0,1]",   0.0 <= fc <= 1.0)
    # After fuse, signals cleared
    fv2, fc2 = mf.fuse()
    check("fuser clears after fuse", fv2 == 0.0)


# ── camera/capture.py (no real camera — test API only) ─────────────────────────

def test_camera_api():
    print("\n[camera/capture — API only, no hardware]")
    from camera.capture import CameraCapture

    cam = CameraCapture(device=99, width=320, height=240, max_retries=1, retry_delay_s=0.0)
    result = cam.open()
    check("open non-existent → False", not result)
    check("is_open False after failed open", not cam.is_open())

    # read() with no open device → (None, timestamp)
    frame, ts = cam.read()
    check("read without open → None",  frame is None)
    check("read returns timestamp",     ts > 0)

    cam.release()   # should not raise
    check("release after failed open → safe", True)

    # frame_shape reflects requested dims even before open
    cam2 = CameraCapture(device=0, width=1280, height=720)
    check("frame_shape reflects requested w", cam2.frame_shape[0] == 1280)
    check("frame_shape reflects requested h", cam2.frame_shape[1] == 720)


# ── mediapipe_models/loader.py ─────────────────────────────────────────────────

def test_mediapipe_loader():
    print("\n[mediapipe_models/loader]")
    from mediapipe_models.loader import (
        _resolve_model_path, landmarks_to_numpy, make_mp_image
    )
    import numpy as np

    # _resolve_model_path — should find the model in project root
    try:
        path = _resolve_model_path()
        check("model found",     os.path.isfile(path))
        check("model is .task",  path.endswith(".task"))
        check("model size > 1MB", os.path.getsize(path) > 1_000_000)
    except FileNotFoundError as e:
        check("model found", False, str(e))

    # make_mp_image — wrap a numpy frame
    import mediapipe as mp
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    img = make_mp_image(rgb)
    check("make_mp_image type",  isinstance(img, mp.Image))
    check("make_mp_image height", img.height == 480)
    check("make_mp_image width",  img.width  == 640)

    # landmarks_to_numpy — with empty result (no faces)
    class FakeResult:
        face_landmarks = []
    result = landmarks_to_numpy(FakeResult())
    check("no faces → None", result is None)


# ── Integration: all imports together ─────────────────────────────────────────

def test_all_imports():
    print("\n[integration — all imports]")
    modules = [
        "smooth",
        "face_mesh",
        "camera.capture",
        "mediapipe_models.loader",
        "fusion.signal_fusion",
        "geometry.math_utils",
        "landmarks.indices",
        "features.eye_features",
        "features.face_features",
        "features.head_pose",
        "features.gaze_features",
        "features.mouth_features",
        "features.brow_features",
        "features.action_units",
        "temporal.eye_temporal",
        "temporal.head_pose_temporal",
        "temporal.gaze_temporal",
        "temporal.mouth_temporal",
        "state.social_state",
        "outputs.json_output",
        "utils.timing",
    ]
    for m in modules:
        try:
            __import__(m)
            check(f"import {m}", True)
        except Exception as e:
            check(f"import {m}", False, str(e))


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_smooth,
        test_face_mesh,
        test_signal_fusion,
        test_camera_api,
        test_mediapipe_loader,
        test_all_imports,
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
