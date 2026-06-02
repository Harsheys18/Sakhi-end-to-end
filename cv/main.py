"""
main.py — Emotional Robot Visual Perception Pipeline (production vision)

Realtime webcam → MediaPipe → features → temporal → state → vision_v1 JSON.

Output: event-driven vision_v1 JSON to stdout (one object per emission).
Personal baseline calibration runs automatically within each session.

Usage:
    python main.py                          # vision_v1 JSON, event-driven
    python main.py --force-rate 2.0         # fixed 2 Hz emission
    python main.py --no-display --profile   # headless + latency breakdown
    python main.py --legacy-compact         # old text context (debug)
"""
LATEST_VISION_JSON = None

import argparse
import json
import sys
import os

from camera.capture              import CameraCapture
from mediapipe_models.loader     import (
    build_detector, landmarks_to_numpy, make_mp_image,
    build_pose_detector, pose_to_numpy, pose_world_to_numpy,
    build_hand_detector, hands_to_numpy, hand_world_to_numpy, hand_handedness,
    build_object_detector, detections_to_list,
)
from smooth                      import smooth_landmarks
from visualization.draw_overlays import draw_all

from features.eye_features       import extract_eye_features
from features.face_features      import extract_face_features
from features.head_pose          import extract_head_pose
from features.gaze_features      import extract_gaze_features
from features.gaze_aversion      import GazeAversionTracker
from features.mouth_features     import extract_mouth_features
from features.brow_features      import extract_brow_features
from features.action_units       import estimate_action_units
from features.body_pose          import extract_body_pose_features
from features.hand_features      import extract_hand_features
from features.environment        import extract_environment_features
from features.affective_embedding import extract_affective_embedding

from temporal.eye_temporal       import BlinkTracker
from temporal.head_pose_temporal import HeadPoseTracker
from temporal.gaze_temporal      import GazeTracker
from temporal.mouth_temporal     import MouthTracker
from temporal.face_temporal      import FaceTemporalTracker
from temporal.body_temporal      import BodyTemporalTracker
from temporal.hand_temporal      import HandTemporalTracker
from temporal.social_temporal    import SocialTemporalTracker
from temporal.affective_gru      import AffectiveGRU

from state.social_state          import estimate_social_state
from state.derived_state         import compute_derived_state
from state.uncertainty           import UncertaintyTracker
from state.emotional_dynamics    import EmotionalDynamicsTracker
from state.social_cognition      import SocialCognitionTracker
from state.support_signals       import estimate_support_signals
from state.baseline_tracker      import PersonalBaselineTracker

from outputs.vision_output       import format_vision_v1, new_session_id
from outputs.event_trigger       import EmissionController
from outputs.llm_context         import format_llm_context

from utils.timing                import FPSCounter, RateGate, PipelineTimer

import cv2 as cv
import numpy as np


def req_vision():
    global LATEST_VISION_JSON

    if LATEST_VISION_JSON is None:
        return False

    with open("cv_output.json", "w", encoding="utf-8") as f:
        f.write(LATEST_VISION_JSON)

    return True



def parse_args():
    p = argparse.ArgumentParser(description="Emotional Robot Visual Pipeline")
    p.add_argument("--camera",      type=int,   default=0)
    p.add_argument("--width",       type=int,   default=640)
    p.add_argument("--height",      type=int,   default=480)
    p.add_argument("--alpha",       type=float, default=0.4)
    p.add_argument("--fps",         type=float, default=30.0)
    p.add_argument("--num-faces",   type=int,   default=1)
    p.add_argument("--obj-rate",    type=float, default=2.0)
    p.add_argument("--force-rate",  type=float, default=0.0,
                   help="Override event-driven with fixed Hz (0=event-driven)")
    p.add_argument("--output",      type=str,   default="json",
                   choices=["json", "legacy-compact", "legacy-dict"],
                   help="stdout format (default: vision_v1 JSON)")
    p.add_argument("--baseline-file", type=str, default=None,
                   help="Optional JSON with long-term visual baselines")
    p.add_argument("--include-baseline", action="store_true",
                   help="Embed baseline snapshot in each JSON emission")
    p.add_argument("--session-id",  type=str, default="",
                   help="Session ID (auto-generated if omitted)")
    p.add_argument("--no-display",  action="store_true")
    p.add_argument("--fps-overlay", action="store_true")
    p.add_argument("--profile",     action="store_true")
    return p.parse_args()


class FaceTrackers:
    def __init__(self, fps, external_baselines=None):
        self.blink    = BlinkTracker()
        self.head     = HeadPoseTracker(fps=fps)
        self.gaze     = GazeTracker()
        self.mouth    = MouthTracker(fps=fps)
        self.face     = FaceTemporalTracker()
        self.body     = BodyTemporalTracker(fps=fps)
        self.hand     = HandTemporalTracker()
        self.social   = SocialTemporalTracker()
        self.aversion = GazeAversionTracker()
        self.gru      = AffectiveGRU()
        self.uncert   = UncertaintyTracker()
        self.dynamics = EmotionalDynamicsTracker()
        self.cognition= SocialCognitionTracker()
        self.baseline = PersonalBaselineTracker()
        if external_baselines:
            self.baseline.load_external(external_baselines)

    def reset(self):
        for t in [self.blink, self.head, self.gaze, self.mouth, self.face,
                  self.body, self.hand, self.social, self.aversion,
                  self.gru, self.uncert, self.dynamics, self.cognition]:
            t.reset()
        self.baseline.reset()


def _load_baseline_file(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        visual = data.get("visual", data.get("baselines", {}).get("visual", data))
        if isinstance(visual, dict) and "visual" in visual:
            visual = visual["visual"]
        return visual if isinstance(visual, dict) else None
    except OSError:
        print(f"WARNING: cannot read baseline file {path}", file=sys.stderr)
        return None


def _draw_hud(frame, state):
    urgency = state.get("support_urgency", "none")
    colors  = {"immediate":(0,0,220),"soon":(0,180,220),"monitor":(0,200,100),"none":(60,60,60)}
    banner_col = colors.get(urgency, (60,60,60))
    cv.rectangle(frame, (0,0), (frame.shape[1], 28), banner_col, -1)
    dist    = state.get("distress_level", 0.0)
    d_type  = state.get("distress_type", "")
    d_str   = f"DISTRESS:{dist:.2f} [{d_type}]" if d_type != "none" else f"Distress:{dist:.2f}"
    base_flag = "BL" if state.get("baseline_active") else "POP"
    cv.putText(frame, f"[{urgency.upper()}|{base_flag}] {d_str}", (8,20),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv.LINE_AA)

    dims = [
        ("ENG",   _gv(state, "engagement")),
        ("VAL",   (state.get("gru_valence", 0.0) + 1.0) / 2.0),
        ("ARO",   state.get("gru_arousal",  0.0)),
        ("SUP",   1.0 - state.get("openness_to_support",   0.0)),
        ("SUPR",  state.get("emotional_suppression",       0.0)),
        ("OVWH",  state.get("overwhelm_probability",       0.0)),
        ("COMS",  state.get("comfort_seeking",             0.0)),
        ("WDRL",  state.get("emotional_withdrawal",        0.0)),
    ]
    x0, y0, bw, bh, gap = 8, frame.shape[0]-20, 52, 8, 66
    for i, (label, val) in enumerate(dims):
        x = x0 + i * gap
        cv.rectangle(frame, (x, y0-bh), (x+bw, y0), (40,40,40), -1)
        fill = int(float(val) * bw)
        col  = (200,60,60) if label in ("SUP","SUPR","OVWH","WDRL") and val > 0.5 \
               else (0,200,80) if val > 0.6 else (200,200,0)
        if fill > 0:
            cv.rectangle(frame, (x, y0-bh), (x+fill, y0), col, -1)
        cv.putText(frame, label, (x, y0-bh-2),
                   cv.FONT_HERSHEY_SIMPLEX, 0.26, (200,200,200), 1, cv.LINE_AA)

    flags = []
    if state.get("speaking"):          flags.append("SPK")
    if state.get("eye_contact"):       flags.append("EYE")
    if state.get("genuine_smile"):     flags.append("SMILE")
    if state.get("stress_escalating"): flags.append("↑STR")
    flags.append(state.get("meta",{}).get("head_direction","?")[:3].upper())
    quad = state.get("gru_affect_quadrant","")
    flags.append(f"[{quad}]")
    cv.putText(frame, "  ".join(flags), (8, frame.shape[0]-32),
               cv.FONT_HERSHEY_SIMPLEX, 0.36, (200,200,255), 1, cv.LINE_AA)


def _gv(state, dim):
    v = state.get(dim)
    return float(v.get("value", 0.0)) if isinstance(v, dict) else float(v or 0.0)


def _update_baseline(tracker: PersonalBaselineTracker, eye_t, eye_f, brow_f, aus, mouth_t, gaze_t, ts_ms):
    tracker.update({
        "blink_rate":           eye_t.get("blink_rate", 0.0),
        "avg_EAR":              eye_t.get("avg_EAR", eye_f.get("avg_EAR", 0.28)),
        "brow_tension":         brow_f.get("brow_tension", 0.0),
        "facial_tension":       aus.get("facial_tension", 0.0),
        "lip_tension":          mouth_t.get("lip_tension", 0.0),
        "gaze_volatility":      gaze_t.get("gaze_volatility", 0.0),
        "AU4":                  aus.get("AU4", 0.0),
        "expression_intensity": aus.get("expression_intensity", 0.0),
    }, ts_ms)


def run(args):
    cam = CameraCapture(device=args.camera, width=args.width, height=args.height,
                        target_fps=args.fps)
    if not cam.open():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        sys.exit(1)
    w, h = cam.frame_shape

    external_baselines = _load_baseline_file(args.baseline_file)
    session_id = args.session_id or new_session_id()

    face_det = build_detector(num_faces=args.num_faces)
    pose_det = build_pose_detector()
    hand_det = build_hand_detector()
    obj_det  = build_object_detector()

    fps_counter   = FPSCounter(window=60)
    obj_gate      = RateGate(max_hz=args.obj_rate)
    force_gate    = RateGate(max_hz=args.force_rate) if args.force_rate > 0 else None
    timer         = PipelineTimer(enabled=args.profile, window=30)
    profile_gate  = RateGate(max_hz=0.2)
    emission_ctrl = EmissionController()

    prev_smoothed = None
    trackers: dict[int, FaceTrackers] = {}
    last_detections = []
    active_baseline = False

    print(f"[vision] session={session_id} | camera {args.camera} {w}x{h}", file=sys.stderr)
    print(f"[vision] output={args.output} | emission={'fixed' if args.force_rate else 'event'}",
          file=sys.stderr)
    print("[vision] Q/ESC to quit\n", file=sys.stderr)

    try:
        while True:
            timer.start("capture")
            rgb_frame, ts_ms = cam.read()
            timer.end("capture")
            if rgb_frame is None:
                continue
            fps_counter.tick()
            mp_img = make_mp_image(rgb_frame)

            timer.start("inference")
            face_r = face_det.detect_for_video(mp_img, ts_ms)
            pose_r = pose_det.detect_for_video(mp_img, ts_ms)
            hand_r = hand_det.detect_for_video(mp_img, ts_ms)
            if obj_gate.allow():
                obj_r = obj_det.detect_for_video(mp_img, ts_ms)
                last_detections = detections_to_list(obj_r)
            timer.end("inference")

            timer.start("landmarks")
            curr_np    = landmarks_to_numpy(face_r)
            pose_norm  = pose_to_numpy(pose_r)
            pose_world = pose_world_to_numpy(pose_r)
            hand_norm  = hands_to_numpy(hand_r)
            hand_world = hand_world_to_numpy(hand_r)
            handed     = hand_handedness(hand_r)
            if curr_np is not None:
                smoothed, prev_smoothed = smooth_landmarks(curr_np, prev_smoothed, args.alpha)
            else:
                smoothed = None
                prev_smoothed = None
            timer.end("landmarks")

            timer.start("features")
            t = None
            if smoothed is not None:
                num_faces = smoothed.shape[0]
                for fi in range(num_faces):
                    if fi not in trackers:
                        trackers[fi] = FaceTrackers(fps=args.fps,
                                                   external_baselines=external_baselines)

                fi = 0
                t = trackers[fi]

                face_f  = extract_face_features(smoothed, fi, image_wh=(w, h))
                eye_f   = extract_eye_features(smoothed, fi, image_wh=(w, h))
                head_f  = extract_head_pose(smoothed, fi, image_wh=(w, h))
                gaze_f  = extract_gaze_features(smoothed, fi)
                mouth_f = extract_mouth_features(smoothed, fi, image_wh=(w, h))
                brow_f  = extract_brow_features(smoothed, fi, image_wh=(w, h))
                aus     = estimate_action_units(eye_f, mouth_f, brow_f, smoothed, fi, (w, h))
                body_f  = extract_body_pose_features(pose_world, pose_norm, face_f, head_f)
                hand_f  = extract_hand_features(hand_norm, hand_world, handed, pose_norm, face_f)

                eye_t   = t.blink.update(eye_f, ts_ms)
                head_t  = t.head.update(head_f, ts_ms)
                gaze_t  = t.gaze.update(gaze_f, ts_ms)
                mouth_t = t.mouth.update(mouth_f, ts_ms)
                face_t  = t.face.update(face_f, ts_ms)
                body_t  = t.body.update(pose_world, body_f, ts_ms)
                hand_t  = t.hand.update(hand_f, ts_ms)
                av_t    = t.aversion.update(gaze_t, ts_ms)

                _update_baseline(t.baseline, eye_t, eye_f, brow_f, aus, mouth_t, gaze_t, ts_ms)
                active_baseline = t.baseline.is_active()

                aff_emb = extract_affective_embedding(aus, eye_f, mouth_f, brow_f, head_t)
                gru_out = t.gru.update(aff_emb, ts_ms)

                base = estimate_social_state(
                    eye_t, head_t, gaze_t, mouth_t, brow_f, aus, face_f,
                    baseline=t.baseline,
                )
                soc_t   = t.social.update(base, gaze_t, ts_ms)
                derived = compute_derived_state(
                    base, soc_t, brow_f, aus, gaze_t, head_t,
                    face_t, eye_t, body_f, body_t, hand_f, hand_t,
                )
                sensor_conf = base.get("meta", {}).get("overall_confidence", 0.5)
                unc_out  = t.uncert.update(base, aff_emb, gru_out, sensor_conf)
                dyn_out  = t.dynamics.update(aff_emb, gru_out, base, ts_ms)
                cog_out  = t.cognition.update(
                    base, head_t, gaze_t, mouth_t, eye_t,
                    aff_emb, body_f, aus, ts_ms,
                )
                sup_out = estimate_support_signals(
                    aus, eye_t, head_t, gaze_t, mouth_t, brow_f, face_f,
                    aff_emb, gru_out, body_f, hand_f, face_t, dyn_out,
                )

                full_state = {
                    **base, **derived,
                    **aff_emb, **gru_out, **dyn_out, **cog_out,
                    **unc_out, **sup_out,
                    "uncertainty":          unc_out,
                    "gaze_aversion":        av_t,
                    "body":                 body_f,
                    "hands":                hand_f,
                    "eye_contact_rate":     gaze_t.get("eye_contact_rate", 0.0),
                    "blink_rate":           eye_t.get("blink_rate", 0.0),
                    "fidget_probability":   body_t.get("fidget_probability", 0.0),
                    "gesture_frequency":    hand_t.get("gesture_frequency", 0.0),
                    "nod_detected":         head_t.get("nod_detected", False),
                    "shake_detected":       head_t.get("shake_detected", False),
                    "yawning":              mouth_t.get("yawning", False),
                    "prolonged_stillness":  face_t.get("prolonged_stillness", False),
                    "prolonged_downward_gaze": soc_t.get("prolonged_downward_gaze", False),
                    "baseline_active":      active_baseline,
                }

                for gone in set(trackers) - set(range(num_faces)):
                    trackers[gone].reset()
                    del trackers[gone]

            else:
                for tr in trackers.values():
                    tr.reset()
                trackers.clear()
                active_baseline = False
                base = estimate_social_state({}, {}, {}, {}, {}, {}, {"face_presence": False})
                sup_out = {
                    "distress_level": 0.0, "distress_type": "none",
                    "support_urgency": "none", "genuine_smile": False,
                    "genuine_smile_score": 0.0, "emotional_suppression": 0.0,
                    "overwhelm_probability": 0.0, "comfort_seeking": 0.0,
                    "emotional_withdrawal": 0.0, "openness_to_support": 0.0,
                }
                full_state = {**base, **sup_out, "baseline_active": False}
            timer.end("features")

            should_emit = False
            emit_reason = ""
            if force_gate is not None:
                should_emit = force_gate.allow()
                emit_reason = "forced_rate"
            else:
                should_emit, emit_reason = emission_ctrl.should_emit(full_state)

            should_emit=True
            if should_emit:
                changed = emission_ctrl.get_changed_dims(full_state)
                if args.output == "json":
                    bl_snap = t.baseline.export_snapshot() if (
                        args.include_baseline and t is not None
                    ) else None
                    output = format_vision_v1(
                        full_state,
                        session_id=session_id,
                        emit_reason=emit_reason,
                        timestamp_ms=ts_ms,
                        baseline_active=full_state.get("baseline_active", False),
                        baseline_snapshot=bl_snap,
                    )
                else:
                    mode = "compact" if args.output == "legacy-compact" else "dict"
                    output = format_llm_context(
                        full_state,
                        changed_dims=changed,
                        emit_reason=emit_reason,
                        mode=mode,
                    )
                global LATEST_VISION_JSON
                LATEST_VISION_JSON = output
                emission_ctrl.mark_emitted(full_state)

            if not args.no_display:
                timer.start("viz")
                display = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
                draw_all(
                    display,
                    face_lm=smoothed,
                    pose_norm=pose_norm,
                    hand_norm=hand_norm,
                    handedness=handed,
                    detections=last_detections,
                )
                if args.fps_overlay:
                    fw = display.shape[1]
                    cv.putText(display, f"FPS:{fps_counter.fps:.0f}",
                               (fw - 80, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                               (200, 255, 200), 1)
                if full_state.get("face_present"):
                    _draw_hud(display, full_state)
                cv.imshow("Emotional Robot Vision", display)
                timer.end("viz")
                if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if args.profile and profile_gate.allow():
                rep   = timer.report()
                total = sum(v["mean_ms"] for v in rep.values())
                print(f"[profile] {total:.1f}ms | " +
                      " | ".join(f"{k}={v['mean_ms']:.1f}" for k, v in rep.items()),
                      file=sys.stderr)

    except KeyboardInterrupt:
        pass
    finally:
        for d in [face_det, pose_det, hand_det, obj_det]:
            d.close()
        cam.release()
        if not args.no_display:
            cv.destroyAllWindows()
        print(f"\n[vision] stopped | session={session_id}", file=sys.stderr)


if __name__ == "__main__":
    run(parse_args())
