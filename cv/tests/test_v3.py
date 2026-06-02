"""tests/test_v3.py — Tests for v3 support signals, event trigger, LLM context."""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

_passed = 0; _failed = 0
NOW_MS = int(time.time() * 1000)

def check(name, condition, detail=""):
    global _passed, _failed
    if condition: print(f"  ✓ {name}"); _passed += 1
    else: print(f"  ✗ {name}{' — '+detail if detail else ''}"); _failed += 1


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _aus(smile=0.4, au4=0.1, au6=0.3, au7=0.1, tension=0.1):
    return {"AU4":au4,"AU6":au6,"AU7":au7,"AU12":smile,"AU15":0.05,"AU17":0.0,
            "AU23":0.1,"AU24":0.05,"AU26":0.0,"AU43":0.0,
            "facial_tension":tension,"expression_intensity":0.3,"confidence":0.9}

def _eye_t(blink_rate=15.0, ear=0.30, ear_std=0.02):
    return {"blink_rate":blink_rate,"avg_EAR":ear,"blink_trend":0.0,
            "ear_mean":ear,"ear_std":ear_std,"confidence":0.9}

def _head_t(yaw=3.0, pitch=-5.0, stab=0.9, attn=0.85, dir="forward"):
    return {"yaw":yaw,"pitch":pitch,"roll":1.0,"head_stability":stab,
            "attention_score":attn,"attention_direction":dir,
            "nod_detected":False,"shake_detected":False,"confidence":0.9}

def _gaze_t(ec=0.8, ec_rate=0.75, vol=0.10, pat="fixating"):
    return {"eye_contact_score":ec,"eye_contact_rate":ec_rate,
            "gaze_volatility":vol,"gaze_pattern":pat,
            "fixation_active":True,"gaze_x":0.05,"gaze_y":0.0,"confidence":1.0}

def _mouth_t(smile=0.4, speaking=0.0, yawning=False, tension=0.1):
    return {"smile_intensity":smile,"speaking_activity":speaking,
            "yawning":yawning,"lip_tension":tension,"confidence":1.0}

def _brow_f(raise_=0.2, lower=0.1, tension=0.1, asym=0.05):
    return {"brow_raise":raise_,"brow_lower":lower,"brow_tension":tension,
            "brow_asymmetry":asym,"inner_brow_angle":0.0,"confidence":0.9}

def _face_f(cx=0.5, cy=0.5, disp=1.0, dist=55.0):
    return {"face_presence":True,"face_count":1,"face_distance_cm":dist,
            "face_center":[cx,cy],"face_bbox":[0.3,0.3,0.7,0.7],
            "face_bbox_size":0.16,"face_yaw_proxy":0.0,
            "frame_displacement":disp,"confidence":1.0}

def _body_f(lean="center", lean_int=0.0):
    return {"neck_angle":0.0,"leaning_direction":lean,"leaning_intensity":lean_int,
            "pose_model_active":False,"spine_angle":0.0}

def _hand_f():
    return {"hand_near_face":False,"hand_to_face_score":0.0,
            "self_touch_behavior":0.0,"crossed_arms_probability":0.0,
            "gesture_primitive":"none","hand_model_active":False,"pose_model_active":False}

def _face_t(restless=0.1, still=False, fidget=0.0):
    return {"face_stability":0.9,"movement_energy":1.0,"movement_variability":restless,
            "movement_trend":0.0,"prolonged_stillness":still,
            "stillness_duration_ms":0.0,"restlessness_score":restless,
            "fidget_probability":fidget}

def _aff():
    from features.affective_embedding import extract_affective_embedding
    return extract_affective_embedding(_aus(),{"avg_EAR":0.30,"blink_detected":False,"ear_confidence":1.0,"metric_space":"pixel"},
        {"MAR":0.12,"smile_intensity":0.4,"lip_tension":0.1,"jaw_drop":0.1,"lip_compression":0.05,"confidence":1.0},
        _brow_f(),_head_t())

def _gru():
    from temporal.affective_gru import AffectiveGRU
    gru=AffectiveGRU(); a=_aff(); ts=NOW_MS; o={}
    for _ in range(20): o=gru.update(a,ts); ts+=33
    return o

def _dyn(gru_out=None):
    from state.emotional_dynamics import EmotionalDynamicsTracker
    t=EmotionalDynamicsTracker(); a=_aff(); g=gru_out or _gru()
    from state.social_state import estimate_social_state
    from temporal.eye_temporal import BlinkTracker
    from temporal.head_pose_temporal import HeadPoseTracker
    from temporal.gaze_temporal import GazeTracker
    from temporal.mouth_temporal import MouthTracker
    b=BlinkTracker();ht=HeadPoseTracker();gt=GazeTracker();mt=MouthTracker()
    et=b.update(_eye_t(),NOW_MS); htt=ht.update({"head_yaw":3.0,"head_pitch":-5.0,"head_roll":1.0,"rotation_vector":[0,0,0],"reprojection_error":2.0,"confidence":0.9},NOW_MS)
    gtt=gt.update(_gaze_t(),NOW_MS); mtt=mt.update(_mouth_t(),NOW_MS)
    base=estimate_social_state(et,htt,gtt,mtt,_brow_f(),_aus(),_face_f())
    ts=NOW_MS; o={}
    for _ in range(20): o=t.update(a,g,base,ts); ts+=33
    return o


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_support_signals():
    print("\n[state/support_signals]")
    from state.support_signals import estimate_support_signals

    # Healthy / calm
    r = estimate_support_signals(
        _aus(smile=0.7,au4=0.0,au6=0.6), _eye_t(), _head_t(), _gaze_t(),
        _mouth_t(smile=0.7), _brow_f(tension=0.0), _face_f(), _aff(), _gru(),
        _body_f(), _hand_f(), _face_t(), _dyn()
    )
    expected_keys = ["distress_level","distress_type","genuine_smile",
                     "genuine_smile_score","emotional_suppression",
                     "overwhelm_probability","comfort_seeking",
                     "emotional_withdrawal","openness_to_support","support_urgency"]
    for k in expected_keys:
        check(f"has {k}", k in r)
    for k in ["distress_level","genuine_smile_score","emotional_suppression",
              "overwhelm_probability","comfort_seeking","emotional_withdrawal","openness_to_support"]:
        check(f"{k} [0,1]", 0.0 <= r[k] <= 1.0, f"got {r[k]}")
    check("urgency valid", r["support_urgency"] in {"immediate","soon","monitor","none"})
    check("distress_type valid", r["distress_type"] in {"overwhelm","sadness","anxiety","anger","none"})
    check("calm → low distress", r["distress_level"] < 0.4, f"got {r['distress_level']:.3f}")
    check("calm → urgency none/monitor", r["support_urgency"] in {"none","monitor"})

    # Duchenne smile
    check("genuine smile detected", r["genuine_smile"] or r["genuine_smile_score"] > 0,
          f"score={r['genuine_smile_score']:.3f}")

    # Stressed / suppressed
    r2 = estimate_support_signals(
        _aus(smile=0.4,au4=0.7,au6=0.05,au7=0.6,tension=0.8),
        _eye_t(blink_rate=28,ear=0.22,ear_std=0.05),
        _head_t(pitch=10,dir="down",stab=0.6),
        _gaze_t(ec=0.15,ec_rate=0.1,vol=0.7,pat="avoidant"),
        _mouth_t(smile=0.2,tension=0.7), _brow_f(tension=0.8,lower=0.7),
        _face_f(), _aff(), _gru(), _body_f(), _hand_f(),
        _face_t(restless=0.5), _dyn()
    )
    check("stressed → higher distress", r2["distress_level"] > r["distress_level"],
          f"{r2['distress_level']:.3f} vs {r['distress_level']:.3f}")
    check("stressed → lower openness", r2["openness_to_support"] < r["openness_to_support"],
          f"{r2['openness_to_support']:.3f} vs {r['openness_to_support']:.3f}")
    check("suppression + smile + tension", r2["emotional_suppression"] >= 0.0)

    # Overwhelm scenario (high arousal, negative valence, dysregulation)
    from temporal.affective_gru import AffectiveGRU
    from features.affective_embedding import extract_affective_embedding
    overwhelm_aus = _aus(smile=0.0,au4=0.5,au6=0.0,au7=0.8,tension=0.9)
    overwhelm_emb = extract_affective_embedding(
        overwhelm_aus,{"avg_EAR":0.26,"blink_detected":False,"ear_confidence":1.0,"metric_space":"pixel"},
        {"MAR":0.08,"smile_intensity":0.0,"lip_tension":0.8,"jaw_drop":0.0,"lip_compression":0.6,"confidence":1.0},
        _brow_f(tension=0.9,lower=0.8), _head_t(stab=0.5)
    )
    gru2 = AffectiveGRU()
    ts = NOW_MS
    for _ in range(30): g2 = gru2.update(overwhelm_emb, ts); ts+=33
    r3 = estimate_support_signals(
        overwhelm_aus, _eye_t(blink_rate=32,ear=0.21,ear_std=0.08),
        _head_t(stab=0.5), _gaze_t(ec=0.1,vol=0.8,pat="avoidant"),
        _mouth_t(smile=0.0,tension=0.9), _brow_f(tension=0.9),
        _face_f(), overwhelm_emb, g2, _body_f(), _hand_f(),
        _face_t(restless=0.8), _dyn()
    )
    check("overwhelm scenario → overwhelm > 0", r3["overwhelm_probability"] > 0.0,
          f"got {r3['overwhelm_probability']:.3f}")

    # Comfort seeking: self-touch
    r4 = estimate_support_signals(
        _aus(smile=0.1,au4=0.4), _eye_t(), _head_t(),
        _gaze_t(ec=0.2,ec_rate=0.15), _mouth_t(smile=0.1), _brow_f(tension=0.5),
        _face_f(), _aff(), _gru(),
        _body_f(lean="forward",lean_int=0.4),
        {"hand_near_face":True,"hand_to_face_score":0.7,"self_touch_behavior":0.7,
         "crossed_arms_probability":0.1,"gesture_primitive":"none",
         "hand_model_active":True,"pose_model_active":False},
        _face_t(fidget=0.6), _dyn()
    )
    check("hand-to-face + lean → comfort_seeking > 0", r4["comfort_seeking"] > 0.1,
          f"got {r4['comfort_seeking']:.3f}")

    # Withdrawal scenario
    r5 = estimate_support_signals(
        _aus(smile=0.0,au4=0.3), _eye_t(),
        _head_t(pitch=18,dir="down",stab=0.8),
        _gaze_t(ec=0.05,ec_rate=0.05,vol=0.15,pat="avoidant"),
        _mouth_t(smile=0.0,speaking=0.0), _brow_f(tension=0.3),
        _face_f(), _aff(), _gru(), _body_f(), _hand_f(),
        _face_t(still=True), _dyn()
    )
    check("avoidant + still + head_down → withdrawal > 0.3",
          r5["emotional_withdrawal"] > 0.3,
          f"got {r5['emotional_withdrawal']:.3f}")


def test_event_trigger():
    print("\n[outputs/event_trigger]")
    from outputs.event_trigger import EmissionController
    import time as t_mod

    ctrl = EmissionController(heartbeat_s=5.0, min_interval_s=0.0)

    state_a = {"distress_level":0.3,"support_urgency":"monitor",
                "engagement":{"value":0.6},"gru_valence":-0.1,
                "stress_escalating":False,"transition_occurred":False,
                "overwhelm_probability":0.1}

    # First emission always fires
    ok, reason = ctrl.should_emit(state_a)
    check("first emission fires",  ok)
    check("reason is first",       reason == "first_emission")
    ctrl.mark_emitted(state_a)

    # Stable state → no emission
    ok2, r2 = ctrl.should_emit(state_a)
    check("stable → no emit", not ok2, f"reason={r2}")

    # Urgency change → immediate
    state_b = {**state_a, "support_urgency":"immediate"}
    ok3, r3 = ctrl.should_emit(state_b)
    check("urgency change → emit", ok3, f"reason={r3}")
    check("reason mentions urgency", "urgency" in r3)
    ctrl.mark_emitted(state_b)

    # Significant distress change
    state_c = {**state_b,"distress_level":0.72,"support_urgency":"immediate"}
    ok4, r4 = ctrl.should_emit(state_c)
    check("large distress delta → emit", ok4, f"reason={r4}")
    ctrl.mark_emitted(state_c)

    # Stress escalation onset
    state_d = {**state_c,"stress_escalating":True,"distress_level":0.73}
    ok5, r5 = ctrl.should_emit(state_d)
    check("stress escalation → emit", ok5, f"reason={r5}")
    ctrl.mark_emitted(state_d)

    # Affect transition
    state_e = {**state_d,"transition_occurred":True,
               "transition_from":"HVLA","transition_to":"LVHA"}
    ok6, r6 = ctrl.should_emit(state_e)
    check("affect transition → emit", ok6, f"reason={r6}")
    ctrl.mark_emitted(state_e)

    # Changed dims list
    changed = ctrl.get_changed_dims({**state_e,"distress_level":0.95})
    check("get_changed_dims returns list", isinstance(changed, list))

    # Reset
    ctrl.reset()
    ok7, r7 = ctrl.should_emit(state_a)
    check("after reset → first emission", ok7 and r7=="first_emission")

    # Rate limiter: min_interval enforced
    ctrl2 = EmissionController(heartbeat_s=5.0, min_interval_s=10.0)
    ctrl2.should_emit(state_a)   # first
    ctrl2.mark_emitted(state_a)
    ok8, r8 = ctrl2.should_emit(state_b)
    check("min_interval blocks rapid emission", not ok8, f"reason={r8}")


def test_llm_context():
    print("\n[outputs/llm_context]")
    from outputs.llm_context import format_llm_context
    from state.support_signals import estimate_support_signals

    sup = estimate_support_signals(
        _aus(smile=0.05,au4=0.6,tension=0.7),
        _eye_t(blink_rate=24,ear=0.24),
        _head_t(pitch=8,dir="down",stab=0.7),
        _gaze_t(ec=0.2,ec_rate=0.15,vol=0.5,pat="avoidant"),
        _mouth_t(smile=0.05,tension=0.6), _brow_f(tension=0.7,lower=0.6),
        _face_f(), _aff(), _gru(), _body_f(), _hand_f(), _face_t(), _dyn()
    )
    from state.social_state import estimate_social_state
    from temporal.eye_temporal import BlinkTracker
    from temporal.head_pose_temporal import HeadPoseTracker
    from temporal.gaze_temporal import GazeTracker
    from temporal.mouth_temporal import MouthTracker
    b=BlinkTracker();ht=HeadPoseTracker();gt=GazeTracker();mt=MouthTracker()
    et=b.update(_eye_t(blink_rate=24,ear=0.24),NOW_MS)
    htt=ht.update({"head_yaw":-8.0,"head_pitch":8.0,"head_roll":2.0,"rotation_vector":[0,0,0],"reprojection_error":3.0,"confidence":0.8},NOW_MS)
    gtt=gt.update(_gaze_t(ec=0.2,ec_rate=0.15,pat="avoidant"),NOW_MS)
    mtt=mt.update(_mouth_t(smile=0.05),NOW_MS)
    base=estimate_social_state(et,htt,gtt,mtt,_brow_f(tension=0.7),_aus(smile=0.05,au4=0.6),_face_f())

    full = {
        **base, **_aff(), **_gru(), **_dyn(), **sup,
        "social_comfort":0.25,"rapport_signal":0.15,
        "conversational_readiness":0.35,"social_responsiveness":0.3,
        "global_uncertainty":0.25,"affective_ambiguity":0.4,
        "eye_contact_rate":0.15,"blink_rate":24.0,"fidget_probability":0.0,
        "face_present":True,
    }
    changed = [("distress_level",0.15),("openness_to_support",0.12)]

    # Compact mode
    ctx = format_llm_context(full, changed_dims=changed,
                              emit_reason="state_change:distress_level",
                              mode="compact")
    check("compact output non-empty",  len(ctx) > 50)
    check("compact is string",         isinstance(ctx, str))
    check("contains SUPPORT ASSESSMENT", "SUPPORT ASSESSMENT" in ctx)
    check("contains EMOTIONAL STATE",    "EMOTIONAL STATE"    in ctx)
    check("contains BEHAVIORAL",         "BEHAVIORAL"         in ctx)
    check("contains SOCIAL",             "SOCIAL"             in ctx)
    check("contains RECENT CHANGES",     "RECENT CHANGES"     in ctx)
    check("contains MULTIMODAL HANDOFF", "MULTIMODAL HANDOFF" in ctx)
    check("compact < 80 lines",          len(ctx.split("\n")) < 80,
          f"got {len(ctx.split(chr(10)))} lines")
    check("compact < 2000 chars",        len(ctx) < 2000, f"got {len(ctx)} chars")

    # Dict mode
    js = format_llm_context(full, mode="dict")
    obj = json.loads(js)
    check("dict has support section",      "support"  in obj)
    check("dict has affect section",       "affect"   in obj)
    check("dict has social section",       "social"   in obj)
    check("dict has dynamics section",     "dynamics" in obj)
    check("dict has multimodal_handoff",   "multimodal_handoff" in obj)
    check("dict has changed_dims",         "changed_dims" in obj)

    # Support section fields
    sup_sec = obj["support"]
    for k in ["urgency","distress_level","distress_type","openness",
              "genuine_smile","suppression","overwhelm","comfort_seeking","withdrawal"]:
        check(f"support.{k}", k in sup_sec)
    check("distress_level [0,1]", 0.0 <= sup_sec["distress_level"] <= 1.0)
    check("openness [0,1]",       0.0 <= sup_sec["openness"]       <= 1.0)

    # Multimodal handoff
    mh = obj["multimodal_handoff"]
    for k in ["visual_distress","visual_valence","visual_arousal","visual_openness",
              "visual_suppression","visual_withdrawal","visual_comfort_seeking",
              "visual_signal_conf"]:
        check(f"handoff.{k}", k in mh)
        check(f"handoff.{k} finite", -1.0 <= float(mh[k]) <= 1.0, f"got {mh[k]}")

    # No-face context
    ctx_nf = format_llm_context({"face_present":False}, mode="compact")
    check("no-face compact mentions no face", "No face" in ctx_nf)

    # Both mode
    ctx_both = format_llm_context(full, mode="both")
    check("both mode has compact section",    "VISUAL EMOTIONAL CONTEXT" in ctx_both)
    check("both mode has structured section", "STRUCTURED" in ctx_both)


def test_v3_end_to_end():
    print("\n[v3 end-to-end integration]")
    from state.support_signals import estimate_support_signals
    from outputs.event_trigger import EmissionController
    from outputs.llm_context import format_llm_context
    from features.affective_embedding import extract_affective_embedding
    from temporal.affective_gru import AffectiveGRU
    from state.uncertainty import UncertaintyTracker
    from state.emotional_dynamics import EmotionalDynamicsTracker
    from state.social_cognition import SocialCognitionTracker
    from state.social_state import estimate_social_state
    from state.derived_state import compute_derived_state
    from temporal.social_temporal import SocialTemporalTracker
    from temporal.face_temporal import FaceTemporalTracker
    from temporal.eye_temporal import BlinkTracker
    from temporal.head_pose_temporal import HeadPoseTracker
    from temporal.gaze_temporal import GazeTracker
    from temporal.mouth_temporal import MouthTracker

    gru=AffectiveGRU(); unc_t=UncertaintyTracker()
    dyn_t=EmotionalDynamicsTracker(); cog_t=SocialCognitionTracker()
    blink=BlinkTracker(); head_tr=HeadPoseTracker()
    gaze_tr=GazeTracker(); mouth_tr=MouthTracker()
    face_tr=FaceTemporalTracker(); soc_tr=SocialTemporalTracker()
    ctrl=EmissionController(heartbeat_s=5.0, min_interval_s=0.0)

    face_feat = _face_f()
    ts = NOW_MS
    emit_count = 0
    full_states = []

    # Simulate: calm for 30 frames, then stress rises
    for i in range(60):
        stress_val = 0.0 if i < 30 else (i - 30) / 30.0 * 0.8
        smile_val  = max(0.0, 0.5 - stress_val)

        aus_i   = _aus(smile=smile_val, au4=stress_val*0.7, tension=stress_val*0.8)
        eye_fi  = _eye_t(blink_rate=15.0 + stress_val*15, ear_std=stress_val*0.05)
        head_fi = {"head_yaw":3.0,"head_pitch":-5.0+stress_val*10,"head_roll":1.0,
                   "rotation_vector":[0,0,0],"reprojection_error":2.0,"confidence":0.9}
        gaze_fi = _gaze_t(ec=max(0.1,0.8-stress_val), ec_rate=max(0.1,0.75-stress_val),
                           vol=stress_val*0.6, pat="avoidant" if stress_val>0.5 else "fixating")
        mouth_fi= _mouth_t(smile=smile_val, tension=stress_val*0.7)
        brow_fi = _brow_f(tension=stress_val*0.8, lower=stress_val*0.7)

        eye_t   = blink.update(eye_fi,    ts)
        head_t  = head_tr.update(head_fi, ts)
        gaze_t  = gaze_tr.update(gaze_fi, ts)
        mouth_t = mouth_tr.update(mouth_fi, ts)
        face_t  = face_tr.update(face_feat, ts)

        aff     = extract_affective_embedding(aus_i, eye_fi, {"MAR":0.1,"smile_intensity":smile_val,"lip_tension":stress_val*0.7,"jaw_drop":0.0,"lip_compression":stress_val*0.4,"confidence":1.0}, brow_fi, head_t)
        gru_out = gru.update(aff, ts)

        base    = estimate_social_state(eye_t, head_t, gaze_t, mouth_t, brow_fi, aus_i, face_feat)
        soc_t   = soc_tr.update(base, gaze_t, ts)
        derived = compute_derived_state(base, soc_t, brow_fi, aus_i, gaze_t, head_t, face_t, eye_t)

        sensor_conf = base.get("meta",{}).get("overall_confidence",0.5)
        unc    = unc_t.update(base, aff, gru_out, sensor_conf)
        dyn    = dyn_t.update(aff, gru_out, base, ts)
        cog    = cog_t.update(base, head_t, gaze_t, mouth_t, eye_t, aff, _body_f(), aus_i, ts)
        sup    = estimate_support_signals(aus_i, eye_t, head_t, gaze_t, mouth_t, brow_fi,
                                          face_feat, aff, gru_out, _body_f(), _hand_f(), face_t, dyn)

        full = {**base, **derived, **aff, **gru_out, **dyn, **cog, **unc, **sup,
                "uncertainty": unc, "eye_contact_rate": gaze_t.get("eye_contact_rate",0.0),
                "blink_rate": eye_t.get("blink_rate",0.0), "fidget_probability":0.0}
        full_states.append(full)

        should, reason = ctrl.should_emit(full)
        if should:
            emit_count += 1
            changed = ctrl.get_changed_dims(full)
            ctx = format_llm_context(full, changed_dims=changed, emit_reason=reason, mode="compact")
            ctrl.mark_emitted(full)

        ts += 33

    check("pipeline ran 60 frames",     True)
    check("emitted < 60 times (event-driven)", emit_count < 60,
          f"emitted {emit_count} times")
    check("emitted at least once",      emit_count > 0)
    check("stressed state has higher distress",
          full_states[-1]["distress_level"] > full_states[0]["distress_level"],
          f"{full_states[-1]['distress_level']:.3f} vs {full_states[0]['distress_level']:.3f}")

    # Final state dict mode
    ctx_dict = format_llm_context(full_states[-1], mode="dict")
    obj = json.loads(ctx_dict)
    check("final dict has support urgency", "urgency" in obj.get("support",{}))
    check("final multimodal handoff valid",
          all(k in obj.get("multimodal_handoff",{}) for k in
              ["visual_distress","visual_valence","visual_signal_conf"]))

    # Context is LLM-ready size
    ctx_compact = format_llm_context(full_states[-1], mode="compact")
    check("context under 2000 chars",   len(ctx_compact) < 2000,
          f"got {len(ctx_compact)} chars")
    check("context has urgency line",   "SUPPORT ASSESSMENT" in ctx_compact)


if __name__ == "__main__":
    tests = [test_support_signals, test_event_trigger, test_llm_context, test_v3_end_to_end]
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
