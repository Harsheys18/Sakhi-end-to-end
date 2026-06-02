"""
state/social_cognition.py

Rich social cognition modeling beyond basic engagement/attention.

Estimates socially meaningful signals that require integrating multiple
behavioral streams over time:

  social_responsiveness    : How readily the person reacts to interaction cues
                             (nods, gaze shifts, eye contact responses)

  interaction_reciprocity  : Degree of mutual behavioral synchrony
                             (mirroring, turn-taking readiness, social alignment)

  conversational_readiness : Whether the person is behaviorally ready to engage
                             in verbal/social interaction

  social_comfort           : Richer than discomfort — integrates proxemic,
                             postural, and gaze signals

  social_openness          : Openness to new interaction (vs withdrawal)

  rapport_signal           : Observable behavioral markers of rapport
                             (Duchenne smile, sustained eye contact, nod response)

  social_attention_quality : Not just looking at the robot, but engaged attending

Design: SocialCognitionTracker — stateful. No I/O.
"""

from collections import deque
import numpy as np
from geometry.math_utils import clamp, rolling_slope

_BUFFER_LEN = 90   # ~3 s


class SocialCognitionTracker:
    """
    Stateful social cognition tracker.

    Usage:
        tracker = SocialCognitionTracker()
        state   = tracker.update(
            base_state, head_t, gaze_t, mouth_t, eye_t,
            affective_emb, body_feats, aus, timestamp_ms
        )
    """

    def __init__(self):
        self._nod_buf:      deque[bool]  = deque(maxlen=_BUFFER_LEN)
        self._ec_buf:       deque[float] = deque(maxlen=_BUFFER_LEN)
        self._smile_buf:    deque[float] = deque(maxlen=_BUFFER_LEN)
        self._attn_buf:     deque[float] = deque(maxlen=_BUFFER_LEN)
        self._engage_buf:   deque[float] = deque(maxlen=_BUFFER_LEN)
        self._speak_buf:    deque[float] = deque(maxlen=_BUFFER_LEN)
        self._comfort_buf:  deque[float] = deque(maxlen=_BUFFER_LEN)

    def update(
        self,
        base_state:    dict,
        head_t:        dict,
        gaze_t:        dict,
        mouth_t:       dict,
        eye_t:         dict,
        affective_emb: dict,
        body_feats:    dict,
        aus:           dict,
        timestamp_ms:  float,
    ) -> dict:
        """
        Returns:
            {
              "social_responsiveness"    : float  # [0,1]
              "interaction_reciprocity"  : float  # [0,1]
              "conversational_readiness" : float  # [0,1]
              "social_comfort"           : float  # [0,1]
              "social_openness"          : float  # [0,1]
              "rapport_signal"           : float  # [0,1]
              "social_attention_quality" : float  # [0,1]
              "proxemic_comfort"         : float  # [0,1] comfort with current distance
              "behavioral_synchrony"     : float  # [0,1] alignment with interaction partner
            }
        """
        # Pull signals
        nod        = bool(head_t.get("nod_detected",     False))
        shake      = bool(head_t.get("shake_detected",   False))
        attn       = float(head_t.get("attention_score", 0.0))
        ec_rate    = float(gaze_t.get("eye_contact_rate",0.0))
        ec_score   = float(gaze_t.get("eye_contact_score",0.0))
        fix_active = bool(gaze_t.get("fixation_active",  False))
        gaze_vol   = float(gaze_t.get("gaze_volatility", 0.0))
        speaking   = float(mouth_t.get("speaking_activity", 0.0))
        smile      = float(mouth_t.get("smile_intensity",   0.0))
        yawning    = bool(mouth_t.get("yawning",            False))
        blink_rate = float(eye_t.get("blink_rate",         0.0))
        au12       = float(aus.get("AU12", 0.0))  # smile (Duchenne check)
        au6        = float(aus.get("AU6",  0.0))  # cheek raiser
        au4        = float(aus.get("AU4",  0.0))  # brow lower
        duchenne   = au6 * au12                   # genuine smile marker
        engage     = float(base_state.get("engagement",   {}).get("value", 0.0))
        discomfort = float(base_state.get("discomfort",   {}).get("value", 0.0))
        valence    = float(affective_emb.get("valence_geom", 0.0))
        lean_int   = float(body_feats.get("leaning_intensity", 0.0))
        lean_dir   = body_feats.get("leaning_direction", "center")
        dist_cm    = float(body_feats.get("neck_angle", 0.0))  # proxy

        # Buffer updates
        self._nod_buf.append(nod)
        self._ec_buf.append(ec_rate)
        self._smile_buf.append(smile)
        self._attn_buf.append(attn)
        self._engage_buf.append(engage)
        self._speak_buf.append(speaking)

        # ── Social responsiveness ─────────────────────────────────────────────
        # Does the person react to interaction cues?
        # Markers: nod rate, gaze shifts toward robot, smile responses
        nod_rate   = sum(self._nod_buf) / max(len(self._nod_buf), 1)
        attn_trend = _trend(self._attn_buf)

        responsiveness = clamp(
            0.30 * nod_rate * 5.0           # nod rate normalized
          + 0.25 * ec_rate
          + 0.20 * float(not yawning)
          + 0.15 * clamp(attn_trend * 50, 0.0, 1.0)
          + 0.10 * float(fix_active)
        , 0.0, 1.0)

        # ── Interaction reciprocity ───────────────────────────────────────────
        # Behavioral synchrony with the robot:
        # - Sustained eye contact + nods = high reciprocity
        # - Speaking turns + looking = reciprocal engagement
        reciprocity = clamp(
            0.35 * ec_rate
          + 0.25 * nod_rate * 5.0
          + 0.20 * speaking
          + 0.20 * (1.0 - gaze_vol)        # stable gaze = attending
        , 0.0, 1.0)

        # ── Conversational readiness ──────────────────────────────────────────
        # Behaviorally open and ready to engage verbally:
        # Not fatigued, facing the robot, open posture, not avoidant
        lean_forward = float(lean_dir == "forward") * 0.15
        conv_ready = clamp(
            0.25 * attn
          + 0.20 * ec_rate
          + 0.15 * (1.0 - discomfort)
          + 0.15 * float(not yawning)
          + 0.10 * (1.0 - gaze_vol)
          + 0.10 * (valence + 1.0) / 2.0   # positive valence helps
          + 0.05 * lean_forward
        , 0.0, 1.0)

        # ── Social comfort ────────────────────────────────────────────────────
        # Richer than inverse-discomfort:
        # Integrates proxemic (distance), postural (lean/open), and gaze signals
        proxemic_ok = clamp(1.0 - lean_int, 0.0, 1.0)  # leaning away = less comfortable
        social_comfort = clamp(
            0.30 * (1.0 - discomfort)
          + 0.20 * proxemic_ok
          + 0.20 * ec_rate
          + 0.15 * (valence + 1.0) / 2.0
          + 0.15 * float(not yawning)
        , 0.0, 1.0)

        self._comfort_buf.append(social_comfort)

        # ── Social openness ───────────────────────────────────────────────────
        # Willingness to be approached / start/continue interaction
        social_openness = clamp(
            0.30 * smile
          + 0.25 * (1.0 - discomfort)
          + 0.20 * attn
          + 0.15 * (1.0 - gaze_vol)
          + 0.10 * duchenne             # genuine smile = open
        , 0.0, 1.0)

        # ── Rapport signal ────────────────────────────────────────────────────
        # Observable markers that signal rapport with the interaction partner:
        # Duchenne smile + sustained eye contact + nodding = strong rapport
        rapport = clamp(
            0.35 * duchenne             # genuine smile (AU6 + AU12)
          + 0.30 * ec_rate
          + 0.20 * nod_rate * 5.0
          + 0.15 * (1.0 - au4)         # brow not furrowed
        , 0.0, 1.0)

        # ── Social attention quality ──────────────────────────────────────────
        # Not just looking but deeply attending (fixation + low volatility + engaged)
        attn_quality = clamp(
            0.35 * attn
          + 0.25 * float(fix_active)
          + 0.25 * (1.0 - gaze_vol)
          + 0.15 * engage
        , 0.0, 1.0)

        # ── Proxemic comfort ──────────────────────────────────────────────────
        # Comfort with the current interaction distance
        # Leaning away or turned away = less comfortable with proximity
        proxemic_comfort = clamp(
            0.50 * proxemic_ok
          + 0.30 * (1.0 - discomfort)
          + 0.20 * social_comfort
        , 0.0, 1.0)

        # ── Behavioral synchrony ──────────────────────────────────────────────
        # How aligned is the person's behavior with the interaction context?
        behavioral_sync = clamp(
            0.40 * reciprocity
          + 0.30 * responsiveness
          + 0.30 * rapport
        , 0.0, 1.0)

        return {
            "social_responsiveness"    : round(responsiveness,   3),
            "interaction_reciprocity"  : round(reciprocity,      3),
            "conversational_readiness" : round(conv_ready,       3),
            "social_comfort"           : round(social_comfort,   3),
            "social_openness"          : round(social_openness,  3),
            "rapport_signal"           : round(rapport,          3),
            "social_attention_quality" : round(attn_quality,     3),
            "proxemic_comfort"         : round(proxemic_comfort, 3),
            "behavioral_synchrony"     : round(behavioral_sync,  3),
        }

    def reset(self) -> None:
        for buf in [self._nod_buf, self._ec_buf, self._smile_buf,
                    self._attn_buf, self._engage_buf, self._speak_buf,
                    self._comfort_buf]:
            buf.clear()


def _trend(buf: deque) -> float:
    if len(buf) < 8:
        return 0.0
    arr = np.array(buf, dtype=np.float32)
    return float(rolling_slope(arr))
