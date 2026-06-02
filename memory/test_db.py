"""
test_sakhi_db.py
================
Full test suite for sakhi_db.put_in() and get_from().

Run:
    python test_sakhi_db.py

Before running, set the DATABASE_URL environment variable:

  PowerShell (cloud):
    $env:DATABASE_URL = "postgresql://user:password@host/dbname?sslmode=require"

  PowerShell (local):
    $env:DATABASE_URL = "postgresql://sakhi:sakhi@localhost/sakhi"

  Linux/macOS:
    export DATABASE_URL="postgresql://user:password@host/dbname?sslmode=require"
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sakhi_db import init_db, put_in, get_from, connect

# ── connection ────────────────────────────────────────────────────────────────
SID = "test_ses_001"   # session id used throughout all tests

# ── result tracker ────────────────────────────────────────────────────────────
_pass = 0
_fail = 0


def check(label: str, condition: bool, *, got=None, expected=None) -> None:
    global _pass, _fail
    if condition:
        print(f"  ✅  {label}")
        _pass += 1
    else:
        print(f"  ❌  {label}")
        if expected is not None:
            print(f"       expected : {expected}")
            print(f"       got      : {got}")
        _fail += 1


# ── sample data ───────────────────────────────────────────────────────────────

FIVE_PERSONS = [
    {"id": "p1", "status": "detected", "label": "nearest robot",
     "name_if_learned": None, "avg_valence": 0.0, "avg_engagement": 0.0,
     "responds_to_humor": False, "humor_types_landed": [],
     "humor_types_flopped": [], "tends_to_initiate": False,
     "quiet_one": False, "low_data": True,
     "mentioned_topics": [], "notable_moments": [],
     "dominant_quadrant_history": []},
    {"id": "p2", "status": "detected", "label": "left of p1",
     "name_if_learned": None, "avg_valence": 0.0, "avg_engagement": 0.0,
     "responds_to_humor": False, "humor_types_landed": [],
     "humor_types_flopped": [], "tends_to_initiate": False,
     "quiet_one": False, "low_data": True,
     "mentioned_topics": [], "notable_moments": [],
     "dominant_quadrant_history": []},
    {"id": "p3", "status": "detected", "label": "center",
     "name_if_learned": None, "avg_valence": 0.0, "avg_engagement": 0.0,
     "responds_to_humor": False, "humor_types_landed": [],
     "humor_types_flopped": [], "tends_to_initiate": False,
     "quiet_one": False, "low_data": True,
     "mentioned_topics": [], "notable_moments": [],
     "dominant_quadrant_history": []},
    {"id": "p4", "status": "detected", "label": "right of p3",
     "name_if_learned": None, "avg_valence": 0.0, "avg_engagement": 0.0,
     "responds_to_humor": False, "humor_types_landed": [],
     "humor_types_flopped": [], "tends_to_initiate": False,
     "quiet_one": False, "low_data": True,
     "mentioned_topics": [], "notable_moments": [],
     "dominant_quadrant_history": []},
    {"id": "p5", "status": "detected", "label": "far right",
     "name_if_learned": None, "avg_valence": 0.0, "avg_engagement": 0.0,
     "responds_to_humor": False, "humor_types_landed": [],
     "humor_types_flopped": [], "tends_to_initiate": False,
     "quiet_one": True, "low_data": True,
     "mentioned_topics": [], "notable_moments": [],
     "dominant_quadrant_history": []},
]

LOG_ENTRIES = [
    {"seq": 1, "ts_elapsed_min": 1.0,  "type": "robot_utterance",
     "intent": "introduction", "target": "group", "response": "group_smile",
     "per_person_reaction": [
         {"id": "p1", "smile_score": 0.55, "laughed": False},
         {"id": "p5", "smile_score": 0.15, "laughed": False},
     ]},
    {"seq": 2, "ts_elapsed_min": 2.0,  "type": "human_utterance",
     "speaker_id": "p1", "directed_at_robot": True,
     "text": "Ha! A robot with jokes."},
    {"seq": 3, "ts_elapsed_min": 2.3,  "type": "robot_utterance",
     "intent": "joke", "humor_category": "self_deprecating",
     "target": "p1", "response": "strong_laugh",
     "per_person_reaction": [
         {"id": "p1", "smile_score": 0.85, "laughed": True},
         {"id": "p4", "smile_score": 0.88, "laughed": True},
     ]},
    {"seq": 4, "ts_elapsed_min": 7.0,  "type": "table_event",
     "event": "food_arrived"},
    {"seq": 5, "ts_elapsed_min": 9.5,  "type": "robot_utterance",
     "intent": "joke", "humor_category": "food",
     "target": "group", "response": "strong_laugh",
     "who_reacted_most": "p4"},
]


def make_snapshot(ts_ms: int, elapsed_min: float,
                  energy: float = 0.74, valence: float = 0.68,
                  laughter: bool = False, distressed: bool = False,
                  pause: bool = False, quadrant: str = "HVHA") -> dict:
    return {
        "schema": "current_state_v1",
        "ts_ms": ts_ms,
        "session_id": SID,
        "elapsed_min": elapsed_min,
        "vision": {
            "face_count": 5,
            "persons": [
                {"id": "p1", "face_visible": True, "valence": 0.72,
                 "arousal": 0.65, "engagement": 0.80,
                 "genuine_smile": True, "genuine_smile_score": 0.61,
                 "emotional_suppression": 0.0, "stress_hints": 0.09,
                 "eye_contact_with_robot": 0.55, "speaking": False,
                 "attention_direction": "forward",
                 "leaning_direction": "forward",
                 "dominant_quadrant": quadrant}
            ],
            "uncertainty": {"global": 0.12, "low_light": False, "faces_occluded": 0}
        },
        "audio": {
            "latest_utterance": {
                "speaker_id": "p4",
                "text": "I can't believe how good this pasta is",
                "directed_at_robot": False,
                "ts_elapsed_min": elapsed_min - 0.1
            },
            "group_audio": {
                "laughter_detected": laughter,
                "laughter_intensity": 0.8 if laughter else 0.0,
                "per_person_laughter": [
                    {"id": "p1", "laughing": laughter, "intensity": 0.8 if laughter else 0.0},
                    {"id": "p4", "laughing": laughter, "intensity": 0.8 if laughter else 0.0},
                ],
                "overlapping_speech": False,
                "silence": False,
                "conversation_pace": "normal"
            },
            "per_person_voice": [
                {"id": "p4", "speaking": True,
                 "arousal_from_voice": 0.71, "valence_from_voice": 0.80}
            ]
        },
        "fused": {
            "group_energy": energy,
            "group_valence": valence,
            "collective_laughter_score": 0.8 if laughter else 0.0,
            "dominant_quadrant": quadrant,
            "group_talking": True,
            "mid_serious_conversation": False,
            "comfortable_silence": False,
            "natural_pause_detected": pause,
            "attention_on_robot": 0.51,
            "robot_being_ignored": False,
            "anyone_distressed": distressed,
            "most_engaged_person": "p1",
            "most_expressive_person": "p4",
            "current_speaker": "p4"
        },
        "robot": {
            "last_spoke_elapsed_min": 0.57,
            "last_utterance": "I'll leave you to it. Food deserves full attention.",
            "last_response_quality": "group_smile",
            "cooldown_active": False,
            "cooldown_remaining_ms": 0,
            "forced_silence_ms": 0,
            "was_ignored_last_attempt": False,
            "open_question_pending": False,
            "open_question": None,
            "current_face_expression": "listening",
            "current_face_intensity": 0.6
        },
        "table_context": {
            "food_ordered": True,
            "food_arrived": True,
            "dessert_stage": False,
            "bill_requested": False,
            "special_occasion": "promotion",
            "occasion_person": "p1",
            "known_names": {"p1": "Raj", "p4": "Meera"}
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────────────────────────────────────
print("Connecting to PostgreSQL...")
conn = connect()   # reads DATABASE_URL from environment

print("Initialising tables...")
init_db(conn)

print(f"Cleaning test session '{SID}'...")
with conn.cursor() as cur:
    cur.execute("DELETE FROM interaction_log WHERE session_id = %s", (SID,))
    cur.execute("DELETE FROM current_state   WHERE session_id = %s", (SID,))
    cur.execute("DELETE FROM session_memory  WHERE session_id = %s", (SID,))
conn.commit()
print("Ready.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  T1  session_meta
# ═════════════════════════════════════════════════════════════════════════════
print("[T1] session_meta")

put_in(conn, "session_memory", "session_meta", {
    "schema": "session_memory_v1",
    "session_id": SID,
    "started_at_ms": 1748000000000,
    "elapsed_min": 0.0,
}, SID)

meta = get_from(conn, "session_memory", "session_meta", session_id=SID)
check("session_id stored",        meta["session_id"] == SID)
check("schema stored",            meta["schema"] == "session_memory_v1")
check("started_at_ms stored",     meta["started_at_ms"] == 1748000000000)
check("elapsed_min = 0.0",        meta["elapsed_min"] == 0.0)

# partial update
put_in(conn, "session_memory", "session_meta", {"elapsed_min": 9.5}, SID)
meta2 = get_from(conn, "session_memory", "session_meta", session_id=SID)
check("partial update elapsed",   meta2["elapsed_min"] == 9.5)
check("session_id preserved",     meta2["session_id"] == SID)


# ═════════════════════════════════════════════════════════════════════════════
#  T2  persons — bulk insert
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T2] persons — bulk insert")

put_in(conn, "session_memory", "persons", FIVE_PERSONS, SID)

all_p = get_from(conn, "session_memory", "persons", session_id=SID)
check("5 persons stored",         len(all_p) == 5)
p1 = get_from(conn, "session_memory", "persons", {"id": "p1"}, SID)
check("p1 label correct",         p1["label"] == "nearest robot")
p5 = get_from(conn, "session_memory", "persons", {"id": "p5"}, SID)
check("p5 quiet_one = True",      p5["quiet_one"] == True)


# ═════════════════════════════════════════════════════════════════════════════
#  T3  persons — upsert
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T3] persons — upsert")

put_in(conn, "session_memory", "persons", {
    "id": "p1",
    "status": "active",
    "low_data": False,
    "name_if_learned": "Raj",
    "responds_to_humor": True,
    "tends_to_initiate": True,
    "humor_types_landed": ["self_deprecating"],
    "avg_valence": 0.68,
    "notable_moments": ["Challenged robot at T=2min"],
}, SID)

p1 = get_from(conn, "session_memory", "persons", {"id": "p1"}, SID)
check("status → active",          p1["status"] == "active")
check("name_if_learned = Raj",    p1["name_if_learned"] == "Raj")
check("responds_to_humor True",   p1["responds_to_humor"] == True)
check("label preserved",          p1["label"] == "nearest robot")
check("still 5 persons",          len(get_from(conn, "session_memory", "persons", session_id=SID)) == 5)

# upsert p5
put_in(conn, "session_memory", "persons", {
    "id": "p5", "status": "active", "low_data": False,
    "notable_moments": ["Silent 10min", "Spoke at T=10:30"],
}, SID)
p5 = get_from(conn, "session_memory", "persons", {"id": "p5"}, SID)
check("p5 status → active",       p5["status"] == "active")
check("p5 quiet_one preserved",   p5["quiet_one"] == True)


# ═════════════════════════════════════════════════════════════════════════════
#  T4  persons — query filters
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T4] persons — query filters")

quiet = get_from(conn, "session_memory", "persons", {"quiet_one": True}, SID)
check("quiet_one filter → 1",     len(quiet) == 1 and quiet[0]["id"] == "p5")

active = get_from(conn, "session_memory", "persons", {"status": "active"}, SID)
check("status=active → 2",        len(active) == 2)

humor = get_from(conn, "session_memory", "persons", {"responds_to_humor": True}, SID)
check("responds_to_humor → p1",   len(humor) == 1 and humor[0]["id"] == "p1")

none_p = get_from(conn, "session_memory", "persons", {"id": "p99"}, SID)
check("unknown id → None",        none_p is None)


# ═════════════════════════════════════════════════════════════════════════════
#  T5  group_profile
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T5] group_profile")

put_in(conn, "session_memory", "group_profile", {
    "overall_receptiveness": 0.78,
    "group_type_guess": "friends",
    "humor_style_fit": ["self_deprecating", "observational", "food"],
    "avg_response_latency_ms": 1100,
}, SID)

gp = get_from(conn, "session_memory", "group_profile", session_id=SID)
check("group_type = friends",     gp["group_type_guess"] == "friends")
check("receptiveness stored",     gp["overall_receptiveness"] == 0.78)

put_in(conn, "session_memory", "group_profile", {"group_type_guess": "colleagues"}, SID)
gp2 = get_from(conn, "session_memory", "group_profile", session_id=SID)
check("full overwrite works",     gp2["group_type_guess"] == "colleagues")
check("old key gone",             "humor_style_fit" not in gp2)


# ═════════════════════════════════════════════════════════════════════════════
#  T6  joke_tracker
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T6] joke_tracker")

put_in(conn, "session_memory", "joke_tracker", {
    "total_attempts": 0, "strong_laugh": 0, "mild_smile": 0,
    "no_reaction": 0, "flopped": 0, "categories_used": [],
    "last_joke_elapsed_min": None, "best_responder": None,
}, SID)

for i, cat in enumerate(["self_deprecating", "food", "observational"], 1):
    jt = get_from(conn, "session_memory", "joke_tracker", session_id=SID)
    jt["total_attempts"] += 1
    jt["strong_laugh"]   += 1
    jt["categories_used"].append(cat)
    jt["last_joke_elapsed_min"] = i * 2.5
    jt["best_responder"] = "p4"
    put_in(conn, "session_memory", "joke_tracker", jt, SID)

jt = get_from(conn, "session_memory", "joke_tracker", session_id=SID)
check("3 attempts",               jt["total_attempts"] == 3)
check("3 strong laughs",          jt["strong_laugh"] == 3)
check("3 categories",             len(jt["categories_used"]) == 3)
check("best_responder = p4",      jt["best_responder"] == "p4")


# ═════════════════════════════════════════════════════════════════════════════
#  T7  topics_discussed
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T7] topics_discussed")

put_in(conn, "session_memory", "topics_discussed", {
    "topic": "promotion celebration",
    "first_raised_elapsed_min": 5.5,
    "raised_by": "robot_question",
    "group_response": "positive",
    "can_revisit": True,
    "details": "Raj promoted to senior engineer",
}, SID)

put_in(conn, "session_memory", "topics_discussed", {
    "topic": "food quality",
    "first_raised_elapsed_min": 9.4,
    "raised_by": "p4",
    "group_response": "excited",
    "can_revisit": True,
}, SID)

topics = get_from(conn, "session_memory", "topics_discussed", session_id=SID)
check("2 topics stored",          len(topics) == 2)

promo = get_from(conn, "session_memory", "topics_discussed", {"topic": "promotion celebration"}, SID)
check("lookup by topic name",     promo is not None and promo["group_response"] == "positive")

revisit = get_from(conn, "session_memory", "topics_discussed", {"can_revisit": True}, SID)
check("can_revisit filter → 2",   len(revisit) == 2)

missing = get_from(conn, "session_memory", "topics_discussed", {"topic": "unknown"}, SID)
check("missing topic → None",     missing is None)

# full replace
put_in(conn, "session_memory", "topics_discussed", [], SID)
check("replace with [] clears",   get_from(conn, "session_memory", "topics_discussed", session_id=SID) == [])


# ═════════════════════════════════════════════════════════════════════════════
#  T8  flags
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T8] flags")

put_in(conn, "session_memory", "flags", {
    "group_mid_serious_conversation": False,
    "food_arrived": False,
    "bill_requested": False,
    "quiet_person_needs_include": True,
    "forced_silence_ms_remaining": 0,
}, SID)

put_in(conn, "session_memory", "flags",
       {"food_arrived": True, "forced_silence_ms_remaining": 90000}, SID)

flags = get_from(conn, "session_memory", "flags", session_id=SID)
check("food_arrived → True",          flags["food_arrived"] == True)
check("forced_silence updated",        flags["forced_silence_ms_remaining"] == 90000)
check("bill_requested preserved",      flags["bill_requested"] == False)
check("quiet_person preserved",        flags["quiet_person_needs_include"] == True)
check("single key lookup",             get_from(conn, "session_memory", "flags", {"key": "food_arrived"}, SID) == True)
check("unknown key → None",            get_from(conn, "session_memory", "flags", {"key": "no_flag"}, SID) is None)

put_in(conn, "session_memory", "flags", {"bill_requested": True}, SID)
check("bill_requested → True",         get_from(conn, "session_memory", "flags", {"key": "bill_requested"}, SID) == True)


# ═════════════════════════════════════════════════════════════════════════════
#  T9  interaction_log — append and query
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T9] interaction_log")

for e in LOG_ENTRIES:
    put_in(conn, "session_memory", "interaction_log", e, SID)

all_log = get_from(conn, "session_memory", "interaction_log", session_id=SID)
check("all 5 entries stored",         len(all_log) == 5)

last2 = get_from(conn, "session_memory", "interaction_log", {"limit": 2}, SID)
check("limit=2 → 2 entries",          len(last2) == 2)
check("limit=2 last is seq 4 or 5",   last2[-1]["seq"] in (4, 5))

robot = get_from(conn, "session_memory", "interaction_log", {"type": "robot_utterance"}, SID)
check("type=robot_utterance → 3",     len(robot) == 3)

human = get_from(conn, "session_memory", "interaction_log", {"type": "human_utterance"}, SID)
check("type=human_utterance → 1",     len(human) == 1)

jokes = get_from(conn, "session_memory", "interaction_log", {"intent": "joke"}, SID)
check("intent=joke → 2",              len(jokes) == 2)

strong = get_from(conn, "session_memory", "interaction_log", {"response": "strong_laugh"}, SID)
check("response=strong_laugh → 2",    len(strong) == 2)

recent = get_from(conn, "session_memory", "interaction_log", {"since_min": 7.0}, SID)
check("since_min=7.0 → 2",            len(recent) == 2)

p1_log = get_from(conn, "session_memory", "interaction_log", {"person_id": "p1"}, SID)
check("person_id=p1 → 3",             len(p1_log) == 3)

p4_log = get_from(conn, "session_memory", "interaction_log", {"person_id": "p4"}, SID)
check("person_id=p4 → 2",             len(p4_log) == 2)

combined = get_from(conn, "session_memory", "interaction_log",
                    {"person_id": "p4", "type": "robot_utterance"}, SID)
check("p4 + robot_utterance → 2",     len(combined) == 2)

joke_strong = get_from(conn, "session_memory", "interaction_log",
                        {"intent": "joke", "response": "strong_laugh"}, SID)
check("joke + strong_laugh → 2",      len(joke_strong) == 2)


# ═════════════════════════════════════════════════════════════════════════════
#  T10  current_state — insert snapshots
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T10] current_state — insert snapshots")

BASE_TS = 1748000000000
snapshots = [
    make_snapshot(BASE_TS + i*5000, elapsed_min=i*0.083,
                  energy=0.5 + i*0.05, valence=0.6 + i*0.03,
                  laughter=(i == 3), distressed=(i == 5),
                  pause=(i == 2), quadrant="HVHA")
    for i in range(1, 9)   # 8 snapshots, ~40s of session
]

for snap in snapshots:
    put_in(conn, "current_state", "snapshot", snap, SID)

# latest snapshot
latest = get_from(conn, "current_state", "snapshot", session_id=SID)
check("latest snapshot returned",     latest is not None)
check("latest is most recent ts_ms",  latest["ts_ms"] == BASE_TS + 8*5000)
check("fused.group_energy correct",   abs(latest["fused"]["group_energy"] - (0.5 + 8*0.05)) < 0.001)


# ═════════════════════════════════════════════════════════════════════════════
#  T11  current_state — sub-section queries
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T11] current_state — sub-section queries")

fused = get_from(conn, "current_state", "fused", session_id=SID)
check("fused sub-section returned",   fused is not None and "group_energy" in fused)

vision = get_from(conn, "current_state", "vision", session_id=SID)
check("vision sub-section returned",  vision is not None and "face_count" in vision)
check("face_count = 5",               vision["face_count"] == 5)

robot = get_from(conn, "current_state", "robot", session_id=SID)
check("robot sub-section returned",   robot is not None and "current_face_expression" in robot)

audio = get_from(conn, "current_state", "audio", session_id=SID)
check("audio sub-section returned",   audio is not None and "latest_utterance" in audio)

table_ctx = get_from(conn, "current_state", "table_context", session_id=SID)
check("table_context returned",       table_ctx is not None and table_ctx["food_arrived"] == True)
check("known_names has Raj",          table_ctx["known_names"]["p1"] == "Raj")


# ═════════════════════════════════════════════════════════════════════════════
#  T12  current_state — time and filter queries
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T12] current_state — time and filter queries")

# all 8 snapshots
all_snaps = get_from(conn, "current_state", "snapshot", {"limit": 50}, SID)
check("all 8 snapshots stored",       len(all_snaps) == 8)

# last 3
last3 = get_from(conn, "current_state", "snapshot", {"limit": 3}, SID)
check("limit=3 → 3 snapshots",        len(last3) == 3)

# since_min
# snapshot i=5 has elapsed_min=5*0.083=0.415, i=6→0.498, i=7→0.581, i=8→0.664
since = get_from(conn, "current_state", "snapshot", {"since_min": 0.4}, SID)
check("since_min=0.4 → 4 snaps",      len(since) == 4)

# laughter — only i=3 has laughter=True
laugh_snaps = get_from(conn, "current_state", "snapshot", {"laughter": True}, SID)
check("laughter=True → 1 snapshot",   len(laugh_snaps) == 1)

# distressed — only i=5
dist_snaps = get_from(conn, "current_state", "snapshot", {"distressed": True}, SID)
check("distressed=True → 1 snapshot", len(dist_snaps) == 1)

# natural pause — only i=2
pause_snaps = get_from(conn, "current_state", "snapshot", {"pause": True}, SID)
check("pause=True → 1 snapshot",      len(pause_snaps) == 1)

# min_energy — energy = 0.5 + i*0.05  →  0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90
# min_energy=0.75 → i=5,6,7,8 → 4 snapshots
high_e = get_from(conn, "current_state", "snapshot", {"min_energy": 0.75}, SID)
check("min_energy=0.75 → 4 snaps",    len(high_e) == 4)

# since_ts_ms
since_ts = get_from(conn, "current_state", "snapshot",
                    {"since_ts_ms": BASE_TS + 6*5000}, SID)
check("since_ts_ms → 3 snaps",        len(since_ts) == 3)


# ═════════════════════════════════════════════════════════════════════════════
#  T13  error handling
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T13] error handling")

try:
    put_in(conn, "session_memory", "bad_section", {}, SID)
    check("bad section → KeyError",               False)
except KeyError:
    check("bad section → KeyError",               True)

try:
    put_in(conn, "bad_schema", "snapshot", {}, SID)
    check("bad schema → KeyError",                False)
except KeyError:
    check("bad schema → KeyError",                True)

try:
    put_in(conn, "session_memory", "interaction_log", ["not", "dict"], SID)
    check("log rejects non-dict",                 False)
except TypeError:
    check("log rejects non-dict",                 True)

try:
    put_in(conn, "current_state", "fused", {}, SID)
    check("current_state sub-section write → ValueError", False)
except ValueError:
    check("current_state sub-section write → ValueError", True)


# ═════════════════════════════════════════════════════════════════════════════
#  T14  missing session safe defaults
# ═════════════════════════════════════════════════════════════════════════════
print("\n[T14] safe defaults — missing session")

GHOST = "ses_ghost_never_created"
check("missing persons → []",        get_from(conn, "session_memory", "persons",          session_id=GHOST) == [])
check("missing flags → {}",          get_from(conn, "session_memory", "flags",            session_id=GHOST) == {})
check("missing joke_tracker → {}",   get_from(conn, "session_memory", "joke_tracker",     session_id=GHOST) == {})
check("missing group_profile → {}",  get_from(conn, "session_memory", "group_profile",    session_id=GHOST) == {})
check("missing topics → []",         get_from(conn, "session_memory", "topics_discussed", session_id=GHOST) == [])
check("missing interaction_log → []",get_from(conn, "session_memory", "interaction_log",  session_id=GHOST) == [])
check("missing current_state → None",get_from(conn, "current_state",  "snapshot",         session_id=GHOST) is None)


# ─────────────────────────────────────────────────────────────────────────────
conn.close()

print(f"\n{'=' * 50}")
print(f"  {_pass} passed    {_fail} failed")
print(f"{'=' * 50}")

if _fail:
    sys.exit(1)
