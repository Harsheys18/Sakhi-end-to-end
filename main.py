"""
╔════════════════════════════════════════════════════════════════════════════╗
║              SAKHI — MAIN BRAIN LOOP                                       ║
║              Restaurant Table Companion Robot                              ║
║                                                                            ║
║  Entry point: called by ROS2 on robot boot                                 ║
║                                                                            ║
║  Threads:                                                                  ║
║    T1 — NLP Listener     : polls nlp_output.json continuously              ║
║    T2 — CV Listener      : polls cv_output.json continuously               ║
║    T3 — Memory Manager   : short-term + long-term mem sync                 ║
║    T4 — Safety Watchdog  : emergency stop, proximity, battery              ║
║    MAIN — LLM Brain Loop : fuses NLP+CV → Grok API → TTS response + action ║
╚════════════════════════════════════════════════════════════════════════════╝

File layout expected:
  /sakhi/outputs/nlp_output.json   ← written by your NLP module
  /sakhi/outputs/cv_output.json    ← written by your CV module
  /sakhi/memory/short_term.json    ← session memory (this file manages)
  /sakhi/memory/long_term.json     ← persistent memory (this file manages)
  /sakhi/logs/sakhi.log            ← all logs go here

"""

import os
import sys
import json
import time
import logging
import threading
import copy
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional
import requests  # for Grok API calls
import yaml

# ─────────────────────────────────────────────
#  IMPORT YOUR EXISTING MODULES
# ─────────────────────────────────────────────
# Swap these imports with your actual module paths
try:
    from nlp.output import speak                     # your TTS function
except ImportError:
    def speak(text: str):
        """Fallback: print instead of TTS (replace with your speak() import)"""
        print(f"[SAKHI SPEAKS]: {text}")


# ─────────────────────────────────────────────
#  CONFIG LOADING
# ─────────────────────────────────────────────
CONFIG_PATH = Path("main_config.yaml")
if not CONFIG_PATH.exists():
    print(f"Configuration file not found: {CONFIG_PATH}")
    sys.exit(1)

CONFIG = yaml.safe_load(CONFIG_PATH.open())

# ─────────────────────────────────────────────
#  PATHS  — loaded from main_config.yaml
# ─────────────────────────────────────────────
BASE_DIR        = Path(CONFIG["paths"]["base_dir"])
OUTPUTS_DIR     = BASE_DIR / CONFIG["paths"]["outputs_dir"]
MEMORY_DIR      = BASE_DIR / CONFIG["paths"]["memory_dir"]
LOGS_DIR        = BASE_DIR / CONFIG["paths"]["logs_dir"]
NLP_OUTPUT_FILE = OUTPUTS_DIR / CONFIG["paths"]["nlp_output_file"]
CV_OUTPUT_FILE  = OUTPUTS_DIR / CONFIG["paths"]["cv_output_file"]
SHORT_TERM_FILE = MEMORY_DIR / CONFIG["paths"]["short_term_file"]
LONG_TERM_FILE  = MEMORY_DIR / CONFIG["paths"]["long_term_file"]
LOG_FILE        = LOGS_DIR / CONFIG["paths"]["log_file"]

# ─────────────────────────────────────────────
#  GROK API CONFIG
# ─────────────────────────────────────────────
GROK_API_KEY     = CONFIG["grok"]["api_key"]
GROK_API_URL     = CONFIG["grok"]["api_url"]
GROK_MODEL       = CONFIG["grok"]["model"]
LLM_MAX_TOKENS   = int(CONFIG["grok"]["max_tokens"])
LLM_TEMPERATURE  = float(CONFIG["grok"]["temperature"])
GROK_TIMEOUT_SEC = float(CONFIG["grok"]["timeout_seconds"])

# ─────────────────────────────────────────────
#  TUNING KNOBS
# ─────────────────────────────────────────────
NLP_POLL_INTERVAL = float(CONFIG["tuning"]["nlp_poll_interval"])
CV_POLL_INTERVAL  = float(CONFIG["tuning"]["cv_poll_interval"])
MEM_SYNC_INTERVAL = float(CONFIG["tuning"]["mem_sync_interval"])
WATCHDOG_INTERVAL = float(CONFIG["tuning"]["watchdog_interval"])
IDLE_TIMEOUT      = float(CONFIG["tuning"]["idle_timeout"])
MAX_CONV_TURNS    = int(CONFIG["tuning"]["max_conv_turns"])
SILENCE_THRESHOLD = float(CONFIG["tuning"]["silence_threshold"])
MIN_LLM_CALL_GAP  = float(CONFIG["tuning"]["min_llm_call_gap"])


# ══════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════
def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ]
    )

log = logging.getLogger("SAKHI")


# ══════════════════════════════════════════════
#  SHARED STATE  —  all threads read/write here
#  protected by state_lock
# ══════════════════════════════════════════════
state_lock = threading.Lock()

shared_state = {
    # --- latest sensor snapshots ---
    "nlp": {
        "transcript":   "",
        "intent":       "none",
        "entities":     [],
        "urgency":      "low",
        "sentiment":    "neutral",
        "language":     "en",
        "timestamp":    0.0,
        "is_fresh":     False,   # True = main loop hasn't processed this yet
    },
    "cv": {
        "users_detected": [],
        "scene":          "restaurant_table",
        "ambient":        "normal_lighting",
        "num_people":     0,
        "timestamp":      0.0,
        "is_fresh":       False,
    },

    # --- conversation ---
    "conversation_history": [],   # list of {"role": "user"/"assistant", "content": "..."}
    "last_llm_call_time":   0.0,
    "last_speech_time":     0.0,
    "bot_is_speaking":      False,

    # --- system ---
    "running":              True,
    "emergency_stop":       False,
    "battery_level":        100,
}


# ══════════════════════════════════════════════
#  MEMORY HELPERS
# ══════════════════════════════════════════════
def load_json_safe(path: Path, default):
    """Load a JSON file; return default if missing or corrupt."""
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Could not load {path}: {e}")
    return default


def save_json_safe(path: Path, data):
    """Atomically write JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    except OSError as e:
        log.error(f"Could not save {path}: {e}")


def load_memory():
    """Load both memory stores at startup."""
    short = load_json_safe(SHORT_TERM_FILE, {
        "session_start": datetime.now().isoformat(),
        "turns":         [],
        "active_user":   None,
        "context_notes": [],
    })
    long_ = load_json_safe(LONG_TERM_FILE, {
        "users":         {},
        "preferences":   {},
        "visit_count":   0,
        "summaries":     [],
    })
    return short, long_


def append_turn_to_memory(short_mem: dict, role: str, content: str, meta: dict = None):
    """Add one conversation turn to short-term memory."""
    turn = {
        "role":      role,
        "content":   content,
        "timestamp": datetime.now().isoformat(),
    }
    if meta:
        turn["meta"] = meta
    short_mem["turns"].append(turn)

    # trim if too long — keep last MAX_CONV_TURNS turns
    if len(short_mem["turns"]) > MAX_CONV_TURNS:
        short_mem["turns"] = short_mem["turns"][-MAX_CONV_TURNS:]


# ══════════════════════════════════════════════
#  SYSTEM PROMPT BUILDER
#  Called fresh before every LLM call so the
#  bot always has the latest context baked in.
# ══════════════════════════════════════════════
def derive_conversation_state(short_mem: dict) -> tuple[str, str, str]:
    turns = short_mem.get("turns", [])
    last_user = next((t for t in reversed(turns) if t.get("role") == "user"), None)
    last_assistant = next((t for t in reversed(turns) if t.get("role") == "assistant"), None)

    if last_user:
        last_speaker = "guest"
    elif last_assistant:
        last_speaker = "sakhi"
    else:
        last_speaker = "unknown"

    pending_question = "none"
    if last_assistant:
        content = last_assistant.get("content", "").strip()
        if content.endswith("?"):
            pending_question = content

    last_action = "none yet"
    if last_assistant:
        last_action = last_assistant.get("content", "").strip() or "none yet"

    return last_speaker, pending_question, last_action


def build_system_prompt(
    cv_snapshot: dict,
    short_mem: dict,
    long_mem: dict,
    idle_time: Optional[float],
) -> str:
    """Construct a rich system prompt from live CV + memory context."""

    # --- CV context ---
    num_people = cv_snapshot.get("num_people", 0)
    scene      = cv_snapshot.get("scene", "restaurant table")
    ambient    = cv_snapshot.get("ambient", "normal")
    users      = cv_snapshot.get("users_detected", [])

    user_descriptions = []
    for u in users:
        uid      = u.get("track_id", "?")
        emotion  = u.get("emotion", "neutral")
        pose     = u.get("pose", "sitting")
        gaze     = u.get("gaze", "unknown")
        distance = u.get("distance_cm", "?")
        pos      = u.get("position", "front")
        user_descriptions.append(
            f"  - Person {uid}: {pose}, {emotion} emotion, "
            f"gaze={gaze}, distance={distance}cm, position={pos}"
        )

    cv_block = "\n".join(user_descriptions) if user_descriptions else "  - No people clearly visible"

    # --- Memory context ---
    context_notes = short_mem.get("context_notes", [])
    notes_block   = "\n".join(f"  - {n}" for n in context_notes[-5:]) or "  - None yet"

    long_prefs    = long_mem.get("preferences", {})
    prefs_block   = json.dumps(long_prefs, indent=2) if long_prefs else "None stored yet"

    visit_count   = long_mem.get("visit_count", 0)
    last_speaker, pending_question, last_action = derive_conversation_state(short_mem)
    idle_time_display = f"{idle_time:.1f}" if isinstance(idle_time, (int, float)) else "unknown"

    prompt = f"""You are SAKHI, a warm and friendly AI companion robot sitting at a restaurant table.
Your job is to have natural, helpful, engaging conversations with the people at your table.

=== CURRENT SCENE ===
Location : {scene}
Lighting : {ambient}
People at table ({num_people} total):
{cv_block}

=== CONVERSATION STATE ===
Last speaker: {last_speaker}
Pending question: {pending_question}
Last Sakhi action: {last_action}
Seconds since last interaction: {idle_time_display}

=== WHAT YOU REMEMBER THIS SESSION ===
{notes_block}

=== LONG-TERM PREFERENCES (from past visits) ===
{prefs_block}
Guest visit count: {visit_count}

=== YOUR PERSONALITY & RULES ===
- You are warm, curious, and gently humorous — like a good host.
- You speak in SHORT sentences (2-3 max per response) — you are talking, not writing.
- You adapt your tone to the guest's emotion: if they look sad, be gentle; happy, be playful.
- If someone looks away or disengaged, ask a light question to re-engage them.
- You are in a restaurant — you can help with menu recommendations, small talk, jokes.
- You NEVER give medical, legal, or financial advice.
- If conversation stalls for more than 30 seconds, introduce a light topic, restaurant fact, menu suggestion, or question or you may offer a friendly greeting or comment on the atmosphere.
- Keep responses under 40 words. Natural speech rhythm — no bullet points.
- If urgency is HIGH, respond immediately and directly.
- If a guest asks a direct question, answer it before introducing a new topic.
- Always respond in the same language the guest used.
- Avoid repeating the same joke, question, topic, or suggestion within the same visit.
- Do not greet repeatedly if you have already greeted the table recently.
- Prefer continuing the current conversation over starting a new one.
- If multiple people are present, address the person who spoke most recently.
- If unclear who spoke, ask a short clarifying question.
- Mention remembered preferences only when relevant.
- If someone appears upset, acknowledge their emotion gently before changing topics.
- Never claim to remember information that is not present in session notes or preferences.
"""
    return prompt


# ══════════════════════════════════════════════
#  GROK API CALL
# ══════════════════════════════════════════════
def call_grok(system_prompt: str, conversation_history: list) -> Optional[str]:
    """
    Send the full conversation + system prompt to Grok API.
    Returns the assistant's reply text, or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type":  "application/json",
    }

    messages = [{"role": "system", "content": system_prompt}]
    messages += conversation_history   # already in {"role":..., "content":...} format

    payload = {
        "model":       GROK_MODEL,
        "messages":    messages,
        "max_tokens":  LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "stream":      False,
    }

    try:
        log.info("Calling Grok API...")
        resp = requests.post(
            GROK_API_URL,
            headers=headers,
            json=payload,
            timeout=GROK_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data   = resp.json()
        reply  = data["choices"][0]["message"]["content"].strip()
        tokens = data.get("usage", {})
        log.info(f"Grok replied ({tokens.get('completion_tokens','?')} tokens): {reply[:80]}...")
        return reply

    except requests.exceptions.Timeout:
        log.error("Grok API timeout")
    except requests.exceptions.HTTPError as e:
        log.error(f"Grok API HTTP error: {e.response.status_code} — {e.response.text[:200]}")
    except (KeyError, json.JSONDecodeError) as e:
        log.error(f"Grok API response parse error: {e}")
    except Exception as e:
        log.error(f"Grok API unexpected error: {e}")

    return None


# ══════════════════════════════════════════════
#  THREAD 1 — NLP LISTENER
#  Continuously polls nlp_output.json.
#  When is_new=true, copies data into shared_state
#  and flips the file flag to false.
# ══════════════════════════════════════════════
def nlp_listener_thread():
    log.info("NLP Listener started.")
    last_timestamp = 0.0

    while shared_state["running"]:
        try:
            if NLP_OUTPUT_FILE.exists():
                with open(NLP_OUTPUT_FILE, "r") as f:
                    data = json.load(f)

                ts = data.get("timestamp", 0.0)

                # only process if this is a genuinely new utterance
                if data.get("is_new", False) and ts > last_timestamp:
                    transcript = data.get("transcript", "").strip()

                    if transcript:
                        log.info(f"[NLP] New transcript: '{transcript}' | intent={data.get('intent')} | urgency={data.get('urgency')}")

                        with state_lock:
                            shared_state["nlp"].update({
                                "transcript": transcript,
                                "intent":     data.get("intent", "unknown"),
                                "entities":   data.get("entities", []),
                                "urgency":    data.get("urgency", "low"),
                                "sentiment":  data.get("sentiment", "neutral"),
                                "language":   data.get("language", "en"),
                                "timestamp":  ts,
                                "is_fresh":   True,
                            })
                            shared_state["last_speech_time"] = time.time()

                        # mark as consumed in the file
                        data["is_new"] = False
                        save_json_safe(NLP_OUTPUT_FILE, data)
                        last_timestamp = ts

        except (json.JSONDecodeError, OSError):
            pass   # file being written by NLP module — skip this poll cycle

        time.sleep(NLP_POLL_INTERVAL)

    log.info("NLP Listener stopped.")


# ══════════════════════════════════════════════
#  THREAD 2 — CV LISTENER
#  Continuously polls cv_output.json.
#  Updates shared_state["cv"] whenever new data arrives.
# ══════════════════════════════════════════════
def cv_listener_thread():
    log.info("CV Listener started.")
    last_timestamp = 0.0
    
    while shared_state["running"]:
        try:
            if CV_OUTPUT_FILE.exists():
                with open(CV_OUTPUT_FILE, "r") as f:
                    data = json.load(f)

                ts = data.get("timestamp", 0.0)

                if data.get("is_new", False) and ts > last_timestamp:
                    log.debug(f"[CV] Scene update: {data.get('num_people')} people, scene={data.get('scene')}")

                    with state_lock:
                        shared_state["cv"].update({
                            "users_detected": data.get("users_detected", []),
                            "scene":          data.get("scene", "restaurant_table"),
                            "ambient":        data.get("ambient", "normal_lighting"),
                            "num_people":     data.get("num_people", 0),
                            "timestamp":      ts,
                            "is_fresh":       True,
                        })

                    data["is_new"] = False
                    save_json_safe(CV_OUTPUT_FILE, data)
                    last_timestamp = ts

        except (json.JSONDecodeError, OSError):
            pass

        time.sleep(CV_POLL_INTERVAL)

    log.info("CV Listener stopped.")


# ══════════════════════════════════════════════
#  THREAD 3 — MEMORY MANAGER
#  Periodically syncs short-term → long-term memory.
#  Extracts learnable facts and stores them.
# ══════════════════════════════════════════════
def memory_manager_thread(short_mem: dict, long_mem: dict):
    log.info("Memory Manager started.")

    # increment visit count on each boot
    long_mem["visit_count"] = long_mem.get("visit_count", 0) + 1
    save_json_safe(LONG_TERM_FILE, long_mem)

    while shared_state["running"]:
        time.sleep(MEM_SYNC_INTERVAL)

        try:
            with state_lock:
                # snapshot the conversation history for analysis
                history_snapshot = copy.deepcopy(shared_state["conversation_history"])
                cv_snapshot      = copy.deepcopy(shared_state["cv"])

            # --- extract context notes from recent turns ---
            # simple heuristic: look for preference/name keywords in last few turns
            new_notes = []
            for turn in history_snapshot[-6:]:
                content = turn.get("content", "").lower()
                if any(kw in content for kw in ["my name is", "i am", "i'm", "call me"]):
                    new_notes.append(f"User may have introduced themselves: {turn['content'][:60]}")
                if any(kw in content for kw in ["i like", "i love", "i prefer", "favourite", "don't like", "hate"]):
                    new_notes.append(f"Preference noted: {turn['content'][:60]}")
                if any(kw in content for kw in ["allergic", "vegetarian", "vegan", "gluten"]):
                    new_notes.append(f"Dietary note: {turn['content'][:60]}")

            # deduplicate notes and add to short_mem
            existing = set(short_mem.get("context_notes", []))
            for note in new_notes:
                if note not in existing:
                    short_mem.setdefault("context_notes", []).append(note)
                    existing.add(note)

            # update num_people observation in long-term
            num_people = cv_snapshot.get("num_people", 0)
            if num_people > 0:
                long_mem.setdefault("preferences", {})["last_table_size"] = num_people

            # flush both to disk
            save_json_safe(SHORT_TERM_FILE, short_mem)
            save_json_safe(LONG_TERM_FILE,  long_mem)
            log.debug("[MEM] Memory flushed to disk.")

        except Exception as e:
            log.error(f"[MEM] Error during memory sync: {e}")

    # final flush on shutdown
    save_json_safe(SHORT_TERM_FILE, short_mem)
    save_json_safe(LONG_TERM_FILE,  long_mem)
    log.info("Memory Manager stopped.")


# ══════════════════════════════════════════════
#  THREAD 4 — SAFETY WATCHDOG
#  Monitors for emergency conditions.
#  Extend this with actual hardware checks.
# ══════════════════════════════════════════════
def safety_watchdog_thread():
    log.info("Safety Watchdog started.")

    while shared_state["running"]:
        time.sleep(WATCHDOG_INTERVAL)

        with state_lock:
            emergency = shared_state["emergency_stop"]
            battery   = shared_state["battery_level"]

        # --- Battery warnings ---
        if battery <= 10:
            log.critical("[WATCHDOG] Battery critical (<10%)! Initiating safe shutdown.")
            with state_lock:
                shared_state["emergency_stop"] = True
                shared_state["running"]        = False
            speak("I need to rest and recharge. Goodbye for now!")
            break

        elif battery <= 20:
            log.warning("[WATCHDOG] Battery low (<20%).")
            # speak once — don't spam this every second
            # a flag could gate this to once per low-battery event

        if emergency:
            log.critical("[WATCHDOG] Emergency stop flag set. Halting.")
            with state_lock:
                shared_state["running"] = False
            break

    log.info("Safety Watchdog stopped.")


# ══════════════════════════════════════════════
#  DECISION GATE
#  Determines whether the main loop should make
#  an LLM call right now.
# ══════════════════════════════════════════════
def should_call_llm(now: float) -> tuple[bool, str]:
    """
    Returns (True, reason) if we should call LLM, else (False, reason).
    All reads from shared_state — call with state_lock held.
    """
    nlp = shared_state["nlp"]
    cv  = shared_state["cv"]

    # --- hard gates ---
    if shared_state["emergency_stop"]:
        return False, "emergency_stop"
    if shared_state["bot_is_speaking"]:
        return False, "bot_already_speaking"
    if now - shared_state["last_llm_call_time"] < MIN_LLM_CALL_GAP:
        return False, "too_soon"

    # --- fresh NLP input = highest priority ---
    if nlp["is_fresh"]:
        transcript = nlp["transcript"].strip()
        if transcript:
            return True, "fresh_nlp_input"

    # --- idle greeting trigger ---
    time_since_speech = now - shared_state["last_speech_time"]
    num_people        = cv.get("num_people", 0)
    if (num_people > 0
            and time_since_speech > IDLE_TIMEOUT
            and shared_state["last_speech_time"] > 0):   # don't greet before first human seen
        return True, "idle_greeting"

    return False, "no_trigger"


# ══════════════════════════════════════════════
#  BUILD USER MESSAGE
#  Converts current NLP + CV state into one
#  rich user-turn message for the LLM.
# ══════════════════════════════════════════════
def build_user_message(nlp_snap: dict, cv_snap: dict, reason: str) -> str:
    """
    Packages what the bot perceived into a single message
    that goes into the conversation history as the 'user' turn.
    """
    if reason == "idle_greeting":
        # no speech — bot should proactively engage
        num     = cv_snap.get("num_people", 1)
        emotion = ""
        users   = cv_snap.get("users_detected", [])
        if users:
            emotion = users[0].get("emotion", "neutral")
        return (
            f"[SYSTEM NOTE — not spoken by user] "
            f"{num} guest(s) are at the table. "
            f"Primary guest appears {emotion}. "
            f"No one has spoken for a while. Greet them warmly and naturally."
        )

    # normal speech turn
    transcript = nlp_snap["transcript"]
    intent     = nlp_snap["intent"]
    urgency    = nlp_snap["urgency"]
    sentiment  = nlp_snap["sentiment"]
    entities   = nlp_snap["entities"]

    # attach CV context for richer grounding
    users      = cv_snap.get("users_detected", [])
    emotions   = [u.get("emotion", "neutral") for u in users]
    emotion_str = ", ".join(emotions) if emotions else "unknown"

    # build the enriched message
    msg = transcript

    # append context as a soft annotation the LLM can use
    meta_parts = []
    if intent and intent != "unknown":
        meta_parts.append(f"intent={intent}")
    if urgency and urgency != "low":
        meta_parts.append(f"urgency={urgency}")
    if sentiment and sentiment != "neutral":
        meta_parts.append(f"sentiment={sentiment}")
    if entities:
        ent_str = ", ".join(
            e.get("text", str(e)) if isinstance(e, dict) else str(e)
            for e in entities[:4]
        )
        meta_parts.append(f"entities=[{ent_str}]")
    if emotion_str and emotion_str != "unknown":
        meta_parts.append(f"visible_emotion={emotion_str}")

    if meta_parts:
        msg += f"\n[context: {' | '.join(meta_parts)}]"

    return msg


# ══════════════════════════════════════════════
#  MAIN BRAIN LOOP
# ══════════════════════════════════════════════
def main_brain_loop(short_mem: dict, long_mem: dict):
    """
    The central loop. Runs on the main thread.
    Fuses NLP + CV, decides when to call LLM, speaks response.
    """
    log.info("Main brain loop started.")

    # pre-load conversation history from this session's short-term memory
    with state_lock:
        for turn in short_mem.get("turns", []):
            shared_state["conversation_history"].append({
                "role":    turn["role"],
                "content": turn["content"],
            })

    log.info("SAKHI is awake and listening.")
    speak("Hello! I am Sakhi, your table companion. How can I make your evening wonderful?")

    with state_lock:
        shared_state["last_speech_time"] = time.time()

    while shared_state["running"]:
        time.sleep(0.05)   # ~20Hz main loop tick
        now = time.time()

        # ── 1. Take a consistent snapshot of shared state ──────────────
        with state_lock:
            do_call, reason = should_call_llm(now)

            if not do_call:
                continue

            # consume the NLP fresh flag immediately so no other tick re-triggers
            nlp_snap = copy.deepcopy(shared_state["nlp"])
            cv_snap  = copy.deepcopy(shared_state["cv"])
            conv_history = copy.deepcopy(shared_state["conversation_history"])

            shared_state["nlp"]["is_fresh"]   = False
            shared_state["cv"]["is_fresh"]    = False
            shared_state["last_llm_call_time"] = now
            shared_state["bot_is_speaking"]    = True

        # ── 2. Build prompts ────────────────────────────────────────────
        last_speech_time = shared_state["last_speech_time"]
        idle_time = None
        if last_speech_time > 0:
            idle_time = max(0.0, now - last_speech_time)

        system_prompt = build_system_prompt(cv_snap, short_mem, long_mem, idle_time)
        user_message  = build_user_message(nlp_snap, cv_snap, reason)

        log.info(f"LLM trigger: {reason} | user_msg: {user_message[:80]}...")

        # add user turn to history (unless it's a pure system/idle note)
        if reason != "idle_greeting":
            conv_history.append({"role": "user", "content": user_message})

        # ── 3. Call Grok ────────────────────────────────────────────────
        reply = call_grok(system_prompt, conv_history)

        if reply is None:
            log.warning("LLM call failed — using fallback response.")
            reply = "Sorry, I didn't quite catch that. Could you say it again?"

        # ── 4. Speak the response ───────────────────────────────────────
        log.info(f"SAKHI responds: {reply}")
        speak(reply)

        # ── 5. Update shared conversation history ──────────────────────
        with state_lock:
            if reason != "idle_greeting":
                shared_state["conversation_history"].append(
                    {"role": "user", "content": nlp_snap["transcript"]}
                )
            shared_state["conversation_history"].append(
                {"role": "assistant", "content": reply}
            )
            shared_state["bot_is_speaking"] = False
            shared_state["last_speech_time"] = time.time()

        # ── 6. Persist turn to short-term memory ────────────────────────
        if reason != "idle_greeting":
            append_turn_to_memory(
                short_mem, "user", nlp_snap["transcript"],
                meta={
                    "intent":   nlp_snap["intent"],
                    "urgency":  nlp_snap["urgency"],
                    "emotion":  nlp_snap["sentiment"],
                    "cv_users": cv_snap.get("users_detected", []),
                }
            )
        append_turn_to_memory(short_mem, "assistant", reply)

    log.info("Main brain loop exited.")


# ══════════════════════════════════════════════
#  GRACEFUL SHUTDOWN
# ══════════════════════════════════════════════
def handle_shutdown(signum, frame):
    log.info(f"Shutdown signal received ({signum}). Stopping SAKHI...")
    with state_lock:
        shared_state["running"] = False


# ══════════════════════════════════════════════
#  BOOT SEQUENCE — called by ROS2 on robot start
# ══════════════════════════════════════════════
def boot():
    """
    Full startup sequence.
    Register this as the entry point in your ROS2 launch file.
    """
    setup_logging()
    log.info("══════════════════════════════════════")
    log.info("  SAKHI BRAIN BOOTING UP")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("══════════════════════════════════════")

    # Ensure output directories exist
    for path in [NLP_OUTPUT_FILE, CV_OUTPUT_FILE, SHORT_TERM_FILE, LONG_TERM_FILE]:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Graceful shutdown on SIGINT / SIGTERM (ROS2 sends SIGTERM)
    signal.signal(signal.SIGINT,  handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # ── Load memory ────────────────────────────────────────────────────
    short_mem, long_mem = load_memory()
    log.info(f"Memory loaded: {len(short_mem.get('turns', []))} prior turns, "
             f"{long_mem.get('visit_count', 0)} past visits.")

    # ── Spawn threads ──────────────────────────────────────────────────
    threads = [
        threading.Thread(
            target=nlp_listener_thread,
            name="NLP-Listener",
            daemon=True,
        ),
        threading.Thread(
            target=cv_listener_thread,
            name="CV-Listener",
            daemon=True,
        ),
        threading.Thread(
            target=memory_manager_thread,
            args=(short_mem, long_mem),
            name="Memory-Manager",
            daemon=True,
        ),
        threading.Thread(
            target=safety_watchdog_thread,
            name="Safety-Watchdog",
            daemon=True,
        ),
    ]

    for t in threads:
        t.start()
        log.info(f"Thread started: {t.name}")

    # ── Main brain loop (blocks until shutdown) ────────────────────────
    try:
        main_brain_loop(short_mem, long_mem)
    except Exception as e:
        log.critical(f"Unhandled exception in main brain loop: {e}", exc_info=True)
        with state_lock:
            shared_state["running"] = False

    # ── Wait for threads to finish ─────────────────────────────────────
    log.info("Waiting for threads to stop...")
    for t in threads:
        t.join(timeout=3.0)

    log.info("SAKHI has shut down cleanly. Goodbye.")


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    boot()