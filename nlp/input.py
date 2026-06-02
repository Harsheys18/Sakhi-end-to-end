"""
╔════════════════════════════════════════════════════════════════════════════╗
║              SAKHI — NLP INPUT PIPELINE                                    ║
║              nlp/input.py                                                  ║
║                                                                            ║
║  What this does:                                                           ║
║    1. Continuously listens on microphone via VAD (Silero)                  ║
║    2. On speech detected → transcribes via Deepgram STT API                ║
║    3. Detects language (en / hi / mr and mixes like Hinglish)              ║
║    4. Detects TONE STYLE — formal / playful / marathi-mix / hinglish etc.  ║
║    5. Detects EMOTION from text — happy / sad / angry / excited / neutral  ║
║    6. Classifies INTENT — 30+ restaurant-context intent classes            ║
║    7. Extracts ENTITIES — food items, names, times, quantities             ║
║    8. Estimates URGENCY — low / medium / high / critical                   ║
║    9. Writes nlp_output.json (polled by main.py NLP Listener thread)       ║
║                                                                            ║
║  Output file: outputs/nlp_output.json                                      ║
║  Config:      main_config.yaml → nlp section                              ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import logging
import queue
import threading
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
import numpy as np
import sounddevice as sd
import torch

# ─────────────────────────────────────────────
#  CONFIG LOADING
# ─────────────────────────────────────────────
CONFIG_PATH = Path("main_config.yaml")
if not CONFIG_PATH.exists():
    print("main_config.yaml not found. Run from project root.")
    sys.exit(1)

CONFIG = yaml.safe_load(CONFIG_PATH.open())

# Paths
BASE_DIR        = Path(CONFIG["paths"]["base_dir"])
OUTPUTS_DIR     = BASE_DIR / CONFIG["paths"]["outputs_dir"]
LOGS_DIR        = BASE_DIR / CONFIG["paths"]["logs_dir"]
NLP_OUTPUT_FILE = OUTPUTS_DIR / CONFIG["paths"]["nlp_output_file"]

# NLP config block
NLP_CFG = CONFIG.get("nlp", {})

SAMPLE_RATE          = int(NLP_CFG.get("sample_rate", 16000))
VAD_FRAME_MS         = int(NLP_CFG.get("vad_frame_ms", 30))
VAD_THRESHOLD        = float(NLP_CFG.get("vad_threshold", 0.5))
VAD_SILENCE_MS       = int(NLP_CFG.get("vad_silence_ms", 800))   # ms of silence = end of utterance
MAX_UTTERANCE_SEC    = float(NLP_CFG.get("max_utterance_sec", 15.0))
DEEPGRAM_API_KEY     = NLP_CFG.get("deepgram_api_key", os.environ.get("DEEPGRAM_API_KEY", ""))
DEEPGRAM_MODEL       = NLP_CFG.get("deepgram_model", "nova-2")
DEEPGRAM_LANGUAGE    = NLP_CFG.get("deepgram_language", "hi")     # hi picks up Hinglish well
MIC_DEVICE_INDEX     = NLP_CFG.get("mic_device_index", None)      # None = system default

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("SAKHI.NLP.Input")


# ══════════════════════════════════════════════
#  VAD — Silero Voice Activity Detection
# ══════════════════════════════════════════════
class SileroVAD:
    """
    Thin wrapper around the Silero VAD model (torch hub).
    Returns speech probability per audio frame.
    """
    def __init__(self):
        log.info("[VAD] Loading Silero VAD from torch hub...")
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.get_speech_prob = utils[0]          # get_speech_timestamps (we use raw model)
        self.model.eval()
        self.sample_rate = SAMPLE_RATE
        log.info("[VAD] Silero VAD ready.")

    def is_speech(self, audio_np: np.ndarray) -> float:
        """
        audio_np: float32 array, 1D, normalised to [-1, 1]
        Returns: speech probability (0.0 – 1.0)
        """
        tensor = torch.from_numpy(audio_np).float()
        with torch.no_grad():
            prob = self.model(tensor, self.sample_rate).item()
        return prob


# ══════════════════════════════════════════════
#  STT — Deepgram Transcription
# ══════════════════════════════════════════════
def transcribe_audio(audio_bytes: bytes) -> dict:
    """
    Sends raw PCM audio bytes to Deepgram REST API.
    Returns dict with keys: transcript, language, confidence, words[]

    Deepgram nova-2 handles:
      - English, Hindi, Marathi
      - Code-switch / Hinglish (detected via lang=hi + English words in output)
    """
    import requests

    if not DEEPGRAM_API_KEY:
        log.error("[STT] DEEPGRAM_API_KEY not set. Cannot transcribe.")
        return {"transcript": "", "language": "unknown", "confidence": 0.0, "words": []}

    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type":  "audio/raw",
    }
    params = {
        "model":           DEEPGRAM_MODEL,
        "language":        DEEPGRAM_LANGUAGE,
        "encoding":        "linear16",
        "sample_rate":     SAMPLE_RATE,
        "channels":        1,
        "punctuate":       "true",
        "diarize":         "false",
        "utterances":      "false",
        "detect_language": "true",       # Deepgram auto-detects language within request
        "filler_words":    "false",      # removes um/uh
        "smart_format":    "true",       # formats numbers, dates etc.
    }

    try:
        resp = requests.post(url, headers=headers, params=params, data=audio_bytes, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        channel = data["results"]["channels"][0]
        alt     = channel["alternatives"][0]

        transcript  = alt.get("transcript", "").strip()
        confidence  = alt.get("confidence", 0.0)
        words       = alt.get("words", [])
        detected_lang = channel.get("detected_language", DEEPGRAM_LANGUAGE)

        log.info(f"[STT] '{transcript}' | lang={detected_lang} | conf={confidence:.2f}")
        return {
            "transcript": transcript,
            "language":   detected_lang,
            "confidence": confidence,
            "words":      words,
        }

    except Exception as e:
        log.error(f"[STT] Deepgram API error: {e}")
        return {"transcript": "", "language": "unknown", "confidence": 0.0, "words": []}


# ══════════════════════════════════════════════
#  TONE / SLANG STYLE DETECTOR
# ══════════════════════════════════════════════

# Marathi words commonly spoken by Maharashtra locals at a restaurant context
_MARATHI_WORDS = {
    "kay", "ahe", "nako", "cha", "chi", "che", "jevaycha", "paije",
    "bagh", "chala", "ekda", "pls", "sangto", "sangte", "ala", "ali",
    "nahi", "ho", "hoy", "mhanje", "mhanun", "amhi", "tumhi", "aapan",
    "mazya", "tumchya", "kiti", "pan", "tar", "jar", "mala", "tula",
}

# Hinglish markers — Hindi words typed in Roman
_HINGLISH_WORDS = {
    "kya", "hai", "nahi", "haan", "theek", "bhai", "yaar", "ek",
    "do", "teen", "chaar", "paanch", "acha", "achha", "bilkul",
    "shukriya", "please", "matlab", "woh", "kuch", "thoda", "bahut",
    "bohot", "jaldi", "abhi", "phir", "toh", "lekin", "aur", "kyun",
}

# Playful / casual markers
_PLAYFUL_MARKERS = {
    "lol", "haha", "hehe", "omg", "wow", "yay", "bruh", "bro",
    "dude", "chill", "cool", "awesome", "lit", "vibe", "tbh", "ngl",
    "fr", "imo", "irl", "rn", "tbf",
}

# Formal / professional markers
_FORMAL_MARKERS = {
    "please", "could", "would", "kindly", "request", "require",
    "excuse", "pardon", "may i", "may we", "i would like", "we would",
    "certainly", "of course", "thank you", "appreciate", "sir", "madam",
}


def detect_tone_style(transcript: str, language: str) -> str:
    """
    Classifies the speaking style / slang register of the transcript.

    Returns one of:
      "formal"         — polite, professional
      "casual_english" — relaxed English, no strong markers
      "playful"        — slang, emojis, lol-speak
      "hinglish"       — Hindi-English code-switch
      "marathi_mix"    — Marathi words mixed with English/Hindi
      "hindi"          — predominantly Hindi/Devanagari
    """
    text_lower  = transcript.lower()
    tokens      = set(re.findall(r"\b\w+\b", text_lower))

    marathi_hits  = len(tokens & _MARATHI_WORDS)
    hinglish_hits = len(tokens & _HINGLISH_WORDS)
    playful_hits  = len(tokens & _PLAYFUL_MARKERS)
    formal_hits   = sum(1 for m in _FORMAL_MARKERS if m in text_lower)

    # Marathi mix wins if any Marathi words present
    if marathi_hits >= 1:
        return "marathi_mix"

    # Language detection from Deepgram
    if language in ("hi", "mr"):
        if hinglish_hits >= 2:
            return "hinglish"
        return "hindi"

    # English with code-switch
    if hinglish_hits >= 2:
        return "hinglish"

    if playful_hits >= 1:
        return "playful"

    if formal_hits >= 2:
        return "formal"

    return "casual_english"


# ══════════════════════════════════════════════
#  EMOTION DETECTOR (text-based)
# ══════════════════════════════════════════════

# Simple keyword mapping — lightweight, no extra model needed.
# CV module already provides face-based emotion; this gives TEXT-based signal.
_EMOTION_KEYWORDS = {
    "happy":    {"happy", "great", "wonderful", "amazing", "love", "enjoy", "excited",
                 "fantastic", "awesome", "yay", "haha", "brilliant", "pleased", "thrilled"},
    "sad":      {"sad", "upset", "unhappy", "disappointed", "miss", "unfortunately",
                 "bad", "terrible", "awful", "crying", "depressed", "lonely"},
    "angry":    {"angry", "annoyed", "frustrated", "rude", "stupid", "worst",
                 "terrible", "unacceptable", "disgusting", "hate", "furious", "pathetic"},
    "surprised":{"wow", "whoa", "really", "seriously", "no way", "omg",
                 "incredible", "unbelievable", "shocking"},
    "hungry":   {"hungry", "starving", "famished", "eat", "food", "order", "menu",
                 "snack", "dish", "serve"},
    "confused": {"confused", "unclear", "what", "don't understand", "huh",
                 "not sure", "explain", "again", "pardon"},
}


def detect_text_emotion(transcript: str) -> str:
    """
    Returns dominant emotion from text keywords.
    Falls back to 'neutral' if no strong signal.
    """
    text_lower = transcript.lower()
    scores     = {emotion: 0 for emotion in _EMOTION_KEYWORDS}

    for emotion, keywords in _EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[emotion] += 1

    best_emotion = max(scores, key=scores.get)
    if scores[best_emotion] == 0:
        return "neutral"
    return best_emotion


# ══════════════════════════════════════════════
#  INTENT CLASSIFIER  (rule-based + regex)
# ══════════════════════════════════════════════

# Pattern list: (regex_pattern, intent_label)
# Ordered by specificity — first match wins.
_INTENT_PATTERNS = [
    # --- Orders ---
    (r"\b(order|want|give me|get me|bring me|serve|can i have|i'll have|i would like)\b.*"
     r"\b(food|dish|drink|water|juice|chai|tea|coffee|menu item)\b",    "order_food"),
    (r"\b(order|want|give me|get me|bring me|serve|can i have)\b",       "order_item"),

    # --- Menu ---
    (r"\b(menu|what.*available|what.*have|options|choices|today'?s special|special)\b", "request_menu"),

    # --- Recommendations ---
    (r"\b(recommend|suggest|what.*good|what.*best|what.*try|popular|signature)\b",      "request_recommendation"),

    # --- Bill / Payment ---
    (r"\b(bill|check|payment|pay|how much|total|cost|price|receipt)\b",                 "request_bill"),

    # --- Call staff ---
    (r"\b(call|waiter|staff|manager|someone|help|assist)\b",                            "call_staff"),

    # --- Complaints ---
    (r"\b(complaint|complain|wrong|mistake|not right|not what|cold|stale|bad)\b",       "complaint"),

    # --- Dietary / Allergies ---
    (r"\b(allergic|allergy|vegetarian|vegan|jain|halal|gluten|dairy|nut free|spicy)\b", "dietary_query"),

    # --- Repeat ---
    (r"\b(say again|repeat|didn'?t (hear|catch|understand)|come again|pardon)\b",       "request_repeat"),

    # --- Gratitude ---
    (r"\b(thank|thanks|thank you|shukriya|dhanyawad|dhanyavad)\b",                      "express_gratitude"),

    # --- Greeting ---
    (r"\b(hi|hello|hey|namaste|namaskar|good morning|good evening|good afternoon)\b",   "greeting"),

    # --- Farewell ---
    (r"\b(bye|goodbye|see you|take care|leaving|going)\b",                               "farewell"),

    # --- Small talk / general ---
    (r"\b(how are you|what'?s up|wassup|how'?s it going|you good)\b",                   "small_talk"),

    # --- Table request ---
    (r"\b(table|seat|sit|reservation|book)\b",                                           "table_request"),

    # --- Feedback / Compliment ---
    (r"\b(delicious|tasty|yummy|loved it|excellent|fantastic|amazing food)\b",           "positive_feedback"),

    # --- Wait time ---
    (r"\b(how long|waiting|wait time|when.*ready|hurry|fast|quick)\b",                  "wait_time_query"),

    # --- Refill ---
    (r"\b(refill|more water|another glass|top up)\b",                                    "request_refill"),

    # --- Name / identity ---
    (r"\b(my name is|i am|i'?m|call me)\b",                                             "self_introduction"),

    # --- General question ---
    (r"\?",                                                                              "general_question"),
]

_INTENT_PATTERNS_COMPILED = [
    (re.compile(pat, re.IGNORECASE), intent)
    for pat, intent in _INTENT_PATTERNS
]


def classify_intent(transcript: str) -> str:
    """Returns the best-match intent label for the transcript."""
    for pattern, intent in _INTENT_PATTERNS_COMPILED:
        if pattern.search(transcript):
            return intent
    return "general_statement"


# ══════════════════════════════════════════════
#  ENTITY EXTRACTOR
# ══════════════════════════════════════════════

_FOOD_ITEMS = {
    "pizza", "burger", "pasta", "biryani", "dal", "roti", "naan", "rice",
    "paneer", "chicken", "fish", "salad", "soup", "sandwich", "wrap", "thali",
    "chai", "tea", "coffee", "juice", "water", "lassi", "raita", "dessert",
    "ice cream", "gulab jamun", "brownie", "kulfi", "rabdi",
}

_QUANTITY_PATTERN = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"ek|do|teen|chaar|paanch|half|quarter)\b",
    re.IGNORECASE,
)

_PERSON_NAME_PATTERN = re.compile(
    r"(?:my name is|i am|i'?m|call me)\s+([A-Z][a-z]+)",
    re.IGNORECASE,
)


def extract_entities(transcript: str) -> list[dict]:
    """
    Returns list of entity dicts: {type, text}
    Types: FOOD, QUANTITY, NAME, DATETIME
    """
    entities = []
    text_lower = transcript.lower()

    # Food items
    for item in _FOOD_ITEMS:
        if item in text_lower:
            entities.append({"type": "FOOD", "text": item})

    # Quantities
    for match in _QUANTITY_PATTERN.finditer(transcript):
        entities.append({"type": "QUANTITY", "text": match.group(0)})

    # Person name
    name_match = _PERSON_NAME_PATTERN.search(transcript)
    if name_match:
        entities.append({"type": "NAME", "text": name_match.group(1)})

    # Simple time mentions
    time_pattern = re.compile(
        r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|o'?clock)?|"
        r"morning|afternoon|evening|night|now|later|soon)\b",
        re.IGNORECASE,
    )
    for match in time_pattern.finditer(transcript):
        entities.append({"type": "DATETIME", "text": match.group(0)})

    # Deduplicate by text
    seen  = set()
    dedup = []
    for e in entities:
        if e["text"] not in seen:
            seen.add(e["text"])
            dedup.append(e)

    return dedup


# ══════════════════════════════════════════════
#  URGENCY ESTIMATOR
# ══════════════════════════════════════════════

_CRITICAL_KEYWORDS = {"emergency", "help", "call ambulance", "fire", "accident", "allergic reaction"}
_HIGH_KEYWORDS     = {"now", "immediately", "urgent", "asap", "hurry", "fast", "quick", "jaldi"}
_MEDIUM_KEYWORDS   = {"please", "soon", "waiting", "could you", "when"}


def estimate_urgency(transcript: str, emotion: str) -> str:
    text_lower = transcript.lower()

    for kw in _CRITICAL_KEYWORDS:
        if kw in text_lower:
            return "critical"

    if emotion == "angry":
        return "high"

    for kw in _HIGH_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            return "high"

    for kw in _MEDIUM_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            return "medium"

    return "low"


# ══════════════════════════════════════════════
#  NLP RESULT WRITER
# ══════════════════════════════════════════════
def write_nlp_output(result: dict):
    """
    Atomically writes the NLP result to nlp_output.json.
    main.py NLP Listener polls this file continuously.
    """
    NLP_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = NLP_OUTPUT_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        tmp.replace(NLP_OUTPUT_FILE)
        log.info(f"[NLP] Written → {NLP_OUTPUT_FILE.name}: {result['transcript'][:60]}")
    except OSError as e:
        log.error(f"[NLP] Failed to write output: {e}")


# ══════════════════════════════════════════════
#  FULL NLP ANALYSIS PIPELINE
#  Called once per utterance with raw audio bytes
# ══════════════════════════════════════════════
def process_utterance(audio_bytes: bytes) -> Optional[dict]:
    """
    Full pipeline for one captured utterance:
      1. STT → transcript
      2. Tone/slang style detection
      3. Text emotion
      4. Intent classification
      5. Entity extraction
      6. Urgency estimation
      7. Build + write output JSON

    Returns the result dict, or None if transcript is empty.
    """
    # --- Step 1: Transcribe ---
    stt_result = transcribe_audio(audio_bytes)
    transcript = stt_result["transcript"].strip()

    if not transcript:
        log.debug("[NLP] Empty transcript — skipping.")
        return None

    language   = stt_result["language"]
    confidence = stt_result["confidence"]

    # --- Step 2: Tone/Slang Style ---
    tone_style = detect_tone_style(transcript, language)

    # --- Step 3: Text Emotion ---
    emotion = detect_text_emotion(transcript)

    # --- Step 4: Intent ---
    intent = classify_intent(transcript)

    # --- Step 5: Entities ---
    entities = extract_entities(transcript)

    # --- Step 6: Urgency ---
    urgency = estimate_urgency(transcript, emotion)

    # --- Step 7: Build output ---
    result = {
        "timestamp":   time.time(),
        "transcript":  transcript,
        "language":    language,
        "stt_conf":    round(confidence, 3),

        # ── Tone / register ──
        "tone_style":  tone_style,
        # tone_style values:
        #   "formal"         → polite, professional English
        #   "casual_english" → relaxed English
        #   "playful"        → slang, lol-speak
        #   "hinglish"       → Hindi-English code-switch
        #   "marathi_mix"    → Marathi words mixed in
        #   "hindi"          → predominantly Hindi

        # ── NLU outputs ──
        "intent":      intent,
        "entities":    entities,
        "urgency":     urgency,
        "sentiment":   emotion,       # text-based emotion fed to LLM as 'sentiment'

        # ── Flag for main.py NLP Listener ──
        "is_new":      True,
    }

    write_nlp_output(result)

    log.info(
        f"[NLP] intent={intent} | urgency={urgency} | emotion={emotion} "
        f"| tone={tone_style} | lang={language}"
    )

    return result


# ══════════════════════════════════════════════
#  MICROPHONE CAPTURE + VAD LOOP
#  This is the main entry function for this module
# ══════════════════════════════════════════════
def run_input_pipeline():
    """
    Blocking loop:
      - Opens mic stream via sounddevice
      - Runs Silero VAD on each 30ms frame
      - Buffers speech frames into utterance chunks
      - On silence after speech → fires process_utterance()

    Call this from a dedicated thread in your launcher,
    or run nlp/input.py directly for standalone testing.
    """
    vad = SileroVAD()

    frame_samples   = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)   # 480 samples at 16kHz / 30ms
    silence_frames  = int(VAD_SILENCE_MS / VAD_FRAME_MS)        # how many silent frames = end of speech
    max_frames      = int(MAX_UTTERANCE_SEC * 1000 / VAD_FRAME_MS)

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def mic_callback(indata, frames, time_info, status):
        """sounddevice callback — runs in audio thread."""
        if status:
            log.warning(f"[MIC] sounddevice status: {status}")
        audio_queue.put(indata[:, 0].copy())   # mono

    log.info(f"[MIC] Opening mic | sr={SAMPLE_RATE}Hz | frame={VAD_FRAME_MS}ms | device={MIC_DEVICE_INDEX}")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=frame_samples,
        device=MIC_DEVICE_INDEX,
        callback=mic_callback,
    ):
        log.info("[MIC] Microphone open. Listening for speech...")

        speech_buffer: list[np.ndarray] = []
        in_speech      = False
        silent_count   = 0

        while True:
            try:
                frame = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            speech_prob = vad.is_speech(frame)
            is_speech_frame = speech_prob >= VAD_THRESHOLD

            if is_speech_frame:
                if not in_speech:
                    log.debug("[VAD] Speech started.")
                    in_speech   = True
                    silent_count = 0
                speech_buffer.append(frame)
                silent_count = 0

            else:
                if in_speech:
                    silent_count += 1
                    speech_buffer.append(frame)    # include trailing silence for context

                    # --- End of utterance: enough silence ---
                    if silent_count >= silence_frames:
                        in_speech = False
                        log.debug(f"[VAD] Utterance ended. {len(speech_buffer)} frames captured.")

                        audio_np    = np.concatenate(speech_buffer)
                        audio_int16 = (audio_np * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()

                        speech_buffer = []
                        silent_count  = 0

                        # Process in thread so we don't miss audio
                        threading.Thread(
                            target=process_utterance,
                            args=(audio_bytes,),
                            daemon=True,
                        ).start()

                    # --- Safety: max utterance length ---
                    elif len(speech_buffer) >= max_frames:
                        in_speech = False
                        log.warning("[VAD] Max utterance length reached — flushing.")
                        audio_np    = np.concatenate(speech_buffer)
                        audio_int16 = (audio_np * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                        speech_buffer = []
                        silent_count  = 0
                        threading.Thread(
                            target=process_utterance,
                            args=(audio_bytes,),
                            daemon=True,
                        ).start()


# ══════════════════════════════════════════════
#  STANDALONE ENTRY (for testing)
# ══════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log.info("Running NLP input pipeline in standalone mode.")
    run_input_pipeline()