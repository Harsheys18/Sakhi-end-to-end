"""
╔════════════════════════════════════════════════════════════════════════════╗
║              SAKHI — NLP OUTPUT PIPELINE                                   ║
║              nlp/output.py                                                 ║
║                                                                            ║
║  What this does:                                                           ║
║    1. Receives text response from main.py brain loop                       ║
║    2. Adapts the text to the guest's detected tone_style:                  ║
║         - formal       → clean, polished output                            ║
║         - playful      → adds warmth, light expressiveness                 ║
║         - hinglish     → keeps Hinglish-friendly phrasing                  ║
║         - marathi_mix  → inserts light Marathi flavour words               ║
║         - hindi        → pass to TTS as-is (Deepgram handles)             ║
║    3. Converts text → speech via Deepgram TTS API (Aura voices)           ║
║    4. Plays audio on speaker via sounddevice                               ║
║    5. Exposes speak(text) as the single public interface for main.py       ║
║                                                                            ║
║  Imported by main.py:   from nlp.output import speak                       ║
║  Config:                main_config.yaml → nlp section                    ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import io
import json
import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml

# ─────────────────────────────────────────────
#  CONFIG LOADING
# ─────────────────────────────────────────────
CONFIG_PATH = Path("main_config.yaml")
if not CONFIG_PATH.exists():
    # When imported from a different working directory
    CONFIG_PATH = Path(__file__).parent.parent / "main_config.yaml"

CONFIG = yaml.safe_load(CONFIG_PATH.open())

NLP_CFG = CONFIG.get("nlp", {})

DEEPGRAM_API_KEY     = NLP_CFG.get("deepgram_api_key", os.environ.get("DEEPGRAM_API_KEY", ""))
OUTPUTS_DIR          = Path(CONFIG["paths"]["base_dir"]) / CONFIG["paths"]["outputs_dir"]
NLP_OUTPUT_FILE      = OUTPUTS_DIR / CONFIG["paths"]["nlp_output_file"]

# TTS config
TTS_VOICE_ENGLISH    = NLP_CFG.get("tts_voice_english",  "aura-asteria-en")
TTS_VOICE_HINDI      = NLP_CFG.get("tts_voice_hindi",    "aura-asteria-en")   # Deepgram handles Hindi
TTS_SAMPLE_RATE      = int(NLP_CFG.get("tts_sample_rate", 24000))
SPEAKER_DEVICE_INDEX = NLP_CFG.get("speaker_device_index", None)               # None = system default

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
log = logging.getLogger("SAKHI.NLP.Output")

# Thread lock — only one TTS playback at a time
_speak_lock = threading.Lock()


# ══════════════════════════════════════════════
#  TONE ADAPTER
#  Adjusts the LLM's response text to match
#  the guest's detected speaking style.
# ══════════════════════════════════════════════

# Light Marathi filler/flavour words to sprinkle in naturally
_MARATHI_FILLERS = ["Ha,", "Aho,", "Bagh,", "Chala,"]

# Hinglish softeners to occasionally insert
_HINGLISH_SOFTENERS = ["Bilkul,", "Haan,", "Acha,", "Theek hai,"]


def adapt_text_to_tone(text: str, tone_style: str, language: str) -> str:
    """
    Lightly adapts the response text to match the guest's tone_style.

    Rules (all kept subtle — SAKHI never overdoes it):
      formal       → strip any slang that crept in, clean punctuation
      playful      → text is returned as-is (LLM already matched it via system prompt)
      hinglish     → prepend a soft Hinglish acknowledgement word occasionally
      marathi_mix  → prepend a light Marathi filler word occasionally
      hindi        → return as-is
      casual_en    → return as-is

    This is intentionally lightweight — the LLM system prompt already steers
    the response tone. This layer adds only micro-adjustments.
    """
    if not text:
        return text

    if tone_style == "formal":
        # Ensure proper capitalisation and no informal contractions
        text = text.replace(" gonna ", " going to ")
        text = text.replace(" wanna ", " want to ")
        text = text.replace(" gotta ", " have to ")
        text = text.replace(" kinda ", " kind of ")
        # Ensure it ends with a period if it doesn't already end with punctuation
        if text and text[-1] not in ".!?":
            text += "."
        return text

    if tone_style == "marathi_mix":
        # Rotate through filler words based on current second (avoids always picking same one)
        filler = _MARATHI_FILLERS[int(time.time()) % len(_MARATHI_FILLERS)]
        # Only prepend if the response doesn't already start with a greeting-type word
        if not any(text.lower().startswith(g) for g in ("hello", "hi", "hey", "sure", "of course")):
            text = f"{filler} {text}"
        return text

    if tone_style == "hinglish":
        softener = _HINGLISH_SOFTENERS[int(time.time()) % len(_HINGLISH_SOFTENERS)]
        if not any(text.lower().startswith(g) for g in ("hello", "hi", "hey", "sure", "of course")):
            text = f"{softener} {text}"
        return text

    # playful, casual_english, hindi → pass through
    return text


# ══════════════════════════════════════════════
#  DEEPGRAM TTS — text → audio bytes
# ══════════════════════════════════════════════
def _synthesise_deepgram(text: str, language: str) -> Optional[bytes]:
    """
    Calls Deepgram Aura TTS REST API.
    Returns raw audio bytes (mp3 or linear16 depending on config),
    or None on failure.

    Deepgram Aura supports English. For Hindi/Marathi mixing, the
    English voice pronounces transliterated words naturally enough
    for restaurant context.
    """
    import requests

    if not DEEPGRAM_API_KEY:
        log.error("[TTS] DEEPGRAM_API_KEY not set. Cannot synthesise speech.")
        return None

    voice = TTS_VOICE_HINDI if language in ("hi", "mr") else TTS_VOICE_ENGLISH

    url     = f"https://api.deepgram.com/v1/speak?model={voice}"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {"text": text}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10, stream=True)
        resp.raise_for_status()
        audio_bytes = resp.content
        log.info(f"[TTS] Deepgram synthesised {len(audio_bytes)} bytes | voice={voice}")
        return audio_bytes

    except Exception as e:
        log.error(f"[TTS] Deepgram TTS error: {e}")
        return None


# ══════════════════════════════════════════════
#  AUDIO PLAYBACK
# ══════════════════════════════════════════════
def _play_audio_bytes(audio_bytes: bytes):
    """
    Decodes mp3/wav audio bytes using soundfile and plays via sounddevice.
    Blocks until playback is complete.
    """
    try:
        buf = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(buf, dtype="float32")

        # Ensure mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        log.info(f"[TTS] Playing audio | sr={samplerate}Hz | duration={len(data)/samplerate:.2f}s")

        sd.play(data, samplerate=samplerate, device=SPEAKER_DEVICE_INDEX)
        sd.wait()   # blocks until done

    except Exception as e:
        log.error(f"[TTS] Audio playback error: {e}")


# ══════════════════════════════════════════════
#  TONE STYLE READER
#  Reads the last detected tone_style from the
#  NLP output file so speak() can auto-adapt.
# ══════════════════════════════════════════════
def _read_current_tone() -> tuple[str, str]:
    """
    Returns (tone_style, language) from the latest nlp_output.json.
    Falls back to ("casual_english", "en") if file missing or unreadable.
    """
    try:
        if NLP_OUTPUT_FILE.exists():
            with open(NLP_OUTPUT_FILE, "r") as f:
                data = json.load(f)
            return (
                data.get("tone_style", "casual_english"),
                data.get("language",   "en"),
            )
    except Exception:
        pass
    return "casual_english", "en"


# ══════════════════════════════════════════════
#  FALLBACK: PRINT TO CONSOLE
#  Used when Deepgram key missing or TTS fails.
# ══════════════════════════════════════════════
def _fallback_print(text: str):
    """Print response to terminal when TTS is unavailable."""
    print(f"\n🤖 SAKHI: {text}\n")


# ══════════════════════════════════════════════
#  PUBLIC API
#  speak(text) — called by main.py brain loop
# ══════════════════════════════════════════════
def speak(text: str, tone_style: Optional[str] = None, language: Optional[str] = None):
    """
    Main public function. Converts text to speech and plays it.

    Args:
        text:       The response string to speak.
        tone_style: Override tone style (optional). If None, auto-reads from
                    nlp_output.json to match the guest's detected style.
        language:   Override language code (optional). If None, auto-reads.

    Called by main.py:
        from nlp.output import speak
        speak("Hello! Welcome to the restaurant.")
    """
    if not text or not text.strip():
        log.warning("[TTS] speak() called with empty text — skipping.")
        return

    # Acquire lock — one playback at a time
    with _speak_lock:
        # --- Auto-detect tone from latest NLP output ---
        if tone_style is None or language is None:
            detected_tone, detected_lang = _read_current_tone()
            tone_style = tone_style or detected_tone
            language   = language   or detected_lang

        # --- Adapt text to tone ---
        adapted_text = adapt_text_to_tone(text, tone_style, language)
        log.info(f"[TTS] speak() | tone={tone_style} | lang={language} | text: {adapted_text[:80]}")

        # --- Synthesise via Deepgram ---
        audio_bytes = _synthesise_deepgram(adapted_text, language)

        if audio_bytes:
            _play_audio_bytes(audio_bytes)
        else:
            # Graceful fallback — never crash the brain loop
            log.warning("[TTS] TTS failed — using console fallback.")
            _fallback_print(adapted_text)


# ══════════════════════════════════════════════
#  CONVENIENCE: speak_async()
#  Non-blocking version for use in background threads.
# ══════════════════════════════════════════════
def speak_async(text: str, tone_style: Optional[str] = None, language: Optional[str] = None):
    """
    Non-blocking speak — fires and forgets.
    Use when you don't want the caller to wait for playback.
    """
    threading.Thread(
        target=speak,
        args=(text, tone_style, language),
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

    log.info("Running NLP output pipeline in standalone test mode.")

    # Test all tone styles
    test_cases = [
        ("Hello! Welcome to our restaurant. How may I assist you today?",      "formal",         "en"),
        ("Hey! Welcome! What can I get you? We've got some great specials today!", "playful",      "en"),
        ("Bilkul! Aapka swagat hai. Kya loge aap aaj?",                           "hinglish",     "hi"),
        ("Ha, bagh — aamhi specials aahot aaj. Sangto tumhala!",                  "marathi_mix",  "mr"),
        ("Sure! What can I get for you today?",                                   "casual_english","en"),
    ]

    for text, tone, lang in test_cases:
        print(f"\n--- Testing tone: {tone} | lang: {lang} ---")
        speak(text, tone_style=tone, language=lang)
        time.sleep(0.5)