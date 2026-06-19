"""
outputs/cv_bridge.py

Bridge between vision_v1 emissions and the simplified cv_output.json
expected by the main brain loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _infer_emotion(valence: float, arousal: float) -> str:
    if valence >= 0.3 and arousal >= 0.5:
        return "excited"
    if valence >= 0.3:
        return "happy"
    if valence <= -0.3 and arousal >= 0.5:
        return "angry"
    if valence <= -0.3:
        return "sad"
    if arousal <= 0.25:
        return "calm"
    return "neutral"


def build_cv_output(vision: dict | str, *, timestamp: float | None = None) -> dict:
    if isinstance(vision, str):
        obj = json.loads(vision)
    else:
        obj = vision

    face_present = bool(obj.get("face_present", False))
    affect = obj.get("affect", {}) if isinstance(obj.get("affect"), dict) else {}
    social = obj.get("social", {}) if isinstance(obj.get("social"), dict) else {}

    valence = float(affect.get("valence", 0.0))
    arousal = float(affect.get("arousal", 0.0))
    emotion = _infer_emotion(valence, arousal) if face_present else "neutral"

    ts_ms = obj.get("ts_ms")
    if timestamp is None:
        if isinstance(ts_ms, (int, float)):
            timestamp = float(ts_ms) / 1000.0
        else:
            timestamp = time.time()

    users_detected = []
    if face_present:
        users_detected.append({
            "track_id": "p1",
            "emotion": emotion,
            "pose": "sitting",
            "gaze": social.get("attention_direction", "unknown"),
            "distance_cm": obj.get("face_distance_cm"),
            "position": "front",
        })

    return {
        "timestamp": timestamp,
        "is_new": True,
        "scene": "restaurant_table",
        "ambient": "normal_lighting",
        "num_people": 1 if face_present else 0,
        "users_detected": users_detected,
    }


def write_cv_output(vision: dict | str, output_path: str | Path) -> bool:
    payload = build_cv_output(vision)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)
    return True
