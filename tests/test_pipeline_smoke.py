"""Smoke tests for pipeline glue components."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CV_ROOT = ROOT / "cv"
if str(CV_ROOT) not in sys.path:
    sys.path.insert(0, str(CV_ROOT))

from outputs.cv_bridge import build_cv_output


def test_main_config_structure():
    cfg = yaml.safe_load((ROOT / "main_config.yaml").read_text())
    assert "paths" in cfg
    assert "nlp" in cfg
    assert "cv" in cfg
    assert "grok" in cfg
    assert "tuning" in cfg

    paths = cfg["paths"]
    for key in ["outputs_dir", "memory_dir", "nlp_output_file", "cv_output_file", "reply_output_file"]:
        assert key in paths


def test_cv_bridge_with_face():
    vision = {
        "ts_ms": 1700000000000,
        "face_present": True,
        "affect": {"valence": 0.7, "arousal": 0.4},
        "social": {"attention_direction": "forward"},
    }
    out = build_cv_output(vision)
    assert out["num_people"] == 1
    assert out["users_detected"]
    assert out["users_detected"][0]["emotion"] in {"happy", "excited"}
    assert out["users_detected"][0]["gaze"] == "forward"


def test_cv_bridge_no_face():
    vision = {
        "ts_ms": 1700000000000,
        "face_present": False,
        "affect": {"valence": -0.7, "arousal": 0.9},
    }
    out = build_cv_output(vision)
    assert out["num_people"] == 0
    assert out["users_detected"] == []
