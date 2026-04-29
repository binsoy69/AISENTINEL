#!/usr/bin/env python3
"""Shared HEF model path defaults for Pi front-node test programs."""

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = REPO_ROOT / "models"


def first_existing(candidates):
    """Return the first existing model path, or the preferred path if none exist."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])


POSE_MODEL_CANDIDATES = (
    MODELS_DIR / "yolo_pose_model.hef",
    MODELS_DIR / "yolov8s_pose.hef",
    MODELS_DIR / "archive" / "yolov8m_pose.hef",
    MODELS_DIR / "archive" / "yolov11_custom.hef",
)

HAND_MODEL_CANDIDATES = (
    MODELS_DIR / "hand-latest.hef",
    MODELS_DIR / "hand_model.hef",
    MODELS_DIR / "sentinel-yolo11n-min.hef",
)

OBJECT_MODEL_CANDIDATES = (
    MODELS_DIR / "object-updated.hef",
    MODELS_DIR / "cheat-sheet_phone_model.hef",
    MODELS_DIR / "yolov11n-sentinel-new" / "sentinel-yolov11n_new.hef",
    MODELS_DIR / "archive" / "yolov11n-sentinel-new" / "sentinel-yolov11n_new.hef",
)

POSE_MODEL_PATH = first_existing(POSE_MODEL_CANDIDATES)
HAND_MODEL_PATH = first_existing(HAND_MODEL_CANDIDATES)
OBJECT_MODEL_PATH = first_existing(OBJECT_MODEL_CANDIDATES)
