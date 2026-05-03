#!/usr/bin/env python3
"""Runtime-like INI support for active Pi behavior test scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent.parent
RUNTIME_CONFIG_PATH = REPO_ROOT / "runtime" / "edge_node_runtime" / "runtime_config.py"

DEFAULT_CONFIG_PATH = TEST_DIR / "config.ini"
DEFAULT_CONFIG_EXAMPLE_PATH = TEST_DIR / "config.ini.example"
DEFAULT_VIDEO_CONFIG_PATH = DEFAULT_CONFIG_PATH
DEFAULT_WEBCAM_CONFIG_PATH = DEFAULT_CONFIG_PATH


def _load_runtime_config_module():
    module_name = "_aisentinel_front_node_runtime_config"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, RUNTIME_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runtime config parser from {RUNTIME_CONFIG_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_runtime_config = _load_runtime_config_module()
load_runtime_config = _runtime_config.load_runtime_config


def _is_default_config_path(path) -> bool:
    return Path(path).resolve(strict=False) == DEFAULT_CONFIG_PATH


def add_config_arg(parser, default_config_path=DEFAULT_CONFIG_PATH) -> None:
    fallback_note = ""
    if _is_default_config_path(default_config_path):
        fallback_note = f"; falls back to {DEFAULT_CONFIG_EXAMPLE_PATH}"
    parser.add_argument(
        "--config",
        default=None,
        help=(
            f"Runtime-like INI config path (default: {default_config_path}"
            f"{fallback_note})"
        ),
    )


def load_test_config(config_arg=None, default_config_path=DEFAULT_CONFIG_PATH):
    if config_arg is None:
        default_path = Path(default_config_path)
        if _is_default_config_path(default_path) and not default_path.exists():
            default_config_path = DEFAULT_CONFIG_EXAMPLE_PATH

    return load_runtime_config(config_arg, default_config_path)


def path_arg(path) -> str | None:
    return str(path) if path is not None else None


def cli_or_config(cli_value, config_value):
    return cli_value if cli_value is not None else config_value


def apply_head_config(head_mod, config) -> None:
    head_mod.POSE_MODEL_PATH = config.pose_model
    head_mod.EVIDENCE_DIR = config.evidence_root / "head_behavior"
    head_mod.HEAD_TILT_ANGLE_DEG = config.head_behavior.head_tilt_angle_deg
    head_mod.HEAD_TURN_RATIO = config.head_behavior.head_turn_ratio
    head_mod.SHOULDER_TURN_ANGLE_DEG = config.head_behavior.shoulder_turn_angle_deg
    head_mod.SUSTAINED_SEC = config.head_behavior.sustained_sec
    head_mod.EVENT_COOLDOWN_SEC = config.head_behavior.event_cooldown_sec
    head_mod.KP_CONF_THRESH = config.head_behavior.keypoint_confidence


def apply_passing_config(pass_mod, config) -> None:
    pass_mod.POSE_MODEL_PATH = config.pose_model
    pass_mod.EVIDENCE_DIR = config.evidence_root / "passing_papers"
    pass_mod.EVENT_COOLDOWN_SEC = config.passing_papers.event_cooldown_sec
    pass_mod.KP_CONF_THRESH = config.passing_papers.keypoint_confidence
    pass_mod.ROW_TOLERANCE_PX = config.passing_papers.row_tolerance_px
    pass_mod.REFERENCE_BBOX_HEIGHT = config.passing_papers.reference_bbox_height
    pass_mod.WRIST_PROXIMITY_PX = config.passing_papers.wrist_proximity_px
    pass_mod.MIN_INTERACTION_SEC = config.passing_papers.min_interaction_sec


def apply_hands_config(hands_mod, config) -> None:
    hands_mod.POSE_MODEL_PATH = config.pose_model
    hands_mod.HAND_MODEL_PATH = config.hand_model
    hands_mod.EVIDENCE_DIR = config.evidence_root / "hands"
    hands_mod.HAND_CONFIDENCE = config.hands_under_table.hand_confidence
    hands_mod.PERSON_CONFIDENCE = config.hands_under_table.person_confidence
    hands_mod.HANDS_MISSING_SUSTAIN_SEC = (
        config.hands_under_table.hands_missing_sustain_sec
    )
    hands_mod.EVENT_COOLDOWN_SEC = config.hands_under_table.event_cooldown_sec
    hands_mod.MIN_VISIBLE_HANDS = config.hands_under_table.min_visible_hands
    hands_mod.HAND_ASSOC_MARGIN_PX = config.hands_under_table.hand_assoc_margin_px
    hands_mod.SMOOTH_WINDOW_FRAMES = config.hands_under_table.smooth_window_frames
    hands_mod.SMOOTH_MISSING_RATIO = config.hands_under_table.smooth_missing_ratio
    hands_mod.STUDENT_ABSENT_RESET_SEC = (
        config.hands_under_table.student_absent_reset_sec
    )
    hands_mod.TABLE_EDGE_NEAR_PX = config.hands_under_table.table_edge_near_px
    hands_mod.EDGE_DISAPPEAR_ARM_SEC = config.hands_under_table.edge_disappear_arm_sec


def apply_object_config(obj_mod, config) -> None:
    obj_mod.POSE_MODEL_PATH = config.pose_model
    obj_mod.OBJ_MODEL_PATH = config.object_model
    obj_mod.EVIDENCE_DIR = config.evidence_root / "objects"
    obj_mod.PERSON_CONFIDENCE = config.object_detection.person_confidence
    obj_mod.EVENT_COOLDOWN_SEC = config.object_detection.event_cooldown_sec
    obj_mod.ASSOC_IOU_THRESH = config.object_detection.assoc_iou_thresh
    obj_mod.CONFIDENCE_THRESHOLDS["phone"] = config.object_detection.phone_confidence
    obj_mod.CONFIDENCE_THRESHOLDS["cheat_sheet"] = (
        config.object_detection.cheat_sheet_confidence
    )


def apply_all_behavior_config(combined_mod, config) -> None:
    apply_head_config(combined_mod.head_mod, config)
    apply_passing_config(combined_mod.pass_mod, config)
    apply_hands_config(combined_mod.hands_mod, config)
    apply_object_config(combined_mod.obj_mod, config)

    combined_mod.POSE_MODEL_PATH = config.pose_model
    combined_mod.HAND_MODEL_PATH = config.hand_model
    combined_mod.OBJECT_MODEL_PATH = config.object_model
    combined_mod.EVIDENCE_DIR = config.evidence_root
    combined_mod.HEAD_EVIDENCE_DIR = config.evidence_root / "head_behavior"
    combined_mod.PASSING_EVIDENCE_DIR = config.evidence_root / "passing_papers"
    combined_mod.HANDS_EVIDENCE_DIR = config.evidence_root / "hands"
    combined_mod.OBJECT_EVIDENCE_DIR = config.evidence_root / "objects"
    combined_mod.EVIDENCE_PRE_EVENT_FRAMES = config.evidence.pre_event_frames
    combined_mod.EVIDENCE_POST_EVENT_FRAMES = config.evidence.post_event_frames

    if hasattr(combined_mod, "DUPLICATE_SUPPRESSION_SEC"):
        combined_mod.DUPLICATE_SUPPRESSION_SEC = (
            config.spam_suppression.duplicate_suppression_sec
        )
    if hasattr(combined_mod, "SUPPRESSION_CLEAR_REQUIRED_SEC"):
        combined_mod.SUPPRESSION_CLEAR_REQUIRED_SEC = (
            config.spam_suppression.clear_required_sec
        )
    if hasattr(combined_mod, "setup_io"):
        combined_mod.setup_io.SETUP_PROFILE_DIR = config.setup_profile_dir
