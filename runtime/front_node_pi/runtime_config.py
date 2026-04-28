#!/usr/bin/env python3
"""Configuration helpers for the structured front-node Pi runtime."""

from __future__ import annotations

from dataclasses import dataclass
import configparser
import os
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNTIME_DIR.parent.parent
DEFAULT_CONFIG_PATH = RUNTIME_DIR / "config.ini"
DEFAULT_EVIDENCE_ROOT = RUNTIME_DIR / "data" / "evidence_combined"
DEFAULT_SETUP_PROFILE_DIR = RUNTIME_DIR / "data" / "setup_profiles"
CONFIG_ENV_VAR = "AISENTINEL_FRONT_NODE_CONFIG"
DEFAULT_VIDEO_CONFIG_PATH = RUNTIME_DIR / "config_video.ini"
DEFAULT_WEBCAM_CONFIG_PATH = RUNTIME_DIR / "config_webcam.ini"


@dataclass(frozen=True)
class TrackingConfig:
    iou_threshold: float
    max_lost: int


@dataclass(frozen=True)
class VideoSourceConfig:
    default_video: Path | None
    default_setup_profile: Path | None
    auto_use_saved_setup: bool


@dataclass(frozen=True)
class WebcamSourceConfig:
    camera_index: int
    camera_name: str
    capture_width: int
    capture_height: int
    capture_fps: float
    warmup_frames: int
    default_setup_profile: Path | None
    auto_use_saved_setup: bool


@dataclass(frozen=True)
class HeadBehaviorConfig:
    head_tilt_angle_deg: float
    head_turn_ratio: float
    shoulder_turn_angle_deg: float
    sustained_sec: float
    event_cooldown_sec: float
    keypoint_confidence: float


@dataclass(frozen=True)
class PassingPapersConfig:
    event_cooldown_sec: float
    keypoint_confidence: float
    row_tolerance_px: int
    reference_bbox_height: float
    wrist_proximity_px: int
    min_interaction_sec: float


@dataclass(frozen=True)
class HandsUnderTableConfig:
    hand_confidence: float
    person_confidence: float
    hands_missing_sustain_sec: float
    event_cooldown_sec: float
    min_visible_hands: int
    hand_assoc_margin_px: int
    smooth_window_frames: int
    smooth_missing_ratio: float
    student_absent_reset_sec: float
    table_edge_near_px: int
    edge_disappear_arm_sec: float


@dataclass(frozen=True)
class ObjectDetectionConfig:
    person_confidence: float
    phone_confidence: float
    cheat_sheet_confidence: float
    event_cooldown_sec: float
    assoc_iou_thresh: float


@dataclass(frozen=True)
class EvidenceConfig:
    pre_event_frames: int
    post_event_frames: int


@dataclass(frozen=True)
class SpamSuppressionConfig:
    duplicate_suppression_sec: float
    clear_required_sec: float


@dataclass(frozen=True)
class SoundSensorConfig:
    enabled: bool
    calibration_config: Path | None
    alert_threshold_db: float
    incident_cooldown_sec: float
    i2c_bus: int
    i2c_address: int
    adc_channel: int
    full_scale: float
    data_rate: int
    sample_interval: float
    window_seconds: float


@dataclass(frozen=True)
class WebDashboardConfig:
    username: str
    password: str
    secret_key: str
    session_ttl_minutes: int


@dataclass(frozen=True)
class FrontNodeRuntimeConfig:
    config_path: Path
    pose_model: Path
    hand_model: Path
    object_model: Path
    pose_confidence: float
    object_confidence: float
    port: int
    default_video: Path | None
    default_setup_profile: Path | None
    auto_use_saved_setup: bool
    evidence_root: Path
    setup_profile_dir: Path
    video_source: VideoSourceConfig
    webcam_source: WebcamSourceConfig
    tracking: TrackingConfig
    head_behavior: HeadBehaviorConfig
    passing_papers: PassingPapersConfig
    hands_under_table: HandsUnderTableConfig
    object_detection: ObjectDetectionConfig
    evidence: EvidenceConfig
    spam_suppression: SpamSuppressionConfig
    sound_sensor: SoundSensorConfig
    web_dashboard: WebDashboardConfig


def resolve_cli_path(raw_value: str | None) -> Path | None:
    """Resolve a CLI or environment path relative to the current working dir."""
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def resolve_repo_path(raw_value: str | None) -> Path | None:
    """Resolve a config path relative to the repository root."""
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def resolve_config_path(
    config_arg: str | None = None,
    default_config_path: Path | None = None,
) -> Path:
    """Resolve the active config file path."""
    config_path = resolve_cli_path(config_arg)
    if config_path is not None:
        return config_path

    env_path = resolve_cli_path(os.environ.get(CONFIG_ENV_VAR))
    if env_path is not None:
        return env_path

    return default_config_path or DEFAULT_CONFIG_PATH


def _get_value(
    parser: configparser.ConfigParser,
    sections: list[str],
    option: str,
    fallback,
    getter_name: str = "get",
):
    for section in sections:
        if parser.has_option(section, option):
            getter = getattr(parser, getter_name)
            return getter(section, option)
    return fallback


def load_runtime_config(
    config_arg: str | None = None,
    default_config_path: Path | None = None,
) -> FrontNodeRuntimeConfig:
    """Load runtime settings from the selected INI file."""
    config_path = resolve_config_path(config_arg, default_config_path)
    parser = configparser.ConfigParser()
    loaded = parser.read(config_path, encoding="utf-8")
    if not loaded:
        raise FileNotFoundError(f"Runtime config not found: {config_path}")

    default_video = resolve_repo_path(
        _get_value(
            parser,
            ["video_source", "runtime"],
            "default_video",
            "",
        )
    )
    default_setup_profile = resolve_repo_path(
        _get_value(
            parser,
            ["video_source", "runtime"],
            "default_setup_profile",
            "",
        )
    )
    auto_use_saved_setup = _get_value(
        parser,
        ["video_source", "runtime"],
        "auto_use_saved_setup",
        True,
        getter_name="getboolean",
    )
    webcam_default_setup_profile = resolve_repo_path(
        _get_value(
            parser,
            ["webcam_source"],
            "default_setup_profile",
            "",
        )
    )
    webcam_auto_use_saved_setup = _get_value(
        parser,
        ["webcam_source"],
        "auto_use_saved_setup",
        True,
        getter_name="getboolean",
    )
    dashboard_secret_key = _get_value(
        parser,
        ["web_dashboard"],
        "secret_key",
        os.environ.get("AISENTINEL_DASHBOARD_SECRET", "change-this-secret-key"),
    )

    return FrontNodeRuntimeConfig(
        config_path=config_path,
        pose_model=resolve_repo_path(
            _get_value(parser, ["models"], "pose", "models/yolov8s_pose.hef")
        ),
        hand_model=resolve_repo_path(
            _get_value(
                parser,
                ["models"],
                "hand",
                "models/sentinel-yolo11n-min.hef",
            )
        ),
        object_model=resolve_repo_path(
            _get_value(
                parser,
                ["models"],
                "object",
                "models/object-updated.hef",
            )
        ),
        pose_confidence=_get_value(
            parser,
            ["inference", "thresholds"],
            "pose_confidence",
            0.5,
            getter_name="getfloat",
        ),
        object_confidence=_get_value(
            parser,
            ["inference", "thresholds"],
            "object_confidence",
            0.25,
            getter_name="getfloat",
        ),
        port=_get_value(
            parser,
            ["runtime"],
            "port",
            8080,
            getter_name="getint",
        ),
        default_video=default_video,
        default_setup_profile=default_setup_profile,
        auto_use_saved_setup=auto_use_saved_setup,
        evidence_root=resolve_repo_path(
            _get_value(
                parser,
                ["outputs"],
                "evidence_root",
                str(DEFAULT_EVIDENCE_ROOT.relative_to(REPO_ROOT)),
            )
        ),
        setup_profile_dir=resolve_repo_path(
            _get_value(
                parser,
                ["outputs"],
                "setup_profile_dir",
                str(DEFAULT_SETUP_PROFILE_DIR.relative_to(REPO_ROOT)),
            )
        ),
        video_source=VideoSourceConfig(
            default_video=default_video,
            default_setup_profile=default_setup_profile,
            auto_use_saved_setup=auto_use_saved_setup,
        ),
        webcam_source=WebcamSourceConfig(
            camera_index=_get_value(
                parser,
                ["webcam_source"],
                "camera_index",
                0,
                getter_name="getint",
            ),
            camera_name=_get_value(
                parser,
                ["webcam_source"],
                "camera_name",
                "webcam_0",
            ),
            capture_width=_get_value(
                parser,
                ["webcam_source"],
                "capture_width",
                0,
                getter_name="getint",
            ),
            capture_height=_get_value(
                parser,
                ["webcam_source"],
                "capture_height",
                0,
                getter_name="getint",
            ),
            capture_fps=_get_value(
                parser,
                ["webcam_source"],
                "capture_fps",
                30.0,
                getter_name="getfloat",
            ),
            warmup_frames=_get_value(
                parser,
                ["webcam_source"],
                "warmup_frames",
                15,
                getter_name="getint",
            ),
            default_setup_profile=webcam_default_setup_profile,
            auto_use_saved_setup=webcam_auto_use_saved_setup,
        ),
        tracking=TrackingConfig(
            iou_threshold=_get_value(
                parser,
                ["tracking"],
                "iou_threshold",
                0.3,
                getter_name="getfloat",
            ),
            max_lost=_get_value(
                parser,
                ["tracking"],
                "max_lost",
                60,
                getter_name="getint",
            ),
        ),
        head_behavior=HeadBehaviorConfig(
            head_tilt_angle_deg=_get_value(
                parser,
                ["head_behavior"],
                "head_tilt_angle_deg",
                30.0,
                getter_name="getfloat",
            ),
            head_turn_ratio=_get_value(
                parser,
                ["head_behavior"],
                "head_turn_ratio",
                0.18,
                getter_name="getfloat",
            ),
            shoulder_turn_angle_deg=_get_value(
                parser,
                ["head_behavior"],
                "shoulder_turn_angle_deg",
                20.0,
                getter_name="getfloat",
            ),
            sustained_sec=_get_value(
                parser,
                ["head_behavior"],
                "sustained_sec",
                2.5,
                getter_name="getfloat",
            ),
            event_cooldown_sec=_get_value(
                parser,
                ["head_behavior"],
                "event_cooldown_sec",
                10.0,
                getter_name="getfloat",
            ),
            keypoint_confidence=_get_value(
                parser,
                ["head_behavior"],
                "keypoint_confidence",
                0.3,
                getter_name="getfloat",
            ),
        ),
        passing_papers=PassingPapersConfig(
            event_cooldown_sec=_get_value(
                parser,
                ["passing_papers"],
                "event_cooldown_sec",
                10.0,
                getter_name="getfloat",
            ),
            keypoint_confidence=_get_value(
                parser,
                ["passing_papers"],
                "keypoint_confidence",
                0.3,
                getter_name="getfloat",
            ),
            row_tolerance_px=_get_value(
                parser,
                ["passing_papers"],
                "row_tolerance_px",
                80,
                getter_name="getint",
            ),
            reference_bbox_height=_get_value(
                parser,
                ["passing_papers"],
                "reference_bbox_height",
                300.0,
                getter_name="getfloat",
            ),
            wrist_proximity_px=_get_value(
                parser,
                ["passing_papers"],
                "wrist_proximity_px",
                160,
                getter_name="getint",
            ),
            min_interaction_sec=_get_value(
                parser,
                ["passing_papers"],
                "min_interaction_sec",
                0.03,
                getter_name="getfloat",
            ),
        ),
        hands_under_table=HandsUnderTableConfig(
            hand_confidence=_get_value(
                parser,
                ["hands_under_table"],
                "hand_confidence",
                0.3,
                getter_name="getfloat",
            ),
            person_confidence=_get_value(
                parser,
                ["hands_under_table"],
                "person_confidence",
                0.5,
                getter_name="getfloat",
            ),
            hands_missing_sustain_sec=_get_value(
                parser,
                ["hands_under_table"],
                "hands_missing_sustain_sec",
                3.0,
                getter_name="getfloat",
            ),
            event_cooldown_sec=_get_value(
                parser,
                ["hands_under_table"],
                "event_cooldown_sec",
                10.0,
                getter_name="getfloat",
            ),
            min_visible_hands=_get_value(
                parser,
                ["hands_under_table"],
                "min_visible_hands",
                2,
                getter_name="getint",
            ),
            hand_assoc_margin_px=_get_value(
                parser,
                ["hands_under_table"],
                "hand_assoc_margin_px",
                60,
                getter_name="getint",
            ),
            smooth_window_frames=_get_value(
                parser,
                ["hands_under_table"],
                "smooth_window_frames",
                12,
                getter_name="getint",
            ),
            smooth_missing_ratio=_get_value(
                parser,
                ["hands_under_table"],
                "smooth_missing_ratio",
                0.6,
                getter_name="getfloat",
            ),
            student_absent_reset_sec=_get_value(
                parser,
                ["hands_under_table"],
                "student_absent_reset_sec",
                2.0,
                getter_name="getfloat",
            ),
            table_edge_near_px=_get_value(
                parser,
                ["hands_under_table"],
                "table_edge_near_px",
                35,
                getter_name="getint",
            ),
            edge_disappear_arm_sec=_get_value(
                parser,
                ["hands_under_table"],
                "edge_disappear_arm_sec",
                0.75,
                getter_name="getfloat",
            ),
        ),
        object_detection=ObjectDetectionConfig(
            person_confidence=_get_value(
                parser,
                ["object_detection"],
                "person_confidence",
                0.5,
                getter_name="getfloat",
            ),
            phone_confidence=_get_value(
                parser,
                ["object_detection"],
                "phone_confidence",
                0.4,
                getter_name="getfloat",
            ),
            cheat_sheet_confidence=_get_value(
                parser,
                ["object_detection"],
                "cheat_sheet_confidence",
                0.3,
                getter_name="getfloat",
            ),
            event_cooldown_sec=_get_value(
                parser,
                ["object_detection"],
                "event_cooldown_sec",
                10.0,
                getter_name="getfloat",
            ),
            assoc_iou_thresh=_get_value(
                parser,
                ["object_detection"],
                "assoc_iou_thresh",
                0.05,
                getter_name="getfloat",
            ),
        ),
        evidence=EvidenceConfig(
            pre_event_frames=_get_value(
                parser,
                ["evidence"],
                "pre_event_frames",
                10,
                getter_name="getint",
            ),
            post_event_frames=_get_value(
                parser,
                ["evidence"],
                "post_event_frames",
                10,
                getter_name="getint",
            ),
        ),
        spam_suppression=SpamSuppressionConfig(
            duplicate_suppression_sec=max(
                0.0,
                _get_value(
                    parser,
                    ["spam_suppression"],
                    "duplicate_suppression_sec",
                    60.0,
                    getter_name="getfloat",
                ),
            ),
            clear_required_sec=max(
                0.0,
                _get_value(
                    parser,
                    ["spam_suppression"],
                    "clear_required_sec",
                    3.0,
                    getter_name="getfloat",
                ),
            ),
        ),
        sound_sensor=SoundSensorConfig(
            enabled=_get_value(
                parser,
                ["sound_sensor"],
                "enabled",
                False,
                getter_name="getboolean",
            ),
            calibration_config=resolve_repo_path(
                _get_value(
                    parser,
                    ["sound_sensor"],
                    "calibration_config",
                    "",
                )
            ),
            alert_threshold_db=_get_value(
                parser,
                ["sound_sensor"],
                "alert_threshold_db",
                55.0,
                getter_name="getfloat",
            ),
            incident_cooldown_sec=_get_value(
                parser,
                ["sound_sensor"],
                "incident_cooldown_sec",
                10.0,
                getter_name="getfloat",
            ),
            i2c_bus=_get_value(
                parser,
                ["sound_sensor"],
                "i2c_bus",
                1,
                getter_name="getint",
            ),
            i2c_address=int(
                str(
                    _get_value(
                        parser,
                        ["sound_sensor"],
                        "i2c_address",
                        "0x48",
                    )
                ),
                0,
            ),
            adc_channel=_get_value(
                parser,
                ["sound_sensor"],
                "adc_channel",
                0,
                getter_name="getint",
            ),
            full_scale=_get_value(
                parser,
                ["sound_sensor"],
                "full_scale",
                4.096,
                getter_name="getfloat",
            ),
            data_rate=_get_value(
                parser,
                ["sound_sensor"],
                "data_rate",
                1600,
                getter_name="getint",
            ),
            sample_interval=_get_value(
                parser,
                ["sound_sensor"],
                "sample_interval",
                0.002,
                getter_name="getfloat",
            ),
            window_seconds=_get_value(
                parser,
                ["sound_sensor"],
                "window_seconds",
                1.0,
                getter_name="getfloat",
            ),
        ),
        web_dashboard=WebDashboardConfig(
            username=_get_value(
                parser,
                ["web_dashboard"],
                "username",
                "admin",
            ),
            password=_get_value(
                parser,
                ["web_dashboard"],
                "password",
                "admin123",
            ),
            secret_key=dashboard_secret_key.strip() or "change-this-secret-key",
            session_ttl_minutes=max(
                1,
                _get_value(
                    parser,
                    ["web_dashboard"],
                    "session_ttl_minutes",
                    480,
                    getter_name="getint",
                ),
            ),
        ),
    )
