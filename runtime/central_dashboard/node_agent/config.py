"""Configuration loader for the standalone node agent."""

from __future__ import annotations

from dataclasses import dataclass
import configparser
import logging
import os
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent.parent
PLACEHOLDER_TOKENS = ("CHANGE_ME", "CENTRAL_DASHBOARD_HOST_OR_IP", "dev-key", "admin123")


def _resolve_path(raw_value: str | None) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def _get_first_value(
    parser: configparser.ConfigParser,
    candidates: tuple[tuple[str, str], ...],
    fallback,
    getter_name: str = "get",
):
    for section, option in candidates:
        if parser.has_option(section, option):
            getter = getattr(parser, getter_name)
            return getter(section, option)
    return fallback


@dataclass(frozen=True, slots=True)
class NodeAgentConfig:
    config_path: Path
    node_id: str
    display_name: str
    camera_label: str
    profile: str
    host: str
    port: int
    api_key: str
    central_base_url: str
    registration_interval_sec: float
    heartbeat_interval_sec: float
    http_timeout_sec: float
    source_mode: str
    camera_index: int
    video_path: Path | None
    preview_width: int
    preview_fps: float
    jpeg_quality: int
    detector_mode: str
    runtime_config_path: Path | None
    motion_threshold: float
    motion_min_area_ratio: float
    motion_cooldown_sec: float
    annotated_banner_ttl_sec: float
    evidence_root: Path
    pre_event_frames: int
    post_event_frames: int
    gif_frame_count: int = 5
    gif_max_width: int = 640
    gif_fps: float = 4.0
    startup_detection_delay_sec: float = 5.0


def load_node_agent_config(config_path: str | os.PathLike[str]) -> NodeAgentConfig:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve(strict=False)

    parser = configparser.ConfigParser()
    loaded = parser.read(path, encoding="utf-8")
    if not loaded:
        raise FileNotFoundError(f"Node agent config not found: {path}")

    detector_mode = parser.get("detector", "mode", fallback="motion").strip() or "motion"
    runtime_config_path = _resolve_path(
        parser.get("detector", "runtime_config_path", fallback="")
    )
    if detector_mode == "front_runtime" and runtime_config_path is None:
        runtime_config_path = path
    _warn_on_placeholder_values(path, parser)

    evidence_root = _resolve_path(
        _get_first_value(
            parser,
            (("evidence", "root"), ("outputs", "evidence_root")),
            "runtime/central_dashboard/data/node_front/evidence",
        )
    ) or (REPO_ROOT / "runtime/central_dashboard/data/node_front/evidence")

    return NodeAgentConfig(
        config_path=path,
        node_id=parser.get("agent", "node_id", fallback="front").strip() or "front",
        display_name=parser.get("agent", "display_name", fallback="Node Agent").strip() or "Node Agent",
        camera_label=parser.get("agent", "camera_label", fallback="Camera").strip() or "Camera",
        profile=parser.get("agent", "profile", fallback="default").strip() or "default",
        host=parser.get("agent", "host", fallback="0.0.0.0").strip() or "0.0.0.0",
        port=parser.getint("agent", "port", fallback=8091),
        api_key=parser.get("agent", "api_key", fallback="dev-key").strip() or "dev-key",
        central_base_url=parser.get("agent", "central_base_url", fallback="http://127.0.0.1:8090").rstrip("/"),
        registration_interval_sec=max(
            2.0,
            parser.getfloat("agent", "registration_interval_sec", fallback=10.0),
        ),
        heartbeat_interval_sec=max(
            1.0,
            parser.getfloat("agent", "heartbeat_interval_sec", fallback=3.0),
        ),
        http_timeout_sec=max(
            1.0,
            parser.getfloat("agent", "http_timeout_sec", fallback=5.0),
        ),
        source_mode=parser.get("capture", "source_mode", fallback="webcam").strip() or "webcam",
        camera_index=_get_first_value(
            parser,
            (("capture", "camera_index"), ("webcam_source", "camera_index")),
            0,
            getter_name="getint",
        ),
        video_path=_resolve_path(parser.get("capture", "video_path", fallback="")),
        preview_width=max(320, parser.getint("preview", "width", fallback=640)),
        preview_fps=max(1.0, parser.getfloat("preview", "fps", fallback=6.0)),
        jpeg_quality=max(20, min(95, parser.getint("preview", "jpeg_quality", fallback=60))),
        detector_mode=detector_mode,
        runtime_config_path=runtime_config_path,
        motion_threshold=max(1.0, parser.getfloat("detector", "motion_threshold", fallback=24.0)),
        motion_min_area_ratio=max(0.001, parser.getfloat("detector", "motion_min_area_ratio", fallback=0.012)),
        motion_cooldown_sec=max(1.0, parser.getfloat("detector", "motion_cooldown_sec", fallback=8.0)),
        annotated_banner_ttl_sec=max(1.0, parser.getfloat("detector", "annotated_banner_ttl_sec", fallback=4.0)),
        startup_detection_delay_sec=max(
            0.0,
            parser.getfloat("detector", "startup_detection_delay_sec", fallback=5.0),
        ),
        evidence_root=evidence_root,
        pre_event_frames=max(1, parser.getint("evidence", "pre_event_frames", fallback=8)),
        post_event_frames=max(1, parser.getint("evidence", "post_event_frames", fallback=8)),
        gif_frame_count=max(1, parser.getint("evidence", "gif_frame_count", fallback=5)),
        gif_max_width=max(1, parser.getint("evidence", "gif_max_width", fallback=640)),
        gif_fps=max(0.1, parser.getfloat("evidence", "gif_fps", fallback=4.0)),
    )


def _warn_on_placeholder_values(config_path: Path, parser: configparser.ConfigParser) -> None:
    logger = logging.getLogger(__name__)
    watched = (
        ("agent", "api_key"),
        ("agent", "central_base_url"),
        ("web_dashboard", "password"),
        ("web_dashboard", "secret_key"),
        ("models", "pose"),
        ("models", "hand"),
        ("models", "object"),
        ("video_source", "default_video"),
        ("sound_sensor", "calibration_config"),
    )
    for section, option in watched:
        if not parser.has_option(section, option):
            continue
        value = parser.get(section, option, fallback="")
        if any(token in value for token in PLACEHOLDER_TOKENS):
            logger.warning(
                "Config %s still contains placeholder/default value for [%s] %s.",
                config_path,
                section,
                option,
            )
