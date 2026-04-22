#!/usr/bin/env python3
"""Shared helpers for the structured front-node Pi runtime."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2

from runtime_config import FrontNodeRuntimeConfig, RUNTIME_DIR


@dataclass(frozen=True)
class FrontNodeRuntimeModules:
    combined_mod: object
    setup_io: object
    head_mod: object
    hands_mod: object
    obj_mod: object
    pass_mod: object


WEBCAM_OPEN_TIMEOUT_MS = 1000
WEBCAM_READ_TIMEOUT_MS = 1000
WEBCAM_STARTUP_READ_ATTEMPTS = 30
WEBCAM_STARTUP_READ_PAUSE_SEC = 0.04
WEBCAM_BUFFER_SIZE = 1
WEBCAM_POST_CONFIG_SETTLE_SEC = 0.30
WEBCAM_MIN_ACCEPTABLE_FPS = 10.0
WEBCAM_MIN_ACCEPTABLE_FPS_RATIO = 0.60
WEBCAM_WARMUP_REQUIRED_SUCCESS_FRAMES = 5
WEBCAM_WARMUP_REQUIRED_SUCCESS_STREAK = 3
WEBCAM_WARMUP_EXTRA_ATTEMPTS = 12
WEBCAM_WARMUP_READ_ATTEMPTS = 2
WEBCAM_FALLBACK_CAPTURE_PROFILES = (
    (640, 480, "640x480 fallback"),
    (0, 0, "driver default"),
)


def load_runtime_modules() -> FrontNodeRuntimeModules:
    """Import the self-contained front-node Pi runtime modules."""
    if str(RUNTIME_DIR) not in sys.path:
        sys.path.insert(0, str(RUNTIME_DIR))

    def _load_local_module(module_name: str):
        module_path = RUNTIME_DIR / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec for {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    return FrontNodeRuntimeModules(
        head_mod=_load_local_module("front_node_head_behavior_pi"),
        pass_mod=_load_local_module("front_node_passing_papers_pi"),
        hands_mod=_load_local_module("front_node_hands_under_table_pi"),
        obj_mod=_load_local_module("front_node_cellphone_cheat_pi"),
        setup_io=_load_local_module("front_node_all_behavior_setup_io"),
        combined_mod=_load_local_module("front_node_all_behavior_pi"),
    )


def configure_runtime_paths(
    modules: FrontNodeRuntimeModules,
    config: FrontNodeRuntimeConfig,
) -> None:
    """Redirect evidence and setup artifacts into the structured runtime folder."""
    config.evidence_root.mkdir(parents=True, exist_ok=True)
    config.setup_profile_dir.mkdir(parents=True, exist_ok=True)

    modules.combined_mod.EVIDENCE_DIR = config.evidence_root
    modules.combined_mod.HEAD_EVIDENCE_DIR = config.evidence_root / "head_behavior"
    modules.combined_mod.PASSING_EVIDENCE_DIR = (
        config.evidence_root / "passing_papers"
    )
    modules.combined_mod.HANDS_EVIDENCE_DIR = config.evidence_root / "hands"
    modules.combined_mod.OBJECT_EVIDENCE_DIR = config.evidence_root / "objects"
    modules.combined_mod.NOISE_EVIDENCE_DIR = config.evidence_root / "noise"

    modules.head_mod.EVIDENCE_DIR = config.evidence_root / "head_behavior"
    modules.pass_mod.EVIDENCE_DIR = config.evidence_root / "passing_papers"
    modules.hands_mod.EVIDENCE_DIR = config.evidence_root / "hands"
    modules.obj_mod.EVIDENCE_DIR = config.evidence_root / "objects"

    modules.setup_io.SETUP_PROFILE_DIR = config.setup_profile_dir


def apply_behavior_config(
    modules: FrontNodeRuntimeModules,
    config: FrontNodeRuntimeConfig,
) -> None:
    """Apply INI-driven threshold overrides to the copied runtime modules."""
    modules.head_mod.POSE_MODEL_PATH = config.pose_model
    modules.pass_mod.POSE_MODEL_PATH = config.pose_model
    modules.hands_mod.POSE_MODEL_PATH = config.pose_model
    modules.hands_mod.HAND_MODEL_PATH = config.hand_model
    modules.obj_mod.POSE_MODEL_PATH = config.pose_model
    modules.obj_mod.OBJ_MODEL_PATH = config.object_model
    modules.combined_mod.POSE_MODEL_PATH = config.pose_model
    modules.combined_mod.HAND_MODEL_PATH = config.hand_model
    modules.combined_mod.OBJECT_MODEL_PATH = config.object_model

    modules.combined_mod.EVIDENCE_PRE_EVENT_FRAMES = config.evidence.pre_event_frames
    modules.combined_mod.EVIDENCE_POST_EVENT_FRAMES = config.evidence.post_event_frames

    modules.head_mod.HEAD_TILT_ANGLE_DEG = config.head_behavior.head_tilt_angle_deg
    modules.head_mod.HEAD_TURN_RATIO = config.head_behavior.head_turn_ratio
    modules.head_mod.SHOULDER_TURN_ANGLE_DEG = (
        config.head_behavior.shoulder_turn_angle_deg
    )
    modules.head_mod.SUSTAINED_SEC = config.head_behavior.sustained_sec
    modules.head_mod.EVENT_COOLDOWN_SEC = config.head_behavior.event_cooldown_sec
    modules.head_mod.KP_CONF_THRESH = config.head_behavior.keypoint_confidence

    modules.pass_mod.EVENT_COOLDOWN_SEC = (
        config.passing_papers.event_cooldown_sec
    )
    modules.pass_mod.KP_CONF_THRESH = config.passing_papers.keypoint_confidence
    modules.pass_mod.ROW_TOLERANCE_PX = config.passing_papers.row_tolerance_px
    modules.pass_mod.REFERENCE_BBOX_HEIGHT = (
        config.passing_papers.reference_bbox_height
    )
    modules.pass_mod.WRIST_PROXIMITY_PX = config.passing_papers.wrist_proximity_px
    modules.pass_mod.MIN_INTERACTION_SEC = config.passing_papers.min_interaction_sec

    modules.hands_mod.HAND_CONFIDENCE = config.hands_under_table.hand_confidence
    modules.hands_mod.PERSON_CONFIDENCE = config.hands_under_table.person_confidence
    modules.hands_mod.HANDS_MISSING_SUSTAIN_SEC = (
        config.hands_under_table.hands_missing_sustain_sec
    )
    modules.hands_mod.EVENT_COOLDOWN_SEC = (
        config.hands_under_table.event_cooldown_sec
    )
    modules.hands_mod.MIN_VISIBLE_HANDS = (
        config.hands_under_table.min_visible_hands
    )
    modules.hands_mod.HAND_ASSOC_MARGIN_PX = (
        config.hands_under_table.hand_assoc_margin_px
    )
    modules.hands_mod.SMOOTH_WINDOW_FRAMES = (
        config.hands_under_table.smooth_window_frames
    )
    modules.hands_mod.SMOOTH_MISSING_RATIO = (
        config.hands_under_table.smooth_missing_ratio
    )
    modules.hands_mod.STUDENT_ABSENT_RESET_SEC = (
        config.hands_under_table.student_absent_reset_sec
    )
    modules.hands_mod.TABLE_EDGE_NEAR_PX = config.hands_under_table.table_edge_near_px
    modules.hands_mod.EDGE_DISAPPEAR_ARM_SEC = (
        config.hands_under_table.edge_disappear_arm_sec
    )

    modules.obj_mod.PERSON_CONFIDENCE = config.object_detection.person_confidence
    modules.obj_mod.EVENT_COOLDOWN_SEC = config.object_detection.event_cooldown_sec
    modules.obj_mod.ASSOC_IOU_THRESH = config.object_detection.assoc_iou_thresh
    modules.obj_mod.CONFIDENCE_THRESHOLDS["phone"] = (
        config.object_detection.phone_confidence
    )
    modules.obj_mod.CONFIDENCE_THRESHOLDS["cheat_sheet"] = (
        config.object_detection.cheat_sheet_confidence
    )


def require_detection_environment(modules: FrontNodeRuntimeModules) -> None:
    """Validate the runtime dependencies for full front-node detection."""
    if (
        not modules.head_mod.HAILO_AVAILABLE
        or not modules.hands_mod.HAILO_AVAILABLE
        or not modules.obj_mod.HAILO_AVAILABLE
    ):
        raise RuntimeError(
            "hailo_platform is required. Install: sudo apt install hailo-all"
        )

    if not modules.combined_mod.FLASK_AVAILABLE:
        raise RuntimeError(
            "Flask is required for web streaming. Install: pip install flask"
        )


def require_setup_environment(modules: FrontNodeRuntimeModules) -> None:
    """Validate the runtime dependencies for setup-profile creation."""
    if not modules.head_mod.HAILO_AVAILABLE:
        raise RuntimeError(
            "hailo_platform is required. Install: sudo apt install hailo-all"
        )


def resolve_video_path(
    requested_video: Path | None,
    config: FrontNodeRuntimeConfig,
    pass_mod,
    head_mod,
) -> Path | None:
    """Return the active video path or open the original file dialog flow."""
    if requested_video is not None:
        video_path = requested_video
    elif config.default_video is not None:
        video_path = config.default_video
    else:
        head_mod.log_info("Opening file dialog...")
        selected = pass_mod.select_video_dialog()
        if not selected:
            return None
        video_path = Path(selected).expanduser()
        if not video_path.is_absolute():
            video_path = (Path.cwd() / video_path).resolve(strict=False)

    return video_path.resolve(strict=False)


def get_webcam_source_label(config: FrontNodeRuntimeConfig) -> str:
    """Return the display/profile label for the configured webcam source."""
    name = config.webcam_source.camera_name.strip()
    if name:
        return name
    return f"webcam_{config.webcam_source.camera_index}"


def get_webcam_device_path(camera_index: int) -> Path:
    """Return the Linux device path for a webcam index."""
    return Path(f"/dev/video{camera_index}")


def _iter_webcam_open_sources(camera_index: int):
    """Yield preferred capture sources for the configured camera."""
    device_path = get_webcam_device_path(camera_index)
    if device_path.exists():
        yield str(device_path), str(device_path)
    yield camera_index, f"index {camera_index}"


def _build_webcam_backend_attempts():
    """Return capture backends that make sense for the current platform."""
    backend_attempts = []
    seen = set()

    def add_backend(backend_label: str, backend_id, attempt_defs, *,
                    allow_default: bool = False) -> None:
        if backend_id is None and not allow_default:
            return
        for use_mjpg, force_fps, config_name in attempt_defs:
            key = (backend_label, backend_id, use_mjpg, force_fps, config_name)
            if key in seen:
                continue
            seen.add(key)
            backend_attempts.append(
                (backend_label, backend_id, use_mjpg, force_fps, config_name)
            )

    if sys.platform.startswith("linux"):
        add_backend(
            "V4L2",
            getattr(cv2, "CAP_V4L2", None),
            (
                (True, False, "MJPG, driver FPS"),
                (True, True, "MJPG, forced FPS"),
                (False, False, "native, driver FPS"),
            ),
        )
    elif sys.platform.startswith("win"):
        add_backend(
            "DirectShow",
            getattr(cv2, "CAP_DSHOW", None),
            (
                (False, False, "native, driver FPS"),
                (True, False, "MJPG, driver FPS"),
                (True, True, "MJPG, forced FPS"),
            ),
        )
        add_backend(
            "Media Foundation",
            getattr(cv2, "CAP_MSMF", None),
            (
                (False, False, "native, driver FPS"),
                (True, False, "MJPG, driver FPS"),
            ),
        )
    elif sys.platform == "darwin":
        add_backend(
            "AVFoundation",
            getattr(cv2, "CAP_AVFOUNDATION", None),
            (
                (False, False, "native, driver FPS"),
            ),
        )

    add_backend(
        "default",
        None,
        (
            (True, False, "MJPG, driver FPS"),
            (True, True, "MJPG, forced FPS"),
            (False, False, "native, driver FPS"),
        ),
        allow_default=True,
    )

    return backend_attempts


def _build_webcam_capture_profiles(webcam_cfg):
    """Return capture sizes to try, from preferred to safest fallback."""
    profiles = []
    seen_sizes = set()

    def add_profile(width: int, height: int, label: str) -> None:
        size = (int(width), int(height))
        if size in seen_sizes:
            return
        seen_sizes.add(size)
        profiles.append((size[0], size[1], label))

    if webcam_cfg.capture_width > 0 and webcam_cfg.capture_height > 0:
        add_profile(
            webcam_cfg.capture_width,
            webcam_cfg.capture_height,
            "configured",
        )

    for width, height, label in WEBCAM_FALLBACK_CAPTURE_PROFILES:
        add_profile(width, height, label)

    return profiles


def _prime_webcam_device(camera_index: int, capture_width: int,
                         capture_height: int, capture_fps: float,
                         use_mjpg: bool, force_fps: bool) -> None:
    """Best-effort V4L2 priming to reduce OpenCV negotiation stalls on Pi."""
    device_path = get_webcam_device_path(camera_index)
    v4l2_ctl = shutil.which("v4l2-ctl")
    if not device_path.exists() or not v4l2_ctl:
        return

    pixel_format = "MJPG" if use_mjpg else "YUYV"
    commands = []
    if capture_width > 0 and capture_height > 0:
        commands.append([
            v4l2_ctl,
            "-d",
            str(device_path),
            (
                f"--set-fmt-video=width={capture_width},"
                f"height={capture_height},pixelformat={pixel_format}"
            ),
        ])
    if force_fps and capture_fps > 0:
        commands.append([
            v4l2_ctl,
            "-d",
            str(device_path),
            f"--set-parm={int(round(capture_fps))}",
        ])

    for command in commands:
        try:
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except Exception:
            return


def _decode_fourcc(raw_value) -> str:
    """Return a printable FOURCC string from an OpenCV numeric property."""
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return "unknown"

    chars = [
        chr((value >> shift) & 0xFF)
        for shift in (0, 8, 16, 24)
    ]
    if any(ord(char) < 32 or ord(char) > 126 for char in chars):
        return "unknown"

    return "".join(chars)


def describe_webcam_capture(cap) -> str:
    """Return a compact description of the active webcam mode."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fourcc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

    parts = [f"{width}x{height}"]
    if fps > 0:
        parts.append(f"{fps:.1f} FPS")
    if fourcc and fourcc != "unknown":
        parts.append(f"FOURCC={fourcc}")
    return " | ".join(parts)


def _is_acceptable_webcam_fps(actual_fps: float, target_fps: float) -> bool:
    """Return True when the negotiated webcam FPS is usable for live runtime."""
    if actual_fps <= 0 or actual_fps > 120:
        return True

    min_target_fps = WEBCAM_MIN_ACCEPTABLE_FPS
    if target_fps > 0:
        min_target_fps = max(
            WEBCAM_MIN_ACCEPTABLE_FPS,
            target_fps * WEBCAM_MIN_ACCEPTABLE_FPS_RATIO,
        )

    return actual_fps >= min_target_fps


def _configure_webcam_capture(cap, capture_width: int, capture_height: int,
                              capture_fps: float, use_mjpg: bool,
                              force_fps: bool) -> None:
    """Apply webcam capture settings for live Pi usage."""
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, WEBCAM_OPEN_TIMEOUT_MS)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, WEBCAM_READ_TIMEOUT_MS)
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if capture_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    if capture_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    if force_fps and capture_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, capture_fps)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, WEBCAM_BUFFER_SIZE)


def read_webcam_frame(cap, attempts: int = WEBCAM_STARTUP_READ_ATTEMPTS,
                      pause_sec: float = WEBCAM_STARTUP_READ_PAUSE_SEC):
    """Read a non-empty webcam frame with brief retries for startup jitter."""
    for attempt in range(max(1, attempts)):
        ret, frame = cap.read()
        if ret and frame is not None and getattr(frame, "size", 0) > 0:
            return frame
        if pause_sec > 0 and attempt + 1 < attempts:
            time.sleep(pause_sec)
    return None


def _open_single_webcam_capture(source_ref, camera_index: int, backend_id,
                                capture_width: int, capture_height: int,
                                capture_fps: float, use_mjpg: bool,
                                force_fps: bool):
    """Open one webcam/backend/size combination."""
    _prime_webcam_device(
        camera_index,
        capture_width,
        capture_height,
        capture_fps,
        use_mjpg=use_mjpg,
        force_fps=force_fps,
    )

    if backend_id is None:
        cap = cv2.VideoCapture(source_ref)
    else:
        cap = cv2.VideoCapture(source_ref, backend_id)

    if not cap.isOpened():
        cap.release()
        return None

    _configure_webcam_capture(
        cap,
        capture_width,
        capture_height,
        capture_fps,
        use_mjpg=use_mjpg,
        force_fps=force_fps,
    )
    if WEBCAM_POST_CONFIG_SETTLE_SEC > 0:
        time.sleep(WEBCAM_POST_CONFIG_SETTLE_SEC)

    return cap


def _warmup_webcam_capture(cap, warmup_frames: int, head_mod, source_name: str,
                           backend_name: str, config_name: str,
                           profile_name: str) -> bool:
    """Wait for a few successful startup frames instead of rejecting on one miss."""
    requested_frames = max(0, int(warmup_frames))
    required_successes = max(
        1,
        min(
            max(1, requested_frames),
            WEBCAM_WARMUP_REQUIRED_SUCCESS_FRAMES,
        ),
    )
    required_streak = min(
        required_successes,
        WEBCAM_WARMUP_REQUIRED_SUCCESS_STREAK,
    )
    max_attempts = max(requested_frames, required_successes) + WEBCAM_WARMUP_EXTRA_ATTEMPTS

    head_mod.log_info(
        f"Warming up webcam with up to {requested_frames} frame(s); requiring "
        f"{required_successes} successful read(s) with a {required_streak}-frame "
        f"stable streak."
    )

    success_count = 0
    max_consecutive_successes = 0
    consecutive_successes = 0

    for _ in range(max_attempts):
        frame = read_webcam_frame(
            cap,
            attempts=WEBCAM_WARMUP_READ_ATTEMPTS,
            pause_sec=WEBCAM_STARTUP_READ_PAUSE_SEC,
        )
        if frame is None:
            consecutive_successes = 0
            continue

        success_count += 1
        consecutive_successes += 1
        max_consecutive_successes = max(
            max_consecutive_successes,
            consecutive_successes,
        )

        if (
            success_count >= required_successes
            and consecutive_successes >= required_streak
        ):
            return True

    head_mod.log_info(
        f"Warmup did not stabilize on {source_name} via {backend_name} "
        f"({config_name}, {profile_name}); successful reads={success_count}, "
        f"best streak={max_consecutive_successes}, max attempts={max_attempts}."
    )
    return False


def open_webcam_capture(config: FrontNodeRuntimeConfig, head_mod):
    """Open the configured webcam and apply optional capture settings."""
    webcam_cfg = config.webcam_source
    capture_profiles = _build_webcam_capture_profiles(webcam_cfg)
    backend_attempts = _build_webcam_backend_attempts()

    warmup_frames = max(0, webcam_cfg.warmup_frames)
    tried_configs: list[str] = []

    for capture_width, capture_height, profile_name in capture_profiles:
        for source_ref, source_name in _iter_webcam_open_sources(
            webcam_cfg.camera_index
        ):
            for backend_name, backend_id, use_mjpg, force_fps, config_name in backend_attempts:
                cap = _open_single_webcam_capture(
                    source_ref,
                    webcam_cfg.camera_index,
                    backend_id,
                    capture_width,
                    capture_height,
                    webcam_cfg.capture_fps,
                    use_mjpg=use_mjpg,
                    force_fps=force_fps,
                )
                if cap is None:
                    tried_configs.append(
                        f"{source_name} | {profile_name} | {backend_name} ({config_name})"
                    )
                    continue

                head_mod.log_info(
                    f"Opened webcam {source_name} using "
                    f"{backend_name} backend ({config_name}, {profile_name}) -> "
                    f"{describe_webcam_capture(cap)}."
                )

                actual_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                if not _is_acceptable_webcam_fps(
                    actual_fps,
                    webcam_cfg.capture_fps,
                ):
                    head_mod.log_info(
                        f"Rejecting webcam mode with low negotiated FPS "
                        f"({actual_fps:.1f}); target is "
                        f"{webcam_cfg.capture_fps:.1f}."
                    )
                    cap.release()
                    tried_configs.append(
                        f"{source_name} | {profile_name} | {backend_name} "
                        f"({config_name}, low FPS={actual_fps:.1f})"
                    )
                    continue

                if _warmup_webcam_capture(
                    cap,
                    warmup_frames,
                    head_mod,
                    source_name,
                    backend_name,
                    config_name,
                    profile_name,
                ):
                    return cap

                cap.release()
                tried_configs.append(
                    f"{source_name} | {profile_name} | {backend_name} ({config_name})"
                )

    attempts_text = ", ".join(tried_configs) if tried_configs else "none"
    raise RuntimeError(
        f"Cannot open webcam index {webcam_cfg.camera_index} with a stable frame. "
        f"Tried: {attempts_text}."
    )
