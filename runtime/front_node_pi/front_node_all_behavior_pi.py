#!/usr/bin/env python3
"""
All Behavior Detection - Raspberry Pi + Hailo AI HAT
====================================================
Combined Pi runtime program for the full front-node behavior suite currently
implemented in this repository.

Detections:
  - Head tilt
  - Shoulder turn
  - Passing papers
  - Hands under table
  - Phone
  - Cheat sheet

Workflow:
  1. File dialog opens to select a video
  2. ROI calibration: draw an optional tracking boundary
  3. First frame shown with detected persons - click to assign student numbers
  4. Table-edge calibration: draw one 2-point line per assigned student
  5. Web stream starts at http://<pi-ip>:8080 with live annotations
  6. Console alerts + buffered evidence bursts saved to ./evidence_combined/

Notes:
  - Reuses the stable Pi helpers from the existing behavior-specific scripts.
  - Runs one pose pass per frame for head / passing analysis, plus dedicated
    hand and object passes for hands-under-table and phone / cheat-sheet.
"""

import json
import os
import re
import sys
import time
import socket
import threading
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import front_node_head_behavior_pi as head_mod
import front_node_passing_papers_pi as pass_mod
import front_node_hands_under_table_pi as hands_mod
import front_node_cellphone_cheat_pi as obj_mod
import front_node_all_behavior_setup_io as setup_io
from runtime_config import resolve_cli_path

# ── Paths ────────────────────────────────────────────────────
POSE_MODEL_PATH = head_mod.POSE_MODEL_PATH
HAND_MODEL_PATH = hands_mod.HAND_MODEL_PATH
OBJECT_MODEL_PATH = obj_mod.OBJ_MODEL_PATH

EVIDENCE_DIR = SCRIPT_DIR / "data" / "evidence_combined"
HEAD_EVIDENCE_DIR = EVIDENCE_DIR / "head_behavior"
PASSING_EVIDENCE_DIR = EVIDENCE_DIR / "passing_papers"
HANDS_EVIDENCE_DIR = EVIDENCE_DIR / "hands"
OBJECT_EVIDENCE_DIR = EVIDENCE_DIR / "objects"
EVIDENCE_PRE_EVENT_FRAMES = 10
EVIDENCE_POST_EVENT_FRAMES = 10

# ── Shared globals for Flask streaming ───────────────────────
_latest_frame = None
_latest_stream_jpeg = None
_latest_stream_seq = 0
_frame_lock = threading.Lock()
_dashboard_lock = threading.Lock()

try:
    from flask import (
        Flask,
        Response,
        abort,
        jsonify,
        redirect,
        render_template,
        request,
        send_file,
        session,
        url_for,
    )
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from werkzeug.security import check_password_hash
except ImportError:  # pragma: no cover - optional dependency surface
    check_password_hash = None


TEMPLATE_DIR = SCRIPT_DIR / "web" / "templates"
STATIC_DIR = SCRIPT_DIR / "web" / "static"
EVENTS_DIRNAME = "events"
SESSION_UPLOAD_DIR = SCRIPT_DIR / "data" / "session_uploads"
RECENT_INCIDENT_LIMIT = 40
HISTORY_INCIDENT_LIMIT = 24
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
STREAM_JPEG_QUALITY = 68
STREAM_MAX_WIDTH = 960
STREAM_MAX_FPS = 12.0
DEFAULT_EVIDENCE_REVIEW_STATUS = "unverified"
EVIDENCE_REVIEW_STATUS_CHOICES = {
    DEFAULT_EVIDENCE_REVIEW_STATUS,
    "verified",
    "false_detection",
}

_start_monitoring_callback = None
_dashboard_require_session_setup = False
_dashboard_auth = {
    "username": "admin",
    "password": "admin123",
    "secret_key": "change-this-secret-key",
    "session_ttl_minutes": 480,
}
_dashboard_state = {
    "runtime_mode": "webcam",
    "source_label": "front_webcam",
    "config_path": "",
    "evidence_root": str(EVIDENCE_DIR),
    "setup_profile_path": "",
    "status": "idle",
    "status_message": "Waiting for session setup.",
    "monitoring_active": False,
    "system_state": "idle",
    "session_details": {},
    "metrics": {},
    "recent_incident_ids": deque(),
    "incident_index": {},
    "saved_incident_ids": [],
    "saved_index": {},
    "popup_incident_id": None,
    "last_update_iso": "",
    "last_alert_at_iso": "",
    "current_error": "",
    "session_form_defaults": {},
}


def _dashboard_now() -> datetime:
    return datetime.now()


def _dashboard_now_iso() -> str:
    return _dashboard_now().isoformat(timespec="seconds")


def _format_clock(dt: datetime | None = None) -> str:
    value = dt or _dashboard_now()
    return value.strftime("%I:%M %p").lstrip("0")


def _new_dashboard_metrics() -> dict:
    return {
        "total_incidents": 0,
        "head_alerts": 0,
        "passing_alerts": 0,
        "hand_alerts": 0,
        "hand_warnings": 0,
        "object_alerts": 0,
        "tracked_students": 0,
        "assigned_students": 0,
        "processing_fps": 0.0,
        "source_fps": 0.0,
        "inference_ms": 0.0,
        "frame_idx": 0,
        "total_frames": 0,
        "elapsed_text": "00:00:00",
        "object_confidence_avg": 0.0,
        "hand_detections": 0,
        "object_detections": 0,
        "last_incident_type": "No incidents yet",
        "last_incident_time": "",
    }


def _default_session_details() -> dict:
    now = _dashboard_now()
    return {
        "subject_code": "",
        "professor": "",
        "session_date": now.strftime("%Y-%m-%d"),
        "start_time": now.strftime("%H:%M"),
        "end_time": (now + timedelta(hours=2)).strftime("%H:%M"),
        "video_path": "",
        "setup_profile_override": "",
    }


def _update_dashboard_timestamp_locked() -> None:
    _dashboard_state["last_update_iso"] = _dashboard_now_iso()


def _merge_session_details(raw_details: dict | None) -> dict:
    details = _default_session_details()
    if raw_details:
        details.update(
            {
                key: str(value).strip()
                for key, value in raw_details.items()
                if key in details and value is not None
            }
        )
    details["session_label"] = " / ".join(
        bit for bit in (details["subject_code"], details["professor"]) if bit
    ) or "Live monitoring session"
    details["schedule_label"] = (
        f"{details['session_date']} | {details['start_time']} - {details['end_time']}"
    )
    return details


def _slugify(raw_text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", raw_text.strip().lower())
    return slug.strip("-") or "incident"


def _normalize_evidence_review_status(raw_status: str | None) -> str:
    value = str(raw_status or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in EVIDENCE_REVIEW_STATUS_CHOICES:
        return value
    return DEFAULT_EVIDENCE_REVIEW_STATUS


def _apply_review_defaults(incident: dict) -> dict:
    incident["review_status"] = _normalize_evidence_review_status(
        incident.get("review_status")
    )
    incident["reviewed_at"] = str(incident.get("reviewed_at") or "")
    incident["reviewed_by"] = str(incident.get("reviewed_by") or "")
    return incident


def _incident_public_copy(incident: dict) -> dict:
    item = _apply_review_defaults(dict(incident))
    for key in ("poster_relpath", "gif_relpath", "manifest_relpath"):
        relpath = item.get(key)
        if relpath:
            public_key = key.replace("_relpath", "_url")
            item[public_key] = url_for("evidence_file", relative_path=relpath)
    return item


def _sorted_saved_incident_ids(index: dict) -> list[str]:
    return sorted(
        index,
        key=lambda incident_id: index[incident_id].get("created_at", ""),
        reverse=True,
    )


def _upsert_recent_incident_locked(incident: dict) -> dict:
    incident_id = incident["id"]
    existing = _dashboard_state["incident_index"].get(incident_id)
    if existing is None:
        existing = {}
        _dashboard_state["incident_index"][incident_id] = existing
        _dashboard_state["recent_incident_ids"].appendleft(incident_id)
    existing.update(incident)

    while len(_dashboard_state["recent_incident_ids"]) > RECENT_INCIDENT_LIMIT:
        removed_id = _dashboard_state["recent_incident_ids"].pop()
        _dashboard_state["incident_index"].pop(removed_id, None)

    _dashboard_state["popup_incident_id"] = incident_id
    _dashboard_state["last_alert_at_iso"] = incident.get("created_at", _dashboard_now_iso())
    _update_dashboard_timestamp_locked()
    return existing


def _load_manifest(manifest_path: Path) -> dict | None:
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def refresh_saved_incidents() -> None:
    manifests = []
    for base_dir in (
        HEAD_EVIDENCE_DIR,
        PASSING_EVIDENCE_DIR,
        HANDS_EVIDENCE_DIR,
        OBJECT_EVIDENCE_DIR,
    ):
        events_dir = base_dir / EVENTS_DIRNAME
        if events_dir.exists():
            manifests.extend(events_dir.glob("*/manifest.json"))

    new_index = {}
    for manifest_path in manifests:
        payload = _load_manifest(manifest_path)
        if not payload or "id" not in payload:
            continue
        new_index[payload["id"]] = _apply_review_defaults(payload)

    with _dashboard_lock:
        _dashboard_state["saved_index"] = new_index
        _dashboard_state["saved_incident_ids"] = _sorted_saved_incident_ids(new_index)
        _update_dashboard_timestamp_locked()


def configure_web_dashboard(
    *,
    auth_config,
    runtime_mode: str,
    source_label: str,
    config_path: Path,
    evidence_root: Path,
    setup_profile_path: Path | None = None,
    start_monitoring_callback=None,
    require_session_setup: bool = False,
    session_form_defaults: dict | None = None,
) -> None:
    global _start_monitoring_callback, _dashboard_require_session_setup

    _start_monitoring_callback = start_monitoring_callback
    _dashboard_require_session_setup = require_session_setup
    refresh_saved_incidents()

    with _dashboard_lock:
        _dashboard_auth.update(
            {
                "username": str(auth_config.username).strip() or "admin",
                "password": str(auth_config.password),
                "secret_key": str(auth_config.secret_key).strip()
                or "change-this-secret-key",
                "session_ttl_minutes": max(1, int(auth_config.session_ttl_minutes)),
            }
        )
        _dashboard_state["runtime_mode"] = runtime_mode
        _dashboard_state["source_label"] = source_label
        _dashboard_state["config_path"] = str(config_path)
        _dashboard_state["evidence_root"] = str(evidence_root)
        _dashboard_state["setup_profile_path"] = (
            str(setup_profile_path) if setup_profile_path else ""
        )
        _dashboard_state["status"] = "idle"
        _dashboard_state["status_message"] = (
            "Waiting for session setup."
            if require_session_setup
            else "Dashboard ready."
        )
        _dashboard_state["monitoring_active"] = False
        _dashboard_state["system_state"] = "idle"
        _dashboard_state["session_details"] = {}
        _dashboard_state["metrics"] = _new_dashboard_metrics()
        _dashboard_state["recent_incident_ids"].clear()
        _dashboard_state["incident_index"].clear()
        _dashboard_state["popup_incident_id"] = None
        _dashboard_state["current_error"] = ""
        _dashboard_state["session_form_defaults"] = _merge_session_details(
            session_form_defaults or {}
        )
        _update_dashboard_timestamp_locked()

    _reset_dashboard_stream_cache()


def begin_dashboard_session(session_details: dict | None) -> None:
    with _dashboard_lock:
        _dashboard_state["session_details"] = _merge_session_details(session_details)
        _dashboard_state["status"] = "starting"
        _dashboard_state["status_message"] = "Preparing monitoring session..."
        _dashboard_state["monitoring_active"] = False
        _dashboard_state["system_state"] = "starting"
        _dashboard_state["metrics"] = _new_dashboard_metrics()
        _dashboard_state["recent_incident_ids"].clear()
        _dashboard_state["incident_index"].clear()
        _dashboard_state["popup_incident_id"] = None
        _dashboard_state["current_error"] = ""
        _update_dashboard_timestamp_locked()

    _reset_dashboard_stream_cache()


def set_dashboard_status(
    status: str,
    message: str,
    *,
    monitoring_active: bool | None = None,
    system_state: str | None = None,
    error_message: str = "",
) -> None:
    with _dashboard_lock:
        _dashboard_state["status"] = status
        _dashboard_state["status_message"] = message
        if monitoring_active is not None:
            _dashboard_state["monitoring_active"] = monitoring_active
        if system_state is not None:
            _dashboard_state["system_state"] = system_state
        _dashboard_state["current_error"] = error_message
        _update_dashboard_timestamp_locked()


def update_dashboard_context(**kwargs) -> None:
    with _dashboard_lock:
        for key, value in kwargs.items():
            if key not in _dashboard_state:
                continue
            if isinstance(value, Path):
                _dashboard_state[key] = str(value)
            else:
                _dashboard_state[key] = value
        _update_dashboard_timestamp_locked()


def update_dashboard_metrics(**metrics) -> None:
    with _dashboard_lock:
        current = _dashboard_state.setdefault("metrics", _new_dashboard_metrics())
        current.update(metrics)
        if current.get("total_incidents", 0) > 0:
            _dashboard_state["system_state"] = "alert"
        elif _dashboard_state.get("monitoring_active"):
            _dashboard_state["system_state"] = "active"
        _update_dashboard_timestamp_locked()


def record_dashboard_incident(incident: dict) -> None:
    with _dashboard_lock:
        stored = _upsert_recent_incident_locked(incident)
        metrics = _dashboard_state.setdefault("metrics", _new_dashboard_metrics())
        metrics["last_incident_type"] = stored.get("type_label", "Incident")
        metrics["last_incident_time"] = stored.get("display_time", "")
        _dashboard_state["system_state"] = "alert"


def update_dashboard_incident(incident_id: str, **updates) -> None:
    with _dashboard_lock:
        incident = _dashboard_state["incident_index"].get(incident_id)
        if incident is not None:
            incident.update(updates)
        saved = _dashboard_state["saved_index"].get(incident_id)
        if saved is not None:
            saved.update(updates)
        _update_dashboard_timestamp_locked()


def dismiss_dashboard_popup(incident_id: str | None = None) -> None:
    with _dashboard_lock:
        if incident_id is None or _dashboard_state["popup_incident_id"] == incident_id:
            _dashboard_state["popup_incident_id"] = None
        _update_dashboard_timestamp_locked()


def update_saved_incident_review_status(
    incident_id: str,
    review_status: str | None,
    reviewed_by: str,
) -> dict | None:
    with _dashboard_lock:
        manifest = _dashboard_state["saved_index"].get(incident_id)
        manifest_relpath = str(manifest.get("manifest_relpath", "")) if manifest else ""

    if not manifest_relpath:
        return None

    try:
        manifest_path = _safe_evidence_path(manifest_relpath)
    except ValueError:
        return None

    payload = _load_manifest(manifest_path)
    if not payload or payload.get("id") != incident_id:
        return None

    normalized_status = _normalize_evidence_review_status(review_status)
    payload = _apply_review_defaults(payload)
    payload["review_status"] = normalized_status
    if normalized_status == DEFAULT_EVIDENCE_REVIEW_STATUS:
        payload["reviewed_at"] = ""
        payload["reviewed_by"] = ""
    else:
        payload["reviewed_at"] = _dashboard_now_iso()
        payload["reviewed_by"] = reviewed_by

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with _dashboard_lock:
        _dashboard_state["saved_index"][incident_id] = dict(payload)
        recent = _dashboard_state["incident_index"].get(incident_id)
        if recent is not None:
            recent.update(
                review_status=payload["review_status"],
                reviewed_at=payload["reviewed_at"],
                reviewed_by=payload["reviewed_by"],
            )
        _update_dashboard_timestamp_locked()

    return payload


def _password_matches(configured_password: str, submitted_password: str) -> bool:
    if configured_password.startswith(("pbkdf2:", "scrypt:", "argon2:")) and check_password_hash:
        return check_password_hash(configured_password, submitted_password)
    return configured_password == submitted_password


def _is_authenticated() -> bool:
    return (
        session.get("authenticated") is True
        and session.get("username") == _dashboard_auth["username"]
    )


def _login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not _is_authenticated():
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)

    return wrapper


def _api_login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not _is_authenticated():
            return jsonify({"error": "authentication required"}), 401
        return view_func(*args, **kwargs)

    return wrapper


def _dashboard_snapshot() -> dict:
    with _dashboard_lock:
        recent_incidents = [
            _dashboard_state["incident_index"][incident_id]
            for incident_id in _dashboard_state["recent_incident_ids"]
            if incident_id in _dashboard_state["incident_index"]
        ]
        saved_incidents = [
            _dashboard_state["saved_index"][incident_id]
            for incident_id in _dashboard_state["saved_incident_ids"][:HISTORY_INCIDENT_LIMIT]
            if incident_id in _dashboard_state["saved_index"]
        ]
        popup_incident = None
        popup_id = _dashboard_state.get("popup_incident_id")
        if popup_id:
            popup_incident = _dashboard_state["incident_index"].get(popup_id)

        return {
            "runtime_mode": _dashboard_state["runtime_mode"],
            "source_label": _dashboard_state["source_label"],
            "config_path": _dashboard_state["config_path"],
            "evidence_root": _dashboard_state["evidence_root"],
            "setup_profile_path": _dashboard_state["setup_profile_path"],
            "status": _dashboard_state["status"],
            "status_message": _dashboard_state["status_message"],
            "monitoring_active": _dashboard_state["monitoring_active"],
            "system_state": _dashboard_state["system_state"],
            "session_details": dict(_dashboard_state["session_details"]),
            "metrics": dict(_dashboard_state.get("metrics", _new_dashboard_metrics())),
            "recent_incidents": [dict(item) for item in recent_incidents],
            "saved_incidents": [dict(item) for item in saved_incidents],
            "popup_incident": dict(popup_incident) if popup_incident else None,
            "current_error": _dashboard_state["current_error"],
            "last_update_iso": _dashboard_state["last_update_iso"],
            "requires_session_setup": _dashboard_require_session_setup,
            "session_form_defaults": dict(_dashboard_state["session_form_defaults"]),
        }


def _public_dashboard_snapshot() -> dict:
    snapshot = _dashboard_snapshot()
    snapshot["recent_incidents"] = [
        _incident_public_copy(item) for item in snapshot["recent_incidents"]
    ]
    snapshot["saved_incidents"] = [
        _incident_public_copy(item) for item in snapshot["saved_incidents"]
    ]
    if snapshot["popup_incident"] is not None:
        snapshot["popup_incident"] = _incident_public_copy(snapshot["popup_incident"])
    return snapshot


def _safe_evidence_path(relative_path: str) -> Path:
    candidate = (EVIDENCE_DIR / relative_path).resolve(strict=False)
    evidence_root = EVIDENCE_DIR.resolve(strict=False)
    if evidence_root not in candidate.parents and candidate != evidence_root:
        raise ValueError("Requested evidence path is outside the evidence root.")
    return candidate


def _reset_dashboard_stream_cache() -> None:
    global _latest_frame, _latest_stream_jpeg, _latest_stream_seq
    with _frame_lock:
        _latest_frame = None
        _latest_stream_jpeg = None
        _latest_stream_seq = 0


def _encode_dashboard_frame(frame) -> bytes | None:
    if frame is None:
        return None

    height, width = frame.shape[:2]
    stream_frame = frame
    if width > STREAM_MAX_WIDTH:
        scale = STREAM_MAX_WIDTH / float(width)
        resized_height = max(1, int(round(height * scale)))
        stream_frame = cv2.resize(
            frame,
            (STREAM_MAX_WIDTH, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    success, jpeg = cv2.imencode(
        ".jpg",
        stream_frame,
        [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY],
    )
    if not success:
        return None
    return jpeg.tobytes()


def _publish_dashboard_frame(frame) -> bool:
    global _latest_frame, _latest_stream_jpeg, _latest_stream_seq

    jpeg_bytes = _encode_dashboard_frame(frame)
    if jpeg_bytes is None:
        return False

    with _frame_lock:
        _latest_frame = frame
        _latest_stream_jpeg = jpeg_bytes
        _latest_stream_seq += 1

    return True


def _resolve_runtime_video_path(raw_value: str | None) -> Path:
    video_path = resolve_cli_path(raw_value)
    if video_path is None:
        raise ValueError("Enter a video path before starting video monitoring.")

    suffix = video_path.suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("Unsupported video type. Use MP4, AVI, MOV, MKV, M4V, or WEBM.")

    if not video_path.is_file():
        raise ValueError(f"Video file not found on the runtime device: {video_path}")

    return video_path


def _open_runtime_video_dialog() -> Path | None:
    head_mod.log_info("Opening runtime video file dialog...")
    selected_path = pass_mod.select_video_dialog()
    if not selected_path:
        return None
    return _resolve_runtime_video_path(selected_path)


def _save_uploaded_calibration(upload_storage) -> Path:
    filename = (upload_storage.filename or "").strip()
    if not filename:
        raise ValueError("Select a calibration JSON file.")

    suffix = Path(filename).suffix.lower()
    if suffix != ".json":
        raise ValueError("Calibration file must be a .json setup profile.")

    safe_stem = _slugify(Path(filename).stem)
    target_name = f"{_dashboard_now().strftime('%Y%m%d%H%M%S%f')}_{safe_stem}{suffix}"
    SESSION_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = (SESSION_UPLOAD_DIR / target_name).resolve(strict=False)
    upload_storage.save(target_path)
    return target_path


COLOR_PRIORITY = {
    head_mod.COL_NORMAL: 0,
    hands_mod.COL_STUDENT: 0,
    head_mod.COL_HEAD_TILT: 1,
    head_mod.COL_SHOULDER_TURN: 1,
    pass_mod.COL_WARNING: 1,
    hands_mod.COL_WARNING: 1,
    head_mod.COL_FLAGGED: 2,
    hands_mod.COL_ALERT: 2,
}


def elevate_color(current, new_color):
    """Keep the highest-severity color already assigned to a student box."""
    if COLOR_PRIORITY.get(new_color, 0) >= COLOR_PRIORITY.get(current, 0):
        return new_color
    return current


class SharedHailoPoseEstimator(head_mod.HailoPoseEstimator):
    """Pose estimator with persistent infer pipeline and optional shared VDevice."""

    def __init__(self, hef_path, conf_threshold=0.25, kpt_threshold=0.3,
                 iou_threshold=0.45, vdevice=None):
        if not head_mod.HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform is not installed.")

        self.conf_threshold = conf_threshold
        self.kpt_threshold = kpt_threshold
        self.iou_threshold = iou_threshold
        self._infer_ctx = None
        self._infer_pipeline = None

        head_mod.log_info(f"Loading HEF model: {hef_path}")
        self.hef = head_mod.HEF(str(hef_path))

        self.vdevice = vdevice or head_mod.VDevice()
        configure_params = head_mod.ConfigureParams.create_from_hef(
            self.hef, interface=head_mod.HailoStreamInterface.PCIe
        )
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        self.input_vstreams_params = head_mod.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=head_mod.FormatType.UINT8
        )
        self.output_vstreams_params = head_mod.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=head_mod.FormatType.FLOAT32
        )

        self.input_shape = self.input_vstream_info[0].shape
        self.input_h = self.input_shape[0]
        self.input_w = self.input_shape[1]

        head_mod.log_info(f"Model input shape : {self.input_shape}")
        for out_info in self.output_vstream_info:
            head_mod.log_info(f"Model output layer: {out_info.name} -> {out_info.shape}")
        head_mod.log_info("Hailo device ready.")

    def _ensure_infer_pipeline(self):
        """Create the Hailo infer pipeline once and reuse it."""
        if self._infer_pipeline is not None:
            return

        infer_ctx = None
        try:
            infer_ctx = head_mod.InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            )
            infer_pipeline = infer_ctx.__enter__()
        except Exception as exc:
            if infer_ctx is not None:
                infer_ctx.__exit__(type(exc), exc, exc.__traceback__)
            raise

        self._infer_ctx = infer_ctx
        self._infer_pipeline = infer_pipeline

    def close(self):
        """Release persistent Hailo resources."""
        if self._infer_ctx is not None:
            self._infer_ctx.__exit__(None, None, None)
            self._infer_ctx = None
            self._infer_pipeline = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def detect_pose(self, frame):
        """Run pose estimation and return bboxes + keypoints."""
        img_h, img_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb, axis=0)
        input_dict = {self.input_vstream_info[0].name: input_data}
        self._ensure_infer_pipeline()

        with self.network_group.activate(self.network_group_params):
            results = self._infer_pipeline.infer(input_dict)

        if not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            if isinstance(results, dict):
                for name, arr in results.items():
                    head_mod.log_info(f"Output '{name}': shape={np.array(arr).shape}")

        return self._postprocess(results, img_w, img_h)


class ReacquiringLockedIoUTracker(pass_mod.IoUTracker):
    """Locked IoU tracker that can reacquire assigned students after loss."""

    REACQUIRE_CENTER_DIST_FACTOR = 0.90
    REACQUIRE_MIN_IOU = 0.05
    REACQUIRE_AREA_RATIO_MIN = 0.40
    REACQUIRE_AREA_RATIO_MAX = 2.50

    def keep_only(self, track_ids_to_keep):
        """Lock tracking to the assigned IDs but retain anchors for reacquisition."""
        super().keep_only(track_ids_to_keep)
        self._allowed_ids = set(track_ids_to_keep)
        self._anchor_boxes = {
            tid: tuple(self._tracks[tid]["bbox"])
            for tid in track_ids_to_keep
            if tid in self._tracks
        }

    @staticmethod
    def _box_center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _box_size(box):
        return (max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1]))

    @staticmethod
    def _box_area(box):
        w, h = ReacquiringLockedIoUTracker._box_size(box)
        return w * h

    def _candidate_reference_box(self, tid):
        if tid in self._tracks:
            return tuple(self._tracks[tid]["bbox"])
        return self._anchor_boxes.get(tid)

    def _can_reacquire(self, ref_box, det_box):
        iou = float(self._compute_iou_matrix(
            np.array([ref_box], dtype=np.float32),
            np.array([det_box], dtype=np.float32),
        )[0, 0])

        ref_cx, ref_cy = self._box_center(ref_box)
        det_cx, det_cy = self._box_center(det_box)
        ref_w, ref_h = self._box_size(ref_box)
        det_w, det_h = self._box_size(det_box)

        max_w = max(ref_w, det_w)
        max_h = max(ref_h, det_h)
        dx = abs(det_cx - ref_cx)
        dy = abs(det_cy - ref_cy)

        area_ratio = self._box_area(det_box) / max(1.0, self._box_area(ref_box))
        center_ok = (
            dx <= max_w * self.REACQUIRE_CENTER_DIST_FACTOR
            and dy <= max_h * self.REACQUIRE_CENTER_DIST_FACTOR
        )
        area_ok = (
            self.REACQUIRE_AREA_RATIO_MIN
            <= area_ratio
            <= self.REACQUIRE_AREA_RATIO_MAX
        )

        if not center_ok or not area_ok:
            return False, -9999.0

        center_score = 1.0 - ((dx / max_w) + (dy / max_h)) / 2.0
        score = iou * 2.0 + center_score
        if iou >= self.REACQUIRE_MIN_IOU:
            score += 0.5
        return True, score

    def update(self, detections):
        """Update active tracks and reassign locked IDs when students reappear."""
        result_ids = super().update(detections)
        locked = getattr(self, "_locked", False)

        if not locked or not detections:
            return result_ids

        allowed_ids = getattr(self, "_allowed_ids", None)
        anchor_boxes = getattr(self, "_anchor_boxes", None)
        if not allowed_ids or not anchor_boxes:
            return result_ids

        unmatched_det_indices = [idx for idx, tid in enumerate(result_ids) if tid == -1]
        if not unmatched_det_indices:
            return result_ids

        already_assigned = {tid for tid in result_ids if tid != -1}
        available_tids = [tid for tid in allowed_ids if tid not in already_assigned]
        if not available_tids:
            return result_ids

        pairs = []
        for di in unmatched_det_indices:
            det_box = tuple(detections[di]["bbox"])
            for tid in available_tids:
                ref_box = self._candidate_reference_box(tid)
                if ref_box is None:
                    continue
                ok, score = self._can_reacquire(ref_box, det_box)
                if ok:
                    pairs.append((score, di, tid))

        if not pairs:
            return result_ids

        pairs.sort(reverse=True)
        used_dets = set()
        used_tids = set()

        for _, di, tid in pairs:
            if di in used_dets or tid in used_tids:
                continue
            self._tracks[tid] = {"bbox": detections[di]["bbox"], "lost": 0}
            result_ids[di] = tid
            head_mod.log_info(f"Reacquired track ID {tid} after loss.")
            used_dets.add(di)
            used_tids.add(tid)

        return result_ids


def _fmt_evidence_ts(ts_sec):
    return head_mod.fmt_ts(ts_sec).replace(":", "").replace(".", "_")


def _evidence_frame_tag(relative_idx):
    order_idx = relative_idx + EVIDENCE_PRE_EVENT_FRAMES + 1
    if relative_idx < 0:
        phase_tag = f"pre{abs(relative_idx):02d}"
    elif relative_idx == 0:
        phase_tag = "event"
    else:
        phase_tag = f"post{relative_idx:02d}"
    return f"f{order_idx:02d}_{phase_tag}"


def _relative_evidence_path(path: Path) -> str:
    return path.resolve(strict=False).relative_to(
        EVIDENCE_DIR.resolve(strict=False)
    ).as_posix()


def _evidence_group_dir(behavior_type: str) -> Path:
    if behavior_type == "head":
        return HEAD_EVIDENCE_DIR / EVENTS_DIRNAME
    if behavior_type == "passing":
        return PASSING_EVIDENCE_DIR / EVENTS_DIRNAME
    if behavior_type == "hands":
        return HANDS_EVIDENCE_DIR / EVENTS_DIRNAME
    return OBJECT_EVIDENCE_DIR / EVENTS_DIRNAME


def _sequence_type_label(sequence) -> str:
    if sequence["behavior_type"] == "head":
        mapping = {
            "head_tilt": "Head Tilting",
            "shoulder_turn": "Shoulder Turn",
        }
        return mapping.get(sequence["behavior"], "Head Behavior")
    if sequence["behavior_type"] == "passing":
        return "Passing Paper"
    if sequence["behavior_type"] == "hands":
        return "Hands Missing"
    mapping = {
        "phone": "Using Phone",
        "cheat_sheet": "Cheat Sheet",
    }
    return mapping.get(sequence["class_name"], "Object Detection")


def _sequence_student_numbers(sequence) -> list[int]:
    if sequence["behavior_type"] == "passing":
        return [int(value) for value in sequence["student_nums"]]
    return [int(sequence["student_num"])]


def _sequence_summary(sequence) -> str:
    type_label = _sequence_type_label(sequence)
    student_numbers = _sequence_student_numbers(sequence)
    if sequence["behavior_type"] == "passing":
        return (
            f"Students #{student_numbers[0]:02d} and #{student_numbers[1]:02d} "
            f"{type_label.lower()} detected"
        )
    return f"Student #{student_numbers[0]:02d} {type_label.lower()} detected"


def _build_sequence_incident(sequence, status: str) -> dict:
    confidence = sequence.get("confidence")
    confidence_pct = int(round(confidence * 100)) if confidence is not None else None
    return {
        "id": sequence["incident_id"],
        "created_at": sequence["created_at"],
        "display_time": sequence["display_time"],
        "event_clock": sequence["event_clock"],
        "status": status,
        "severity": "alert",
        "behavior_type": sequence["behavior_type"],
        "type_label": _sequence_type_label(sequence),
        "summary": _sequence_summary(sequence),
        "student_numbers": _sequence_student_numbers(sequence),
        "camera_label": sequence["camera_label"],
        "session_details": dict(sequence["session_details"]),
        "confidence": confidence,
        "confidence_pct": confidence_pct,
        "poster_relpath": sequence.get("poster_relpath", ""),
        "gif_relpath": sequence.get("gif_relpath", ""),
        "manifest_relpath": sequence.get("manifest_relpath", ""),
        "frame_count": len(sequence.get("frame_paths", [])),
        "review_status": DEFAULT_EVIDENCE_REVIEW_STATUS,
        "reviewed_at": "",
        "reviewed_by": "",
    }


def _ensure_sequence_storage(sequence) -> None:
    if sequence.get("event_dir") is not None:
        return

    event_dir = _evidence_group_dir(sequence["behavior_type"]) / sequence["incident_id"]
    frames_dir = event_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sequence["event_dir"] = event_dir
    sequence["frames_dir"] = frames_dir
    sequence["frame_paths"] = []
    sequence["poster_relpath"] = ""
    sequence["gif_relpath"] = ""
    sequence["manifest_relpath"] = ""


def _save_grouped_evidence_frame(sequence, frame, frame_tag: str) -> None:
    _ensure_sequence_storage(sequence)
    frame_path = sequence["frames_dir"] / f"{frame_tag}.jpg"
    cv2.imwrite(str(frame_path), frame)
    sequence["frame_paths"].append(frame_path)
    if frame_tag.endswith("event") or not sequence.get("poster_relpath"):
        sequence["poster_relpath"] = _relative_evidence_path(frame_path)


def _write_sequence_gif(sequence) -> str:
    if not PIL_AVAILABLE or not sequence.get("frame_paths"):
        return ""

    frames = []
    for frame_path in sequence["frame_paths"]:
        with Image.open(frame_path) as image:
            frames.append(image.convert("P", palette=Image.ADAPTIVE))

    if not frames:
        return ""

    gif_path = sequence["event_dir"] / "evidence.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
        optimize=False,
    )
    return _relative_evidence_path(gif_path)


def _finalize_evidence_sequence(sequence) -> None:
    sequence["gif_relpath"] = _write_sequence_gif(sequence)
    manifest_path = sequence["event_dir"] / "manifest.json"
    manifest_relpath = _relative_evidence_path(manifest_path)
    sequence["manifest_relpath"] = manifest_relpath
    manifest = _build_sequence_incident(sequence, status="ready")
    manifest["manifest_relpath"] = manifest_relpath
    manifest["frame_relpaths"] = [
        _relative_evidence_path(frame_path)
        for frame_path in sequence.get("frame_paths", [])
    ]
    manifest["frame_count"] = len(manifest["frame_relpaths"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with _dashboard_lock:
        _dashboard_state["saved_index"][manifest["id"]] = manifest
        _dashboard_state["saved_incident_ids"] = _sorted_saved_incident_ids(
            _dashboard_state["saved_index"]
        )
        _update_dashboard_timestamp_locked()

    update_dashboard_incident(
        sequence["incident_id"],
        status="ready",
        poster_relpath=manifest.get("poster_relpath", ""),
        gif_relpath=manifest.get("gif_relpath", ""),
        manifest_relpath=manifest_relpath,
        frame_count=manifest["frame_count"],
    )


def save_head_evidence(frame, student_num, behavior, event_ts_sec, frame_ts_sec,
                       frame_tag):
    os.makedirs(HEAD_EVIDENCE_DIR, exist_ok=True)
    event_ts_str = _fmt_evidence_ts(event_ts_sec)
    frame_ts_str = _fmt_evidence_ts(frame_ts_sec)
    fname = (
        f"student{student_num}_{behavior}_{event_ts_str}_{frame_tag}_"
        f"{frame_ts_str}.jpg"
    )
    cv2.imwrite(str(HEAD_EVIDENCE_DIR / fname), frame)
    head_mod.log_info(f"Head evidence saved: {fname}")


def save_passing_evidence(frame, student_nums, event_ts_sec, frame_ts_sec,
                          frame_tag):
    os.makedirs(PASSING_EVIDENCE_DIR, exist_ok=True)
    event_ts_str = _fmt_evidence_ts(event_ts_sec)
    frame_ts_str = _fmt_evidence_ts(frame_ts_sec)
    nums_str = "_".join(str(n) for n in student_nums)
    fname = f"passing_s{nums_str}_{event_ts_str}_{frame_tag}_{frame_ts_str}.jpg"
    cv2.imwrite(str(PASSING_EVIDENCE_DIR / fname), frame)
    head_mod.log_info(f"Passing evidence saved: {fname}")


def save_hand_evidence(annotated_frame, raw_frame, video_name, line_idx, student_id,
                       event_ts_sec, frame_ts_sec, frame_tag):
    os.makedirs(HANDS_EVIDENCE_DIR, exist_ok=True)
    event_ts_str = _fmt_evidence_ts(event_ts_sec)
    frame_ts_str = _fmt_evidence_ts(frame_ts_sec)
    fname_ann = (
        f"{video_name}_line{line_idx + 1}_sid{student_id}_{event_ts_str}_"
        f"{frame_tag}_{frame_ts_str}_annotated.jpg"
    )
    fname_raw = (
        f"{video_name}_line{line_idx + 1}_sid{student_id}_{event_ts_str}_"
        f"{frame_tag}_{frame_ts_str}_raw.jpg"
    )
    cv2.imwrite(str(HANDS_EVIDENCE_DIR / fname_ann), annotated_frame)
    cv2.imwrite(str(HANDS_EVIDENCE_DIR / fname_raw), raw_frame)
    hands_mod.log_info(f"Hands evidence saved: {fname_ann} + raw")


def save_object_evidence(annotated_frame, raw_frame, student_num, label, conf,
                         event_ts_sec, frame_ts_sec, frame_tag):
    os.makedirs(OBJECT_EVIDENCE_DIR, exist_ok=True)
    event_ts_str = _fmt_evidence_ts(event_ts_sec)
    frame_ts_str = _fmt_evidence_ts(frame_ts_sec)
    safe_label = label.replace(" ", "_")
    conf_pct = int(round(conf * 100))
    fname_ann = (
        f"student{student_num}_{safe_label}_{conf_pct}pct_{event_ts_str}_"
        f"{frame_tag}_{frame_ts_str}_annotated.jpg"
    )
    fname_raw = (
        f"student{student_num}_{safe_label}_{conf_pct}pct_{event_ts_str}_"
        f"{frame_tag}_{frame_ts_str}_raw.jpg"
    )
    cv2.imwrite(str(OBJECT_EVIDENCE_DIR / fname_ann), annotated_frame)
    cv2.imwrite(str(OBJECT_EVIDENCE_DIR / fname_raw), raw_frame)
    head_mod.log_info(f"Object evidence saved: {fname_ann} + raw")


def get_evidence_target_students(sequence):
    """Return the student number(s) that should be highlighted in evidence."""
    if sequence["behavior_type"] == "passing":
        return list(sequence["student_nums"])
    return [sequence["student_num"]]


def get_evidence_highlight_color(sequence):
    """Pick a highlight color for evidence based on the alert type."""
    behavior_type = sequence["behavior_type"]
    if behavior_type == "hands":
        return hands_mod.COL_ALERT
    if behavior_type == "object":
        return obj_mod.CLASS_COLORS.get(
            sequence["class_name"], head_mod.COL_FLAGGED
        )
    return head_mod.COL_FLAGGED


def build_evidence_frame(sequence, snapshot):
    """Create a clean evidence frame with only the relevant student box(es)."""
    evidence_frame = snapshot["raw_frame"].copy()
    student_boxes = snapshot.get("student_boxes", {})
    highlight_color = get_evidence_highlight_color(sequence)

    for student_num in get_evidence_target_students(sequence):
        bbox = student_boxes.get(student_num)
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), highlight_color, 3)

    if sequence["behavior_type"] == "object":
        object_boxes = snapshot.get("object_boxes", {})
        target_key = (sequence["student_num"], sequence["class_name"])
        for bbox in object_boxes.get(target_key, []):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), highlight_color, 3)

    return evidence_frame


def save_evidence_sequence_frame(sequence, snapshot, frame_tag):
    """Save one evidence frame for the given alert sequence."""
    behavior_type = sequence["behavior_type"]
    annotated_frame = build_evidence_frame(sequence, snapshot)
    raw_frame = snapshot["raw_frame"]
    frame_ts_sec = snapshot["frame_ts_sec"]

    if behavior_type == "head":
        save_head_evidence(
            annotated_frame,
            sequence["student_num"],
            sequence["behavior"],
            sequence["event_ts_sec"],
            frame_ts_sec,
            frame_tag,
        )
    elif behavior_type == "passing":
        save_passing_evidence(
            annotated_frame,
            sequence["student_nums"],
            sequence["event_ts_sec"],
            frame_ts_sec,
            frame_tag,
        )
    elif behavior_type == "hands":
        save_hand_evidence(
            annotated_frame,
            raw_frame,
            sequence["video_name"],
            sequence["line_idx"],
            sequence["student_num"],
            sequence["event_ts_sec"],
            frame_ts_sec,
            frame_tag,
        )
    elif behavior_type == "object":
        save_object_evidence(
            annotated_frame,
            raw_frame,
            sequence["student_num"],
            sequence["class_name"],
            sequence["confidence"],
            sequence["event_ts_sec"],
            frame_ts_sec,
            frame_tag,
        )

    _save_grouped_evidence_frame(sequence, annotated_frame, frame_tag)


def queue_evidence_sequence(sequence_queue, recent_frames, behavior_type,
                            event_ts_sec, **payload):
    """Save buffered pre-event frames, then queue event + post-event frames."""
    with _dashboard_lock:
        camera_label = _dashboard_state["source_label"]
        session_details = dict(_dashboard_state["session_details"])

    incident_seed = payload.get("class_name") or payload.get("behavior") or behavior_type
    sequence = {
        "incident_id": (
            f"{behavior_type}-{_slugify(str(incident_seed))}-"
            f"{_dashboard_now().strftime('%Y%m%d%H%M%S%f')}"
        ),
        "behavior_type": behavior_type,
        "created_at": _dashboard_now_iso(),
        "display_time": _format_clock(),
        "event_clock": head_mod.fmt_ts(event_ts_sec),
        "event_ts_sec": event_ts_sec,
        "next_relative_idx": 0,
        "camera_label": camera_label,
        "session_details": session_details,
        "event_dir": None,
        **payload,
    }
    _ensure_sequence_storage(sequence)

    pre_count = len(recent_frames)
    for relative_idx, snapshot in zip(range(-pre_count, 0), recent_frames):
        save_evidence_sequence_frame(
            sequence,
            snapshot,
            _evidence_frame_tag(relative_idx),
        )

    sequence_queue.append(sequence)
    return sequence


def flush_evidence_sequences(sequence_queue, snapshot):
    """Save the event frame and post-event frames for every active burst."""
    active_sequences = []

    for seq in sequence_queue:
        save_evidence_sequence_frame(
            seq,
            snapshot,
            _evidence_frame_tag(seq["next_relative_idx"]),
        )

        seq["next_relative_idx"] += 1
        if seq["next_relative_idx"] <= EVIDENCE_POST_EVENT_FRAMES:
            active_sequences.append(seq)
        else:
            _finalize_evidence_sequence(seq)

    return active_sequences


def create_flask_app():
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.secret_key = _dashboard_auth["secret_key"]
    app.permanent_session_lifetime = timedelta(
        minutes=_dashboard_auth["session_ttl_minutes"]
    )
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/")
    def index():
        if not _is_authenticated():
            return redirect(url_for("login"))
        if _dashboard_require_session_setup and not _dashboard_snapshot()["session_details"]:
            return redirect(url_for("session_setup"))
        return redirect(url_for("dashboard"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error_message = ""
        next_path = request.args.get("next", "").strip()

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            next_path = request.form.get("next", "").strip()

            if (
                username == _dashboard_auth["username"]
                and _password_matches(_dashboard_auth["password"], password)
            ):
                session.clear()
                session.permanent = True
                session["authenticated"] = True
                session["username"] = username
                if next_path.startswith("/"):
                    return redirect(next_path)
                return redirect(url_for("index"))

            error_message = "Invalid credentials. Check the configured dashboard username and password."

        return render_template(
            "login.html",
            error_message=error_message,
            next_path=next_path,
        )

    @app.route("/logout")
    @_login_required
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/session-setup", methods=["GET", "POST"])
    @_login_required
    def session_setup():
        snapshot = _dashboard_snapshot()
        error_message = ""
        form_seed = snapshot["session_details"] or snapshot.get("session_form_defaults", {})
        form_values = _merge_session_details(form_seed)
        if not form_values.get("setup_profile_override") and snapshot.get("setup_profile_path"):
            form_values["setup_profile_override"] = snapshot["setup_profile_path"]

        if not _dashboard_require_session_setup and _start_monitoring_callback is None:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            if snapshot["runtime_mode"] == "video":
                head_mod.log_info("Video session setup submitted.")
            if snapshot["status"] in {"starting", "manual_setup", "running"}:
                error_message = "Monitoring is already active. Open the dashboard to follow the live session."
            else:
                submission_payload = {
                    "subject_code": request.form.get("subject_code", ""),
                    "professor": request.form.get("professor", ""),
                    "session_date": request.form.get("session_date", ""),
                    "start_time": request.form.get("start_time", ""),
                    "end_time": request.form.get("end_time", ""),
                    "video_path": request.form.get("video_path", ""),
                    "setup_profile_override": request.form.get("existing_setup_profile_override", ""),
                }

                if snapshot["runtime_mode"] == "video":
                    try:
                        runtime_video = _resolve_runtime_video_path(
                            request.form.get("video_path", "")
                        )
                    except ValueError as exc:
                        error_message = str(exc)
                    else:
                        head_mod.log_info(f"Using runtime video path: {runtime_video}")
                        submission_payload["video_path"] = str(runtime_video)

                uploaded_calibration = request.files.get("calibration_file")
                if uploaded_calibration is not None and (uploaded_calibration.filename or "").strip():
                    try:
                        head_mod.log_info(
                            f"Receiving uploaded calibration: {(uploaded_calibration.filename or '').strip()}"
                        )
                        stored_calibration = _save_uploaded_calibration(uploaded_calibration)
                    except ValueError as exc:
                        error_message = str(exc)
                    else:
                        head_mod.log_info(f"Uploaded calibration saved to: {stored_calibration}")
                        submission_payload["setup_profile_override"] = str(stored_calibration)

                submitted = _merge_session_details(submission_payload)
                form_values = submitted
                if (
                    snapshot["runtime_mode"] == "video"
                    and not submitted.get("video_path")
                    and not error_message
                ):
                    error_message = "Select a video file before starting video monitoring."

                if not error_message:
                    begin_dashboard_session(submitted)
                    if _start_monitoring_callback is not None:
                        try:
                            _start_monitoring_callback(submitted)
                        except Exception as exc:
                            error_message = str(exc)
                            set_dashboard_status(
                                "error",
                                "Monitoring could not be started.",
                                monitoring_active=False,
                                system_state="error",
                                error_message=error_message,
                            )
                        else:
                            return redirect(url_for("dashboard"))
                    else:
                        return redirect(url_for("dashboard"))

        snapshot = _dashboard_snapshot()
        setup_profile_path = Path(snapshot["setup_profile_path"]) if snapshot["setup_profile_path"] else None
        has_saved_setup = setup_profile_path is not None and setup_profile_path.exists()
        return render_template(
            "session_setup.html",
            error_message=error_message,
            form_values=form_values,
            runtime_mode=snapshot["runtime_mode"],
            source_label=snapshot["source_label"],
            status=snapshot["status"],
            status_message=snapshot["status_message"],
            has_saved_setup=has_saved_setup,
            setup_profile_path=str(setup_profile_path) if setup_profile_path else "",
            form_defaults=snapshot.get("session_form_defaults", {}),
        )

    @app.route("/dashboard")
    @_login_required
    def dashboard():
        snapshot = _public_dashboard_snapshot()
        return render_template(
            "dashboard.html",
            dashboard_json=json.dumps(snapshot),
            username=session.get("username", _dashboard_auth["username"]),
        )

    @app.route("/api/dashboard")
    @_api_login_required
    def dashboard_api():
        return jsonify(_public_dashboard_snapshot())

    @app.route("/api/session-setup/select-video", methods=["POST"])
    @_api_login_required
    def select_video_api():
        snapshot = _dashboard_snapshot()
        if snapshot["runtime_mode"] != "video":
            return jsonify({"error": "Video selection is only available in video mode."}), 400

        try:
            selected_video = _open_runtime_video_dialog()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - GUI/runtime environment dependent
            return jsonify(
                {
                    "error": (
                        "Could not open the runtime file picker. "
                        f"Check that the device has a desktop session available. Details: {exc}"
                    )
                }
            ), 500

        if selected_video is None:
            return jsonify({"ok": False, "cancelled": True})

        return jsonify({"ok": True, "video_path": str(selected_video)})

    @app.route("/api/popup/dismiss", methods=["POST"])
    @_api_login_required
    def dismiss_popup_api():
        payload = request.get_json(silent=True) or {}
        dismiss_dashboard_popup(payload.get("incident_id"))
        return jsonify({"ok": True})

    @app.route("/api/evidence/review", methods=["POST"])
    @_api_login_required
    def update_evidence_review_api():
        payload = request.get_json(silent=True) or {}
        incident_id = str(payload.get("incident_id") or "").strip()
        if not incident_id:
            return jsonify({"error": "incident_id is required"}), 400

        updated = update_saved_incident_review_status(
            incident_id,
            payload.get("review_status"),
            session.get("username", _dashboard_auth["username"]),
        )
        if updated is None:
            return jsonify({"error": "Incident not found."}), 404

        return jsonify({"ok": True, "incident": _incident_public_copy(updated)})

    @app.route("/evidence/<path:relative_path>")
    @_login_required
    def evidence_file(relative_path):
        try:
            evidence_path = _safe_evidence_path(relative_path)
        except ValueError:
            abort(404)

        if not evidence_path.exists() or not evidence_path.is_file():
            abort(404)
        return send_file(evidence_path)

    @app.route("/video_feed")
    @_login_required
    def video_feed():
        def generate():
            last_seq = -1
            while True:
                with _frame_lock:
                    jpeg_bytes = _latest_stream_jpeg
                    frame_seq = _latest_stream_seq

                if jpeg_bytes is not None and frame_seq != last_seq:
                    last_seq = frame_seq
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg_bytes
                        + b"\r\n"
                    )
                else:
                    time.sleep(0.03)

        return Response(
            generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    return app


def get_local_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "localhost"


def start_web_server(port=8080):
    app = create_flask_app()
    thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True),
        daemon=True,
    )
    thread.start()
    return thread


def draw_fps_badge(frame, fps_value, color):
    badge_text = f"FPS {fps_value:.1f}"
    (badge_w, badge_h), _ = cv2.getTextSize(
        badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    frame_h, frame_w = frame.shape[:2]
    badge_x1 = frame_w - badge_w - 28
    badge_y1 = 12
    badge_x2 = frame_w - 10
    badge_y2 = badge_y1 + badge_h + 16
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 2)
    cv2.putText(
        frame,
        badge_text,
        (badge_x1 + 8, badge_y2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def describe_first_frame_context(first_hand_dets, first_obj_dets):
    preview_bits = [f"hands={len(first_hand_dets)}"]
    if first_obj_dets:
        obj_text = ", ".join(
            f"{det['class_name']}({det['confidence']:.0%})" for det in first_obj_dets
        )
        preview_bits.append(f"objects={obj_text}")
    else:
        preview_bits.append("objects=none")
    head_mod.log_info("First-frame preview: " + " | ".join(preview_bits))


def detect_first_frame_context(first_frame, pose_estimator, tracker,
                               roi_polygon=None, hand_detector=None,
                               object_detector=None):
    """Run first-frame detections used by manual or saved setup flows."""
    head_mod.log_info("Running pose detection on first frame for setup...")
    first_detections = pose_estimator.detect_pose(first_frame)
    first_detections = pass_mod.filter_detections_by_roi(first_detections, roi_polygon)
    first_track_ids = tracker.update(first_detections)

    roi_label = " (within ROI)" if roi_polygon is not None else ""
    head_mod.log_info(
        f"Detected {len(first_detections)} persons on first frame{roi_label}."
    )

    if hand_detector is not None and object_detector is not None:
        first_hand_dets = [
            det for det in hand_detector.detect(first_frame)
            if det["class_name"] == hands_mod.CLASS_HAND
        ]
        first_object_dets = []
        for det in object_detector.detect(first_frame):
            if det["class_name"] not in obj_mod.OBJECT_CLASSES:
                continue
            min_conf = obj_mod.CONFIDENCE_THRESHOLDS.get(det["class_name"], 0.25)
            if det["confidence"] < min_conf:
                continue
            first_object_dets.append(det)
        describe_first_frame_context(first_hand_dets, first_object_dets)

    return first_detections, first_track_ids


def run_manual_setup(first_frame, pose_estimator, tracker, disp_scale,
                     hand_detector=None, object_detector=None):
    """Run the current interactive setup flow and return the runtime bundle."""
    head_mod.log_info("Draw ROI boundary to limit tracking area (or press S to skip).")
    roi_result = pass_mod.calibrate_roi(first_frame, disp_scale)
    if isinstance(roi_result, str) and roi_result == "CANCEL":
        head_mod.log_info("Cancelled. Exiting.")
        return None
    roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    first_detections, first_track_ids = detect_first_frame_context(
        first_frame,
        pose_estimator,
        tracker,
        roi_polygon=roi_polygon,
        hand_detector=hand_detector,
        object_detector=object_detector,
    )

    print()
    print(f"  {head_mod.TC.BOLD}Instructions:{head_mod.TC.RESET}")
    print("    1. Click on a person to select them (cyan highlight)")
    print("    2. Type the student number (digits)")
    print("    3. Press ENTER to assign")
    print("    4. Repeat for each student you want to monitor")
    print("    5. Passing-papers alerts need at least 2 consecutive student numbers")
    print("    6. Press S to continue to table-edge calibration")
    print()

    student_map, baseline_yaw_map = head_mod.run_assignment_phase(
        first_frame, first_detections, first_track_ids, disp_scale
    )
    if student_map is None:
        head_mod.log_info("Assignment cancelled. Exiting.")
        return None
    if len(student_map) == 0:
        head_mod.log_info("No students assigned. Exiting.")
        return None
    if len(student_map) < 2:
        head_mod.log_info(
            "Only one student assigned. Passing-papers alerts need at least 2 "
            "students, but the other behaviors will still run."
        )

    tracker.keep_only(set(student_map.keys()))
    head_mod.log_info(f"Tracker locked to {len(student_map)} assigned student(s).")

    assigned_students = hands_mod.build_assigned_student_list(
        first_detections, first_track_ids, student_map
    )
    head_mod.log_info(
        "Draw one student-side table-edge line per assigned student "
        "(press S to skip an individual student)."
    )
    student_lines = hands_mod.calibrate_table_edge_lines(first_frame, assigned_students)
    if student_lines is None:
        head_mod.log_info("Table-edge calibration cancelled. Exiting.")
        return None

    return {
        "roi_polygon": roi_polygon,
        "first_detections": first_detections,
        "first_track_ids": first_track_ids,
        "student_map": student_map,
        "baseline_yaw_map": baseline_yaw_map,
        "assigned_students": assigned_students,
        "student_lines": student_lines,
    }


def load_setup_from_profile(profile_path, first_frame, pose_estimator, tracker):
    """Load a saved setup profile and map it onto the current first frame."""
    profile = setup_io.load_setup_profile(profile_path)
    frame_shape = profile.get("frame_shape")
    if frame_shape and list(frame_shape) != [first_frame.shape[0], first_frame.shape[1]]:
        head_mod.log_info(
            "Saved setup frame size differs from the current video frame size. "
            "Attempting bbox matching anyway."
        )

    roi_polygon = setup_io.profile_to_roi_polygon(profile)
    if roi_polygon is not None:
        head_mod.log_info(
            f"Loaded saved ROI with {len(roi_polygon)} vertices from {profile_path}."
        )
    else:
        head_mod.log_info(f"Loaded saved setup with no ROI from {profile_path}.")

    first_detections, first_track_ids = detect_first_frame_context(
        first_frame, pose_estimator, tracker, roi_polygon=roi_polygon
    )
    runtime_setup = setup_io.build_runtime_setup_from_profile(
        profile, first_detections, first_track_ids
    )

    student_map = runtime_setup["student_map"]
    if len(student_map) == 0:
        head_mod.log_info(
            "Saved setup could not be matched to the current first frame."
        )
        return None

    tracker.keep_only(set(student_map.keys()))
    head_mod.log_info(
        f"Loaded saved setup for {len(student_map)}/"
        f"{runtime_setup['saved_student_count']} student(s)."
    )
    if runtime_setup["unmatched_student_nums"]:
        nums = ", ".join(str(n) for n in runtime_setup["unmatched_student_nums"])
        head_mod.log_info(f"Unmatched saved students skipped: {nums}")
    if len(student_map) < 2:
        head_mod.log_info(
            "Only one saved student matched. Passing-papers alerts need at least 2 "
            "students, but the other behaviors will still run."
        )

    return {
        "roi_polygon": roi_polygon,
        "first_detections": first_detections,
        "first_track_ids": first_track_ids,
        "student_map": runtime_setup["student_map"],
        "baseline_yaw_map": runtime_setup["baseline_yaw_map"],
        "assigned_students": runtime_setup["assigned_students"],
        "student_lines": runtime_setup["student_lines"],
    }


def run_detection(cap, pose_estimator, hand_detector, object_detector, tracker,
                  student_map, baseline_yaw_map, assigned_students, student_lines,
                  source_label, port, roi_polygon=None, source_mode="video",
                  source_fps=None):
    """Run all behavior detectors in a single Pi-side loop."""
    video_name = Path(str(source_label)).stem
    fps = source_fps or cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_mode == "video" else 0
    )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 and total_frames > 0 else 0
    assigned_tids = set(student_map.keys())
    configured_lines = sum(1 for line in student_lines if line is not None)

    head_students = {
        tid: head_mod.StudentState(
            track_id=tid,
            student_num=student_num,
            baseline_yaw=baseline_yaw_map.get(tid, 0.0),
        )
        for tid, student_num in student_map.items()
    }
    passing_students = {
        tid: pass_mod.StudentState(track_id=tid, student_num=student_num)
        for tid, student_num in student_map.items()
    }
    line_states = [
        hands_mod.LineMonitorState(i, student["track_id"])
        for i, student in enumerate(assigned_students)
    ]
    line_index_by_tid = {
        student["track_id"]: idx for idx, student in enumerate(assigned_students)
    }
    pair_states = {}
    object_cooldowns = defaultdict(lambda: -999.0)
    object_stats = defaultdict(int)
    evidence_sequences = []
    recent_evidence_frames = deque(maxlen=EVIDENCE_PRE_EVENT_FRAMES)

    def get_pair_state(tid_a, tid_b):
        key = frozenset((tid_a, tid_b))
        if key not in pair_states:
            pair_states[key] = pass_mod.PairInteractionState(tid_a=tid_a, tid_b=tid_b)
        return pair_states[key]

    print()
    print("=" * 78)
    print("  AISENTINEL - All Behavior Detection (Pi + Hailo)")
    source_heading = "Video" if source_mode == "video" else "Webcam"
    print(f"  {source_heading:14s}: {Path(str(source_label)).name}")
    if total_frames > 0:
        print(
            f"  Resolution     : {width}x{height} | FPS: {fps:.1f} | "
            f"Duration: {head_mod.fmt_ts(duration)}"
        )
    else:
        print(f"  Resolution     : {width}x{height} | FPS: {fps:.1f} | Live source")
    print(f"  Students       : {len(student_map)} assigned")
    print(f"  Line config    : {configured_lines}/{len(student_lines)}")
    roi_text = (
        f"Yes ({len(roi_polygon)} vertices)"
        if roi_polygon is not None else "No (full frame)"
    )
    print(f"  ROI            : {roi_text}")
    print("  Detecting      : head tilt | shoulder turn | passing papers")
    print("                    hands under table | phone | cheat_sheet")
    print(
        f"  Head sustain   : {head_mod.SUSTAINED_SEC:.1f}s | "
        f"Hands sustain: {hands_mod.HANDS_MISSING_SUSTAIN_SEC:.1f}s | "
        f"Passing sustain: {pass_mod.MIN_INTERACTION_SEC:.2f}s"
    )
    print(
        f"  Object cooldown: {obj_mod.EVENT_COOLDOWN_SEC:.1f}s | "
        f"Hand smoothing: {hands_mod.SMOOTH_WINDOW_FRAMES}f"
    )
    print(
        f"  Evidence       : {EVIDENCE_DIR} | "
        f"{EVIDENCE_PRE_EVENT_FRAMES} pre + alert + "
        f"{EVIDENCE_POST_EVENT_FRAMES} post frames"
    )
    print(f"  Web stream     : http://{get_local_ip()}:{port}")
    print("=" * 78)
    print()

    set_dashboard_status(
        "running",
        "Monitoring live classroom session.",
        monitoring_active=True,
        system_state="active",
    )
    update_dashboard_metrics(
        assigned_students=len(student_map),
        total_frames=total_frames,
        source_fps=fps,
        elapsed_text="00:00:00",
    )

    if source_mode == "video":
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    head_stats = defaultdict(int)
    passing_alert_total = 0
    hand_alert_total = 0
    hand_warning_total = 0
    object_alert_total = 0
    object_alert_conf_total = 0.0
    t_start = time.perf_counter()
    source_start = time.perf_counter()
    stream_publish_interval = 1.0 / STREAM_MAX_FPS if STREAM_MAX_FPS > 0 else 0.0
    last_stream_publish_at = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if source_mode == "video":
                    head_mod.log_info("End of video reached.")
                else:
                    head_mod.log_info("Webcam stream ended.")
                break

            frame_idx += 1
            if source_mode == "video":
                ts_sec = frame_idx / fps if fps > 0 else 0.0
            else:
                ts_sec = time.perf_counter() - source_start
            raw_frame = frame.copy()

            t0 = time.perf_counter()
            pose_dets = pose_estimator.detect_pose(frame)
            pose_dets = pass_mod.filter_detections_by_roi(pose_dets, roi_polygon)
            track_ids = tracker.update(pose_dets)

            hand_raw = hand_detector.detect(frame)
            hand_dets = [
                det for det in hand_raw if det["class_name"] == hands_mod.CLASS_HAND
            ]

            object_raw = object_detector.detect(frame)
            object_raw = obj_mod.filter_detections_by_roi(object_raw, roi_polygon)
            object_dets = []
            for det in object_raw:
                cls_name = det["class_name"]
                if cls_name not in obj_mod.OBJECT_CLASSES:
                    continue
                min_conf = obj_mod.CONFIDENCE_THRESHOLDS.get(cls_name, 0.25)
                if det["confidence"] < min_conf:
                    continue
                object_dets.append(det)

            inference_ms = (time.perf_counter() - t0) * 1000

            annotated = frame.copy()
            if roi_polygon is not None:
                cv2.polylines(
                    annotated, [roi_polygon], True, (0, 255, 255), 1, cv2.LINE_AA
                )

            head_frame_events = []
            passing_frame_events = []
            frame_hand_alerts = []
            frame_hand_warnings = []
            frame_object_alerts = []
            frame_kp_data = {}
            student_tracks = {}
            frame_object_boxes = defaultdict(list)
            per_student_display = {}

            # Pass 1: pose tracking and head behavior analysis.
            for i, det in enumerate(pose_dets):
                tid = track_ids[i]
                bbox = det["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]

                if tid == -1 or tid not in assigned_tids:
                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2), head_mod.COL_UNASSIGNED, 1
                    )
                    continue

                kp = det["keypoints"]
                kp_xy = kp[:, :2]
                kp_conf = kp[:, 2]
                student_tracks[tid] = tuple(bbox)
                frame_kp_data[tid] = (kp_xy, kp_conf)
                per_student_display[tid] = {
                    "color": head_mod.COL_NORMAL,
                    "labels": [],
                }

                pass_mod.draw_skeleton(annotated, kp_xy, kp_conf)
                for kp_idx in (pass_mod.KP_LEFT_WRIST, pass_mod.KP_RIGHT_WRIST):
                    if kp_idx < len(kp_conf) and kp_conf[kp_idx] > pass_mod.KP_CONF_THRESH:
                        wx = int(kp_xy[kp_idx][0])
                        wy = int(kp_xy[kp_idx][1])
                        cv2.circle(
                            annotated, (wx, wy), 5, pass_mod.COL_WRIST, -1, cv2.LINE_AA
                        )

                display = per_student_display[tid]
                head_state = head_students[tid]

                is_tilted, tilt_score = head_mod.detect_head_tilt(
                    kp_xy, kp_conf, baseline_yaw=head_state.baseline_yaw
                )
                if is_tilted:
                    if head_state.head_tilt_start < 0:
                        head_state.head_tilt_start = ts_sec
                    elapsed = ts_sec - head_state.head_tilt_start
                    if (
                        elapsed >= head_mod.SUSTAINED_SEC
                        and head_state.can_flag("head_tilt", ts_sec)
                    ):
                        head_state.head_tilt_flagged_at = ts_sec
                        head_stats["head_tilt"] += 1
                        head_mod.log_alert(
                            "HEAD TILT",
                            head_state.student_num,
                            ts_sec,
                            f"score={tilt_score:.2f}, sustained {elapsed:.1f}s",
                            head_mod.TC.YELLOW,
                        )
                        head_frame_events.append(("head_tilt", head_state.student_num))

                    if elapsed >= 1.0:
                        display["labels"].append(
                            f"HEAD TILT {tilt_score:.1f}x ({elapsed:.1f}s)"
                        )
                        display["color"] = elevate_color(
                            display["color"], head_mod.COL_HEAD_TILT
                        )
                        if elapsed >= head_mod.SUSTAINED_SEC:
                            display["color"] = elevate_color(
                                display["color"], head_mod.COL_FLAGGED
                            )
                else:
                    head_state.head_tilt_start = -1.0

                is_turned, shoulder_angle, turn_dir = head_mod.detect_shoulder_turn(
                    kp_xy, kp_conf
                )
                if (
                    kp_conf[head_mod.KP_LEFT_SHOULDER] > head_mod.KP_CONF_THRESH
                    and kp_conf[head_mod.KP_RIGHT_SHOULDER] > head_mod.KP_CONF_THRESH
                ):
                    ls_pt = (
                        int(kp_xy[head_mod.KP_LEFT_SHOULDER][0]),
                        int(kp_xy[head_mod.KP_LEFT_SHOULDER][1]),
                    )
                    rs_pt = (
                        int(kp_xy[head_mod.KP_RIGHT_SHOULDER][0]),
                        int(kp_xy[head_mod.KP_RIGHT_SHOULDER][1]),
                    )
                    shoulder_color = (
                        head_mod.COL_SHOULDER_TURN if is_turned else (100, 200, 100)
                    )
                    cv2.line(annotated, ls_pt, rs_pt, shoulder_color, 3, cv2.LINE_AA)
                    mid_x = (ls_pt[0] + rs_pt[0]) // 2
                    mid_y = (ls_pt[1] + rs_pt[1]) // 2
                    cv2.putText(
                        annotated,
                        f"S:{shoulder_angle:.0f}deg",
                        (mid_x + 5, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        shoulder_color,
                        1,
                        cv2.LINE_AA,
                    )

                if is_turned:
                    if head_state.shoulder_turn_start < 0:
                        head_state.shoulder_turn_start = ts_sec
                    elapsed = ts_sec - head_state.shoulder_turn_start
                    if (
                        elapsed >= head_mod.SUSTAINED_SEC
                        and head_state.can_flag("shoulder_turn", ts_sec)
                    ):
                        head_state.shoulder_turn_flagged_at = ts_sec
                        head_stats["shoulder_turn"] += 1
                        head_mod.log_alert(
                            "SHOULDER TURN",
                            head_state.student_num,
                            ts_sec,
                            (
                                f"direction={turn_dir}, angle={shoulder_angle:.1f}deg, "
                                f"sustained {elapsed:.1f}s"
                            ),
                            head_mod.TC.CYAN,
                        )
                        head_frame_events.append(
                            ("shoulder_turn", head_state.student_num)
                        )

                    if elapsed >= 1.0:
                        display["labels"].append(
                            f"SHOULDER {turn_dir} {shoulder_angle:.0f}deg ({elapsed:.1f}s)"
                        )
                        display["color"] = elevate_color(
                            display["color"], head_mod.COL_SHOULDER_TURN
                        )
                        if elapsed >= head_mod.SUSTAINED_SEC:
                            display["color"] = elevate_color(
                                display["color"], head_mod.COL_FLAGGED
                            )
                else:
                    head_state.shoulder_turn_start = -1.0

            # Pass 2: hand detection overlays and hand-to-student association.
            hand_boxes = []
            for det in hand_dets:
                hand_boxes.append(det["bbox"])
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), hands_mod.COL_HAND, 2)
                hands_mod.draw_label(
                    annotated,
                    f"hand {det['confidence']:.0%}",
                    x1,
                    y1 - 2,
                    hands_mod.COL_HAND,
                )

            student_hands = defaultdict(list)
            for hand_bbox in hand_boxes:
                hx, hy = hands_mod.bbox_center(hand_bbox)
                best_tid = None
                best_dist = float("inf")

                for tid, student_bbox in student_tracks.items():
                    dist = hands_mod.hand_distance_to_bbox((hx, hy), student_bbox)
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                if best_tid is not None and best_dist <= hands_mod.HAND_ASSOC_MARGIN_PX:
                    student_hands[best_tid].append(hand_bbox)
                    sx1, sy1, sx2, sy2 = student_tracks[best_tid]
                    s_cx = int((sx1 + sx2) / 2)
                    s_cy = int((sy1 + sy2) / 2)
                    cv2.line(
                        annotated,
                        (int(hx), int(hy)),
                        (s_cx, s_cy),
                        hands_mod.COL_ASSOC_LINE,
                        1,
                        cv2.LINE_AA,
                    )

            for i, state in enumerate(line_states):
                tid = state.assigned_student_id
                edge_line = student_lines[i] if i < len(student_lines) else None

                if tid in student_tracks:
                    state.last_student_seen_at = ts_sec
                elif (
                    state.last_student_seen_at > 0
                    and (ts_sec - state.last_student_seen_at)
                    > hands_mod.STUDENT_ABSENT_RESET_SEC
                ):
                    state.reset()

                if edge_line is None or tid not in student_tracks:
                    state.push_observation(True)
                    state.hands_missing_start = -1.0
                    state.edge_disappear_start = -1.0
                    state.last_edge_hand_point = None
                    state.last_visible_hands = 0
                    continue

                hands = student_hands.get(tid, [])
                visible_hands = len(hands)
                near_line_hand_count = 0
                nearest_edge_point = None
                nearest_edge_dist = float("inf")

                for hand_bbox in hands:
                    hx, hy = hands_mod.bbox_center(hand_bbox)
                    edge_dist, edge_point = hands_mod.point_to_segment_distance(
                        (hx, hy), edge_line
                    )
                    if edge_dist <= hands_mod.TABLE_EDGE_NEAR_PX:
                        near_line_hand_count += 1
                        if edge_dist < nearest_edge_dist:
                            nearest_edge_dist = edge_dist
                            nearest_edge_point = edge_point

                hands_present = visible_hands >= hands_mod.MIN_VISIBLE_HANDS
                state.last_visible_hands = visible_hands

                if near_line_hand_count > 0:
                    state.note_visible_hands(
                        ts_sec, nearest_edge_point=nearest_edge_point
                    )
                elif not hands_present:
                    state.maybe_arm_edge_disappearance(ts_sec)
                else:
                    state.edge_disappear_start = -1.0
                    state.last_edge_hand_point = None

                state.push_observation(hands_present)
                smoothed_missing = state.majority_says_missing()
                suspicious_missing = (
                    smoothed_missing and (state.edge_disappear_start >= 0)
                )

                if suspicious_missing:
                    if state.hands_missing_start < 0:
                        state.hands_missing_start = ts_sec
                    elapsed = ts_sec - state.hands_missing_start

                    if (
                        elapsed >= hands_mod.HANDS_MISSING_SUSTAIN_SEC
                        and state.can_flag(ts_sec)
                    ):
                        state.last_flagged_at = ts_sec
                        student_num = student_map.get(tid, tid)
                        if visible_hands == 1:
                            state.total_warnings += 1
                            hand_warning_total += 1
                            hands_mod.log_warning(
                                student_num,
                                i,
                                ts_sec,
                                f"line-disappear, only 1 hand visible for {elapsed:.1f}s",
                            )
                            frame_hand_warnings.append((i, student_num))
                        else:
                            state.total_alerts += 1
                            hand_alert_total += 1
                            hands_mod.log_alert(
                                student_num,
                                i,
                                ts_sec,
                                (
                                    f"line-disappear, 0 hands visible for {elapsed:.1f}s "
                                    f"({sum(1 for v in state.history if not v)}/"
                                    f"{len(state.history)} frames)"
                                ),
                            )
                            frame_hand_alerts.append((i, student_num))
                else:
                    state.hands_missing_start = -1.0

                display = per_student_display.get(tid)
                if display is not None and state.hands_missing_start > 0:
                    elapsed = ts_sec - state.hands_missing_start
                    if (
                        elapsed >= hands_mod.HANDS_MISSING_SUSTAIN_SEC
                        and state.last_visible_hands == 0
                    ):
                        label = f"HANDS 0 ({elapsed:.1f}s)"
                        color = hands_mod.COL_ALERT
                    elif (
                        elapsed >= hands_mod.HANDS_MISSING_SUSTAIN_SEC
                        and state.last_visible_hands == 1
                    ):
                        label = f"HANDS 1 ({elapsed:.1f}s)"
                        color = hands_mod.COL_WARNING
                    else:
                        label = f"HANDS WATCH ({elapsed:.1f}s)"
                        color = hands_mod.COL_WARNING
                    display["labels"].append(label)
                    display["color"] = elevate_color(display["color"], color)

            hands_mod.draw_table_edge_lines(
                annotated, student_lines, line_states, assigned_students
            )

            for i, state in enumerate(line_states):
                if state.hands_missing_start <= 0:
                    continue

                line = student_lines[i] if i < len(student_lines) else None
                if line is None:
                    continue

                elapsed = ts_sec - state.hands_missing_start
                pt1 = line[0]
                pt2 = line[1]
                label_x = int((pt1[0] + pt2[0]) / 2) + 8
                label_y = int((pt1[1] + pt2[1]) / 2) + 22

                if (
                    elapsed >= hands_mod.HANDS_MISSING_SUSTAIN_SEC
                    and state.last_visible_hands == 0
                ):
                    text = f"ALERT! 0 hands ({elapsed:.1f}s)"
                    color = hands_mod.COL_ALERT
                elif (
                    elapsed >= hands_mod.HANDS_MISSING_SUSTAIN_SEC
                    and state.last_visible_hands == 1
                ):
                    text = f"WARNING! 1 hand ({elapsed:.1f}s)"
                    color = hands_mod.COL_WARNING
                else:
                    text = f"Watching line ({elapsed:.1f}s)"
                    color = hands_mod.COL_WARNING

                cv2.putText(
                    annotated,
                    text,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    text,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Pass 3: passing papers interactions on the same tracked poses.
            evaluated_pairs = set()
            for tid_a in list(frame_kp_data.keys()):
                neighbors = pass_mod.find_row_neighbors(
                    tid_a, student_tracks, passing_students
                )

                for tid_b, _ in neighbors:
                    pair_key = frozenset((tid_a, tid_b))
                    if pair_key in evaluated_pairs or tid_b not in frame_kp_data:
                        continue
                    evaluated_pairs.add(pair_key)

                    state_a = passing_students[tid_a]
                    state_b = passing_students[tid_b]
                    if abs(state_a.student_num - state_b.student_num) != 1:
                        continue

                    ps = get_pair_state(tid_a, tid_b)
                    kp_a_xy, kp_a_conf = frame_kp_data[tid_a]
                    kp_b_xy, kp_b_conf = frame_kp_data[tid_b]
                    bbox_a = student_tracks[tid_a]
                    bbox_b = student_tracks[tid_b]
                    center_a = (
                        (bbox_a[0] + bbox_a[2]) / 2,
                        (bbox_a[1] + bbox_a[3]) / 2,
                    )
                    center_b = (
                        (bbox_b[0] + bbox_b[2]) / 2,
                        (bbox_b[1] + bbox_b[3]) / 2,
                    )

                    avg_pair_h = (
                        (bbox_a[3] - bbox_a[1]) + (bbox_b[3] - bbox_b[1])
                    ) / 2.0
                    scaled_prox_px = (
                        pass_mod.WRIST_PROXIMITY_PX
                        * pass_mod._perspective_scale(avg_pair_h)
                    )

                    ps.frame_proximity = False
                    ps.frame_proximity_dist = 9999.0

                    prox_dist = pass_mod.compute_wrist_proximity(
                        kp_a_xy, kp_a_conf, kp_b_xy, kp_b_conf
                    )
                    ps.frame_proximity_dist = prox_dist
                    ps.frame_proximity = prox_dist < scaled_prox_px

                    if prox_dist < scaled_prox_px * 1.5:
                        best_pts = None
                        best_dist = 9999.0
                        for wrist_a, wrist_b in (
                            (
                                pass_mod._kp_pos(
                                    kp_a_xy, kp_a_conf, pass_mod.KP_RIGHT_WRIST
                                ),
                                pass_mod._kp_pos(
                                    kp_b_xy, kp_b_conf, pass_mod.KP_LEFT_WRIST
                                ),
                            ),
                            (
                                pass_mod._kp_pos(
                                    kp_a_xy, kp_a_conf, pass_mod.KP_LEFT_WRIST
                                ),
                                pass_mod._kp_pos(
                                    kp_b_xy, kp_b_conf, pass_mod.KP_RIGHT_WRIST
                                ),
                            ),
                        ):
                            if wrist_a and wrist_b:
                                dist = pass_mod._dist(wrist_a, wrist_b)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_pts = (wrist_a, wrist_b)

                        if best_pts:
                            pt_a, pt_b = best_pts
                            line_color = (
                                head_mod.COL_FLAGGED
                                if ps.frame_proximity else pass_mod.COL_WARNING
                            )
                            cv2.line(
                                annotated,
                                (int(pt_a[0]), int(pt_a[1])),
                                (int(pt_b[0]), int(pt_b[1])),
                                line_color,
                                2,
                                cv2.LINE_AA,
                            )
                            mid_x = int((pt_a[0] + pt_b[0]) / 2)
                            mid_y = int((pt_a[1] + pt_b[1]) / 2)
                            head_mod.draw_label(
                                annotated, f"{prox_dist:.0f}px", mid_x, mid_y, line_color
                            )

                    if ps.frame_proximity:
                        if ps.interaction_start < 0:
                            ps.interaction_start = ts_sec
                        ps.last_proximity_time = ts_sec
                        ps.peak_proximity_dist = min(ps.peak_proximity_dist, prox_dist)

                        interaction_dur = ts_sec - ps.interaction_start
                        alert_dir = "RIGHT" if center_b[0] > center_a[0] else "LEFT"

                        for tid_src, neighbor_num in (
                            (tid_a, state_b.student_num),
                            (tid_b, state_a.student_num),
                        ):
                            display = per_student_display.get(tid_src)
                            if display is None:
                                continue
                            display["color"] = elevate_color(
                                display["color"], pass_mod.COL_WARNING
                            )
                            display["labels"].append(
                                f"PROX S#{neighbor_num} {prox_dist:.0f}px {interaction_dur:.1f}s"
                            )

                        cv2.line(
                            annotated,
                            (int(center_a[0]), int(center_a[1])),
                            (int(center_b[0]), int(center_b[1])),
                            pass_mod.COL_NEIGHBOR_LINE,
                            2,
                            cv2.LINE_AA,
                        )

                        if (
                            interaction_dur >= pass_mod.MIN_INTERACTION_SEC
                            and ps.can_flag(ts_sec)
                        ):
                            ps.last_flagged_at = ts_sec
                            state_a.total_alerts += 1
                            state_b.total_alerts += 1
                            passing_alert_total += 1

                            pass_mod.log_alert(
                                "PASSING PAPERS",
                                [state_a.student_num, state_b.student_num],
                                ts_sec,
                                (
                                    f"{alert_dir}, dur={interaction_dur:.1f}s, "
                                    f"closest={ps.peak_proximity_dist:.0f}px"
                                ),
                                pass_mod.TC.RED,
                            )
                            passing_frame_events.append(
                                (state_a.student_num, state_b.student_num, alert_dir)
                            )

                            for tid_src in (tid_a, tid_b):
                                display = per_student_display.get(tid_src)
                                if display is not None:
                                    display["color"] = elevate_color(
                                        display["color"], head_mod.COL_FLAGGED
                                    )

                            ps.reset_interaction()
                    elif ps.interaction_start > 0:
                        ps.reset_interaction()

            # Pass 4: phone / cheat_sheet association and alerts.
            object_associations = obj_mod.associate_objects_to_students(
                object_dets, student_tracks
            )
            for det, assoc_tid in object_associations:
                cls_name = det["class_name"]
                conf = det["confidence"]
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                color = obj_mod.CLASS_COLORS.get(cls_name, (255, 255, 255))
                object_stats[cls_name] += 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_prefix = cls_name.replace("_", " ")

                if assoc_tid != -1 and assoc_tid in student_map:
                    student_num = student_map[assoc_tid]
                    frame_object_boxes[(student_num, cls_name)].append(
                        tuple(det["bbox"])
                    )
                    hands_mod.draw_label(
                        annotated,
                        f"{label_prefix} {conf:.0%} [S#{student_num}]",
                        x1,
                        y1 - 2,
                        color,
                    )
                    display = per_student_display.get(assoc_tid)
                    if display is not None:
                        display["labels"].append(f"{cls_name.upper()} {conf:.0%}")
                        display["color"] = elevate_color(
                            display["color"], head_mod.COL_FLAGGED
                        )
                else:
                    hands_mod.draw_label(
                        annotated,
                        f"{label_prefix} {conf:.0%}",
                        x1,
                        y1 - 2,
                        color,
                    )

                if assoc_tid == -1 or assoc_tid not in student_map:
                    continue

                student_num = student_map[assoc_tid]
                cooldown_key = (assoc_tid, cls_name)
                if (
                    ts_sec - object_cooldowns[cooldown_key]
                    >= obj_mod.EVENT_COOLDOWN_SEC
                ):
                    object_alert_total += 1
                    object_alert_conf_total += conf
                    obj_mod.log_alert(cls_name, student_num, conf, ts_sec)
                    object_cooldowns[cooldown_key] = ts_sec
                    frame_object_alerts.append(
                        {
                            "class_name": cls_name,
                            "student_num": student_num,
                            "confidence": conf,
                        }
                    )

            # Pass 5: final student boxes and labels.
            for i, det in enumerate(pose_dets):
                tid = track_ids[i]
                if tid == -1 or tid not in assigned_tids:
                    continue

                bbox = det["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                display = per_student_display.get(
                    tid, {"color": head_mod.COL_NORMAL, "labels": []}
                )
                student_num = student_map[tid]
                line_idx = line_index_by_tid.get(tid)
                suffix = f" | L{line_idx + 1}" if line_idx is not None else ""

                cv2.rectangle(annotated, (x1, y1), (x2, y2), display["color"], 2)
                head_mod.draw_label(
                    annotated,
                    f"Student #{student_num}{suffix}",
                    x1,
                    y1 - 2,
                    display["color"],
                )

                label_y = y1 + 18
                for label in display["labels"]:
                    head_mod.draw_label(
                        annotated, label, x1, label_y, display["color"]
                    )
                    label_y += 18

            # HUD + banners.
            elapsed_wall = time.perf_counter() - t_start
            actual_fps = frame_idx / elapsed_wall if elapsed_wall > 0 else 0.0
            head_alert_total = sum(head_stats.values())
            tracked_count = len(student_tracks)
            has_warning = any(
                display["color"] in (
                    head_mod.COL_HEAD_TILT,
                    head_mod.COL_SHOULDER_TURN,
                    pass_mod.COL_WARNING,
                    hands_mod.COL_WARNING,
                )
                for display in per_student_display.values()
            )
            hud_color = pass_mod.COL_HUD
            if (
                head_alert_total > 0
                or passing_alert_total > 0
                or hand_alert_total > 0
                or object_alert_total > 0
            ):
                hud_color = head_mod.COL_FLAGGED
            elif has_warning or hand_warning_total > 0:
                hud_color = hands_mod.COL_WARNING

            hud_lines = [
                (
                    f"Frame: {frame_idx}/{total_frames} | "
                    f"Time: {head_mod.fmt_ts(ts_sec)}"
                    if total_frames > 0
                    else f"Frame: {frame_idx} | Live time: {head_mod.fmt_ts(ts_sec)}"
                ),
                f"Video FPS: {fps:.1f} | Processing FPS: {actual_fps:.1f}",
                (
                    f"Tracked: {tracked_count}/{len(student_map)} | "
                    f"Hands: {len(hand_boxes)} | Obj: {len(object_dets)} | "
                    f"Inf: {inference_ms:.0f}ms"
                ),
                (
                    f"Head A: {head_alert_total} | Passing A: {passing_alert_total} | "
                    f"Hands A/W: {hand_alert_total}/{hand_warning_total} | "
                    f"Obj A: {object_alert_total}"
                ),
            ]

            for idx, line in enumerate(hud_lines):
                y_pos = 25 + idx * 28
                cv2.putText(
                    annotated,
                    line,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    line,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    hud_color,
                    2,
                    cv2.LINE_AA,
                )

            draw_fps_badge(annotated, actual_fps, hud_color)

            banner_y = height - 30
            for event in frame_object_alerts:
                text = (
                    f"ALERT: Student #{event['student_num']} - "
                    f"{event['class_name'].replace('_', ' ').upper()}"
                )
                cv2.putText(
                    annotated,
                    text,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    text,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    head_mod.COL_FLAGGED,
                    2,
                    cv2.LINE_AA,
                )
                banner_y -= 35

            for line_idx, student_num in frame_hand_alerts:
                text = (
                    f"ALERT: Student #{student_num} hands missing near "
                    f"Line #{line_idx + 1}"
                )
                cv2.putText(
                    annotated,
                    text,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    text,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    hands_mod.COL_ALERT,
                    2,
                    cv2.LINE_AA,
                )
                banner_y -= 35

            for line_idx, student_num in frame_hand_warnings:
                text = (
                    f"WARNING: Student #{student_num} long hands-missing event "
                    f"near Line #{line_idx + 1}"
                )
                cv2.putText(
                    annotated,
                    text,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    text,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    hands_mod.COL_WARNING,
                    2,
                    cv2.LINE_AA,
                )
                banner_y -= 35

            for behavior, student_num in head_frame_events:
                text = (
                    f"ALERT: Student #{student_num} - "
                    f"{behavior.replace('_', ' ').upper()}"
                )
                cv2.putText(
                    annotated,
                    text,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    text,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    head_mod.COL_FLAGGED,
                    2,
                    cv2.LINE_AA,
                )
                banner_y -= 35

            for src_num, nbr_num, direction in passing_frame_events:
                text = (
                    f"ALERT: S#{src_num} & S#{nbr_num} PASSING PAPERS ({direction})"
                )
                cv2.putText(
                    annotated,
                    text,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    text,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    head_mod.COL_FLAGGED,
                    2,
                    cv2.LINE_AA,
                )
                banner_y -= 35

            ts_text = head_mod.fmt_ts(ts_sec)
            (ts_width, _), _ = cv2.getTextSize(
                ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.putText(
                annotated,
                ts_text,
                (width - ts_width - 10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                ts_text,
                (width - ts_width - 10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            evidence_snapshot = {
                "raw_frame": raw_frame,
                "frame_ts_sec": ts_sec,
                "student_boxes": {
                    student_map[tid]: tuple(bbox)
                    for tid, bbox in student_tracks.items()
                    if tid in student_map
                },
                "object_boxes": dict(frame_object_boxes),
            }
            new_sequences = []

            for behavior, student_num in head_frame_events:
                new_sequences.append(
                    queue_evidence_sequence(
                        evidence_sequences,
                        recent_evidence_frames,
                        "head",
                        ts_sec,
                        student_num=student_num,
                        behavior=behavior,
                    )
                )

            for src_num, nbr_num, _ in passing_frame_events:
                new_sequences.append(
                    queue_evidence_sequence(
                        evidence_sequences,
                        recent_evidence_frames,
                        "passing",
                        ts_sec,
                        student_nums=[src_num, nbr_num],
                    )
                )

            for line_idx, student_num in frame_hand_alerts:
                new_sequences.append(
                    queue_evidence_sequence(
                        evidence_sequences,
                        recent_evidence_frames,
                        "hands",
                        ts_sec,
                        video_name=video_name,
                        line_idx=line_idx,
                        student_num=student_num,
                    )
                )

            for event in frame_object_alerts:
                new_sequences.append(
                    queue_evidence_sequence(
                        evidence_sequences,
                        recent_evidence_frames,
                        "object",
                        ts_sec,
                        student_num=event["student_num"],
                        class_name=event["class_name"],
                        confidence=event["confidence"],
                    )
                )

            evidence_sequences = flush_evidence_sequences(
                evidence_sequences, evidence_snapshot
            )
            recent_evidence_frames.append(evidence_snapshot)

            for sequence in new_sequences:
                record_dashboard_incident(
                    _build_sequence_incident(sequence, status="recording")
                )

            update_dashboard_metrics(
                total_incidents=(
                    head_alert_total
                    + passing_alert_total
                    + hand_alert_total
                    + object_alert_total
                ),
                head_alerts=head_alert_total,
                passing_alerts=passing_alert_total,
                hand_alerts=hand_alert_total,
                hand_warnings=hand_warning_total,
                object_alerts=object_alert_total,
                tracked_students=tracked_count,
                assigned_students=len(student_map),
                processing_fps=round(actual_fps, 2),
                source_fps=round(fps, 2),
                inference_ms=round(inference_ms, 1),
                frame_idx=frame_idx,
                total_frames=total_frames,
                elapsed_text=head_mod.fmt_ts(ts_sec),
                object_confidence_avg=(
                    round(object_alert_conf_total / object_alert_total, 4)
                    if object_alert_total > 0 else 0.0
                ),
                hand_detections=len(hand_boxes),
                object_detections=len(object_dets),
            )

            now_perf = time.perf_counter()
            if (
                frame_idx == 1
                or stream_publish_interval <= 0
                or (now_perf - last_stream_publish_at) >= stream_publish_interval
            ):
                if _publish_dashboard_frame(annotated):
                    last_stream_publish_at = now_perf

            if total_frames > 0 and frame_idx % 500 == 0:
                pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                head_mod.log_info(
                    f"Progress: {pct:.1f}% ({frame_idx}/{total_frames}) | "
                    f"FPS: {actual_fps:.1f}"
                )

    except KeyboardInterrupt:
        head_mod.log_info("Interrupted by user.")
        set_dashboard_status(
            "completed",
            "Monitoring was stopped by the operator.",
            monitoring_active=False,
            system_state="idle",
        )

    elapsed = time.perf_counter() - t_start
    head_alert_total = sum(head_stats.values())
    update_dashboard_metrics(
        total_incidents=(
            head_alert_total
            + passing_alert_total
            + hand_alert_total
            + object_alert_total
        ),
        head_alerts=head_alert_total,
        passing_alerts=passing_alert_total,
        hand_alerts=hand_alert_total,
        hand_warnings=hand_warning_total,
        object_alerts=object_alert_total,
        processing_fps=round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0,
        monitoring_complete=True,
    )
    set_dashboard_status(
        "completed",
        "Monitoring session ended.",
        monitoring_active=False,
        system_state="idle",
    )

    print()
    print("=" * 78)
    print(f"  Summary: {Path(str(source_label)).name}")
    print("-" * 78)
    print(f"  Frames processed : {frame_idx}")
    if elapsed > 0:
        print(f"  Average FPS      : {frame_idx / elapsed:.1f}")
    print(f"  Students tracked : {len(student_map)}")
    print(f"  Line config      : {configured_lines}/{len(student_lines)}")
    print(f"  Head alerts      : {head_alert_total}")
    for behavior, count in sorted(head_stats.items()):
        print(f"    {behavior:20s}: {count}")
    print(f"  Passing alerts   : {passing_alert_total}")
    for _, state in sorted(
        passing_students.items(), key=lambda item: item[1].student_num
    ):
        if state.total_alerts > 0:
            print(
                f"    Student #{state.student_num:2d} : "
                f"{state.total_alerts} passing events"
            )
    print(f"  Hands alerts     : {hand_alert_total}")
    print(f"  Hands warnings   : {hand_warning_total}")
    for i, state in enumerate(line_states):
        if state.total_alerts <= 0 and state.total_warnings <= 0:
            continue
        tid = state.assigned_student_id
        student_num = student_map.get(tid, "?")
        print(
            f"    Line #{i + 1:2d} (Student #{student_num})"
            f" : {state.total_alerts} alerts, {state.total_warnings} warnings"
        )
    print(f"  Object alerts    : {object_alert_total}")
    for cls_name, count in sorted(object_stats.items()):
        print(f"    {cls_name:20s}: {count} detections")
    if (
        head_alert_total > 0
        or passing_alert_total > 0
        or hand_alert_total > 0
        or hand_warning_total > 0
        or object_alert_total > 0
    ):
        print(f"  Evidence saved to: {EVIDENCE_DIR}")
    else:
        print("  No combined alerts triggered.")
    print("=" * 78)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AISENTINEL - All Behavior Detection (Pi + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 front_node_all_behavior_pi.py
  python3 front_node_all_behavior_pi.py --port 9090
  python3 front_node_all_behavior_pi.py --object-confidence 0.4
  python3 front_node_all_behavior_pi.py --pose-model /path/to/yolov8s_pose.hef
        """,
    )
    parser.add_argument(
        "--pose-model",
        default=str(POSE_MODEL_PATH),
        help=f"Path to pose HEF model (default: {POSE_MODEL_PATH})",
    )
    parser.add_argument(
        "--hand-model",
        default=str(HAND_MODEL_PATH),
        help=f"Path to hand HEF model (default: {HAND_MODEL_PATH})",
    )
    parser.add_argument(
        "--object-model", "--model",
        dest="object_model",
        default=str(OBJECT_MODEL_PATH),
        help=f"Path to phone / cheat_sheet HEF model (default: {OBJECT_MODEL_PATH})",
    )
    parser.add_argument(
        "--pose-confidence",
        type=float,
        default=0.5,
        help="Pose/person confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--object-confidence", "--confidence",
        dest="object_confidence",
        type=float,
        default=0.25,
        help="Base confidence threshold for the object model (default: 0.25)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Flask web server port (default: 8080)",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Path to a saved ROI/assignment/desk-line setup JSON",
    )
    parser.add_argument(
        "--ignore-saved-calibration",
        action="store_true",
        help="Force manual setup even if a saved setup JSON exists",
    )
    args = parser.parse_args()

    print()
    print("=" * 78)
    print("  AISENTINEL - All Behavior Detection")
    print("  Pose model      : shared YOLOv8 pose HEF")
    print("  Hand model      : sentinel-yolo11n-min.hef")
    print("  Object model    : sentinel-yolov11n_new.hef")
    print("  Detects         : head tilt | shoulder turn | passing papers")
    print("                    hands under table | phone | cheat_sheet")
    print("  Calibration flow: ROI -> assignment -> table-edge lines")
    print("  Overlay         : unified stream with processing FPS")
    print("=" * 78)
    print()

    if (
        not head_mod.HAILO_AVAILABLE
        or not hands_mod.HAILO_AVAILABLE
        or not obj_mod.HAILO_AVAILABLE
    ):
        print(f"{head_mod.TC.RED}[ERROR] hailo_platform is required.{head_mod.TC.RESET}")
        print("Install: sudo apt install hailo-all")
        sys.exit(1)

    pose_path = Path(args.pose_model)
    if not pose_path.exists():
        print(f"{head_mod.TC.RED}[ERROR] Pose HEF model not found: {pose_path}{head_mod.TC.RESET}")
        print("See POSE_MODEL_SETUP.md for download instructions.")
        sys.exit(1)

    hand_path = Path(args.hand_model)
    if not hand_path.exists():
        print(f"{head_mod.TC.RED}[ERROR] Hand HEF model not found: {hand_path}{head_mod.TC.RESET}")
        sys.exit(1)

    object_path = Path(args.object_model)
    if not object_path.exists():
        print(f"{head_mod.TC.RED}[ERROR] Object HEF model not found: {object_path}{head_mod.TC.RESET}")
        sys.exit(1)

    head_mod.log_info("Opening file dialog...")
    video_path = pass_mod.select_video_dialog()
    if not video_path:
        head_mod.log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{head_mod.TC.RED}[ERROR] File not found: {video_path}{head_mod.TC.RESET}")
        sys.exit(1)
    head_mod.log_info(f"Selected: {video_path}")

    shared_vdevice = hands_mod.VDevice()
    head_mod.log_info("Hailo VDevice created (shared across all models).")

    pose_estimator = SharedHailoPoseEstimator(
        str(pose_path),
        conf_threshold=args.pose_confidence,
        vdevice=shared_vdevice,
    )
    hand_detector = hands_mod.HailoObjectDetector(
        str(hand_path),
        class_names=hands_mod.HAND_MODEL_CLASS_NAMES,
        conf_threshold=hands_mod.HAND_CONFIDENCE,
        vdevice=shared_vdevice,
    )
    object_detector = obj_mod.HailoObjectDetector(
        str(object_path),
        conf_threshold=args.object_confidence,
        vdevice=shared_vdevice,
    )

    print(f"\n{head_mod.TC.BOLD}Object model classes:{head_mod.TC.RESET}")
    for idx, name in obj_mod.CLASS_NAMES.items():
        role = "  << ALERT" if name in obj_mod.ALERT_CLASSES else "  << IGNORED"
        thresh = obj_mod.CONFIDENCE_THRESHOLDS.get(name, "-")
        print(f"  [{idx}] {name} (thresh={thresh}){role}")

    print(f"\n{head_mod.TC.BOLD}Hand model classes:{head_mod.TC.RESET}")
    for idx, name in hands_mod.HAND_MODEL_CLASS_NAMES.items():
        role = "  << USED" if name == hands_mod.CLASS_HAND else "  << IGNORED"
        print(f"  [{idx}] {name}{role}")
    print()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{head_mod.TC.RED}[ERROR] Cannot open video: {video_path}{head_mod.TC.RESET}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{head_mod.TC.RED}[ERROR] Cannot read first frame.{head_mod.TC.RESET}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    head_mod.log_info(
        f"Video resolution: {width}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )
    calibration_path = None
    explicit_calibration = bool(args.calibration_file)
    if explicit_calibration:
        calibration_path = Path(args.calibration_file)
        if not calibration_path.exists():
            cap.release()
            print(
                f"{head_mod.TC.RED}[ERROR] Setup file not found: "
                f"{calibration_path}{head_mod.TC.RESET}"
            )
            sys.exit(1)
    elif not args.ignore_saved_calibration:
        auto_calibration = setup_io.default_setup_profile_path(video_path)
        if auto_calibration.exists():
            calibration_path = auto_calibration

    setup_bundle = None
    tracker = ReacquiringLockedIoUTracker(iou_threshold=0.3, max_lost=60)

    if calibration_path is not None:
        try:
            head_mod.log_info(f"Loading saved setup: {calibration_path}")
            setup_bundle = load_setup_from_profile(
                calibration_path, first_frame, pose_estimator, tracker
            )
        except Exception as exc:
            if explicit_calibration:
                cap.release()
                print(
                    f"{head_mod.TC.RED}[ERROR] Failed to load setup file: "
                    f"{calibration_path}{head_mod.TC.RESET}"
                )
                print(str(exc))
                sys.exit(1)
            head_mod.log_info(
                f"Saved setup could not be used ({exc}). Falling back to manual setup."
            )
            tracker = ReacquiringLockedIoUTracker(iou_threshold=0.3, max_lost=60)

    if setup_bundle is None:
        if calibration_path is not None:
            head_mod.log_info("Falling back to manual setup.")
        tracker = ReacquiringLockedIoUTracker(iou_threshold=0.3, max_lost=60)
        setup_bundle = run_manual_setup(
            first_frame,
            pose_estimator,
            tracker,
            disp_scale,
            hand_detector=hand_detector,
            object_detector=object_detector,
        )
        if setup_bundle is None:
            cap.release()
            sys.exit(0)

    roi_polygon = setup_bundle["roi_polygon"]
    student_map = setup_bundle["student_map"]
    baseline_yaw_map = setup_bundle["baseline_yaw_map"]
    assigned_students = setup_bundle["assigned_students"]
    student_lines = setup_bundle["student_lines"]

    if not FLASK_AVAILABLE:
        print(f"{head_mod.TC.RED}[ERROR] Flask is required for web streaming.{head_mod.TC.RESET}")
        print("Install: pip install flask")
        sys.exit(1)

    start_web_server(args.port)
    head_mod.log_info(f"Web stream at http://{get_local_ip()}:{args.port}")
    head_mod.log_info("Starting all-behavior detection...")

    try:
        run_detection(
            cap,
            pose_estimator,
            hand_detector,
            object_detector,
            tracker,
            student_map,
            baseline_yaw_map,
            assigned_students,
            student_lines,
            video_path,
            args.port,
            roi_polygon=roi_polygon,
        )
    finally:
        cap.release()
        if hasattr(pose_estimator, "close"):
            pose_estimator.close()
        if hasattr(hand_detector, "close"):
            hand_detector.close()
        if hasattr(object_detector, "close"):
            object_detector.close()

    head_mod.log_info("Done!")


if __name__ == "__main__":
    main()
