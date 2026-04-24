"""Runtime controller for the standalone node agent."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlparse

import cv2

from central_dashboard.node_agent.detector import EvidenceBuilder, MotionDetector, annotate_frame
from central_dashboard.node_agent.front_runtime import run_front_runtime_session
from central_dashboard.node_agent.sync import LocalSyncQueue
from central_dashboard.shared.dto import (
    CommandAck,
    IncidentManifest,
    NodeDescriptor,
    NodeHeartbeat,
    SessionSpec,
    SyncQueueItem,
    utc_now_iso,
)
from central_dashboard.shared.http import StdlibHttpClient
from sound_monitor import DEFAULT_SOUND_SNAPSHOT

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional on target systems
    PIL_AVAILABLE = False


SYNC_OK = "ok"
SYNC_RETRY = "retry"
SYNC_DROP = "drop"


class NodeRuntime:
    """Supervises node registration, session execution, local feeds, and sync."""

    def __init__(self, config, *, http_client=None, front_runtime_runner=None) -> None:
        self.config = config
        self.logger = logging.getLogger(f"central_dashboard.node_agent.{config.node_id}")
        self.http_client = http_client or StdlibHttpClient()
        self.front_runtime_runner = front_runtime_runner or run_front_runtime_session
        self.sync_queue = LocalSyncQueue(config.local_db_path)
        self.config.evidence_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._session_stop = threading.Event()
        self._sync_wake = threading.Event()
        self._background_started = False
        self._session_thread: threading.Thread | None = None

        self._raw_jpeg = None
        self._annotated_jpeg = None
        self._raw_seq = 0
        self._annotated_seq = 0
        self._last_publish_monotonic = 0.0
        self._last_frame_at = ""

        self._status = "idle"
        self._session: SessionSpec | None = None
        self._fps = 0.0
        self._incident_count = 0
        self._incident_ids: set[str] = set()
        self._last_error = ""
        self._banner_text = ""
        self._banner_expires = 0.0
        self._registration_ok = False
        self._sound_state = dict(DEFAULT_SOUND_SNAPSHOT)

    def start_background(self) -> None:
        if self._background_started:
            return
        self._background_started = True
        self.logger.info(
            "starting background workers: central=%s advertised=%s",
            self.config.central_base_url,
            self.advertised_base_url(),
        )
        for target, name in (
            (self._registration_loop, "node-register"),
            (self._heartbeat_loop, "node-heartbeat"),
            (self._sync_loop, "node-sync"),
        ):
            thread = threading.Thread(target=target, daemon=True, name=name)
            thread.start()

    def shutdown(self) -> None:
        self.logger.info("shutdown requested")
        self._shutdown.set()
        self._sync_wake.set()
        self.stop_session()

    def close(self) -> None:
        self.shutdown()
        self.sync_queue.close()

    def advertised_base_url(self) -> str:
        host = self.config.host
        if host not in {"0.0.0.0", "::", "127.0.0.1", "localhost"}:
            return f"http://{host}:{self.config.port}"

        parsed = urlparse(self.config.central_base_url)
        probe_host = parsed.hostname or "8.8.8.8"
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect((probe_host, 80))
            host = sock.getsockname()[0]
        except OSError:
            host = "127.0.0.1"
        finally:
            try:
                sock.close()
            except Exception:
                pass
        return f"http://{host}:{self.config.port}"

    def descriptor(self) -> NodeDescriptor:
        base_url = self.advertised_base_url()
        return NodeDescriptor(
            node_id=self.config.node_id,
            display_name=self.config.display_name,
            camera_label=self.config.camera_label,
            base_url=base_url,
            agent_base_url=base_url,
            profile=self.config.profile,
        )

    def heartbeat(self) -> NodeHeartbeat:
        with self._lock:
            session_id = self._session.session_id if self._session else ""
            state = self._status
            fps = self._fps
            incident_count = self._incident_count
            last_error = self._last_error
            stream_state = {
                "raw_seq": self._raw_seq,
                "annotated_seq": self._annotated_seq,
                "has_raw_frame": self._raw_jpeg is not None,
                "has_annotated_frame": self._annotated_jpeg is not None,
                "last_frame_at": self._last_frame_at,
                "preview_fps": self.config.preview_fps,
            }
        return NodeHeartbeat(
            node_id=self.config.node_id,
            state=state,
            session_id=session_id,
            fps=fps,
            sync_backlog=self.sync_queue.backlog_count(),
            incident_count=incident_count,
            last_error=last_error,
            extra={
                "profile": self.config.profile,
                "detector_mode": self.config.detector_mode,
                "sound": dict(self._sound_state),
                "stream": stream_state,
            },
        )

    def status_payload(self) -> dict:
        heartbeat = self.heartbeat()
        payload = heartbeat.to_dict()
        payload["base_url"] = self.advertised_base_url()
        return payload

    def start_session(self, session_payload: dict) -> CommandAck:
        session = SessionSpec.from_dict(session_payload)
        with self._lock:
            if self._session_thread is not None and self._session_thread.is_alive():
                return CommandAck(
                    ok=False,
                    node_id=self.config.node_id,
                    action="start",
                    session_id=session.session_id,
                    state=self._status,
                    message="Session is already running.",
                )
        if self.config.clear_sync_backlog_on_session_start:
            purged = self.sync_queue.clear_except_session(session.session_id)
            if purged:
                self.logger.info(
                    "cleared %s stale sync queue item(s) before session start: session_id=%s",
                    purged,
                    session.session_id,
                )
        with self._lock:
            self._session = session
            self._session_stop = threading.Event()
            self._status = "starting"
            self._last_error = ""
            self._clear_frames_locked()

        self.logger.info("session start accepted: session_id=%s", session.session_id)
        self._session_thread = threading.Thread(
            target=self._run_session,
            args=(session,),
            daemon=True,
            name=f"node-session-{self.config.node_id}",
        )
        self._session_thread.start()
        return CommandAck(
            ok=True,
            node_id=self.config.node_id,
            action="start",
            session_id=session.session_id,
            state="starting",
            message="Session accepted.",
        )

    def stop_session(self) -> CommandAck:
        with self._lock:
            session_id = self._session.session_id if self._session else ""
            thread = self._session_thread
            self._status = "stopping" if thread and thread.is_alive() else "idle"
        if session_id:
            self.logger.info("session stop requested: session_id=%s", session_id)
        self._session_stop.set()
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        with self._lock:
            self._status = "idle"
            self._session = None
            self._fps = 0.0
            self._clear_frames_locked()
        return CommandAck(
            ok=True,
            node_id=self.config.node_id,
            action="stop",
            session_id=session_id,
            state="idle",
            message="Session stopped.",
        )

    def restart_session(self, session_payload: dict) -> CommandAck:
        self.stop_session()
        return self.start_session(session_payload)

    def _registration_loop(self) -> None:
        while not self._shutdown.is_set():
            self.register_once()
            self._shutdown.wait(self.config.registration_interval_sec)

    def _heartbeat_loop(self) -> None:
        while not self._shutdown.is_set():
            self.heartbeat_once()
            self._shutdown.wait(self.config.heartbeat_interval_sec)

    def _sync_loop(self) -> None:
        while not self._shutdown.is_set():
            self._sync_wake.clear()
            self.sync_once()
            self._sync_wake.wait(self.config.sync_interval_sec)

    def register_once(self) -> bool:
        was_registered = self._registration_ok
        try:
            result = self.http_client.post_json(
                f"{self.config.central_base_url}/api/v1/nodes/register",
                self.descriptor().to_dict(),
                headers=self._auth_headers(),
                timeout=self.config.http_timeout_sec,
            )
            self._registration_ok = result.ok
            if not result.ok:
                message = result.text or "registration failed"
                self._set_error(message)
                self.logger.warning("registration failed: %s", message)
            elif not was_registered:
                self.logger.info(
                    "registered with central: node_id=%s advertised=%s",
                    self.config.node_id,
                    self.advertised_base_url(),
                )
            return result.ok
        except Exception as exc:  # pragma: no cover - runtime network safety
            self._registration_ok = False
            self._set_error(str(exc))
            self.logger.warning("registration failed: %s", exc)
            return False

    def heartbeat_once(self) -> bool:
        try:
            heartbeat = self.heartbeat()
            result = self.http_client.post_json(
                f"{self.config.central_base_url}/api/v1/nodes/heartbeat",
                heartbeat.to_dict(),
                headers=self._auth_headers(),
                timeout=self.config.http_timeout_sec,
            )
            if result.ok:
                with self._lock:
                    if self._last_error.startswith("heartbeat") or self._last_error.startswith("registration"):
                        self._last_error = ""
                stream = heartbeat.extra.get("stream", {})
                self.logger.info(
                    "heartbeat sent: state=%s session=%s fps=%.1f backlog=%s stream=%s",
                    heartbeat.state,
                    heartbeat.session_id or "-",
                    heartbeat.fps,
                    heartbeat.sync_backlog,
                    "ready" if stream.get("has_annotated_frame") or stream.get("has_raw_frame") else "waiting",
                )
            else:
                self._set_error(f"heartbeat failed: {result.text}")
                self.logger.warning("heartbeat failed: %s", result.text)
            return result.ok
        except Exception as exc:  # pragma: no cover - runtime network safety
            self._set_error(f"heartbeat failed: {exc}")
            self.logger.warning("heartbeat failed: %s", exc)
            return False

    def sync_once(self) -> int:
        self.sync_queue.purge_asset_type("frame")
        synced = 0
        for item in self._prioritized_due_items():
            try:
                if item.item_type == "manifest":
                    disposition = SYNC_OK if self._sync_manifest(item.payload) else SYNC_RETRY
                else:
                    if self.sync_queue.has_pending_manifest(item.incident_id):
                        self.sync_queue.mark_retry(item, "waiting for incident manifest sync")
                        continue
                    disposition = self._sync_asset(item)
                if disposition == SYNC_OK:
                    self.sync_queue.mark_done(item.item_id)
                    if item.item_type == "manifest" and _queue_item_manifest_status(item) != "recording":
                        self.sync_queue.purge_recording_manifests(item.incident_id)
                    synced += 1
                elif disposition == SYNC_DROP:
                    self.sync_queue.mark_done(item.item_id)
                else:
                    self.sync_queue.mark_retry(item, "sync failed")
            except Exception as exc:  # pragma: no cover - runtime network safety
                self.sync_queue.mark_retry(item, str(exc))
                self._set_error(f"sync failed: {exc}")
                self.logger.warning("sync failed: %s", exc)
        if synced:
            self.logger.info("synced %s queued item(s)", synced)
        return synced

    def _prioritized_due_items(self) -> list[SyncQueueItem]:
        with self._lock:
            active_session_id = self._session.session_id if self._session else ""

        items = self.sync_queue.due_items(limit=32)
        if not active_session_id:
            return items[:8]

        def item_priority(item: SyncQueueItem) -> tuple[int, str]:
            return (
                0 if _queue_item_session_id(item) == active_session_id else 1,
                item.next_retry_at,
            )

        return sorted(items, key=item_priority)[:8]

    def _sync_manifest(self, payload: dict) -> bool:
        manifest = payload.get("manifest_payload")
        if not isinstance(manifest, dict):
            manifest_path = Path(payload["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = self.http_client.post_json(
            f"{self.config.central_base_url}/api/v1/incidents",
            manifest,
            headers=self._auth_headers(),
            timeout=self.config.http_timeout_sec,
        )
        return result.ok

    def _sync_asset(self, item: SyncQueueItem) -> str:
        payload = item.payload
        file_path = Path(payload["file_path"])
        if not file_path.exists():
            if _asset_payload_type(payload) == "gif" and _write_missing_gif_asset(payload, file_path):
                self.logger.info("rebuilt missing GIF evidence before upload: %s", file_path)
            elif _asset_payload_type(payload) == "gif":
                self._set_error(f"GIF evidence not ready for upload: {file_path}")
                return SYNC_RETRY
            else:
                self._set_error(f"evidence upload dropped: missing file {file_path}")
                return SYNC_DROP
        if not file_path.exists():
            return SYNC_DROP
        content = file_path.read_bytes()
        asset_payload = {
            "incident_id": payload["incident_id"],
            "session_id": payload["session_id"],
            "node_id": self.config.node_id,
            "asset_type": payload["asset_type"],
            "filename": payload["filename"],
            "content_base64": base64.b64encode(content).decode("ascii"),
            "content_sha256": "",
            "size_bytes": len(content),
        }
        result = self.http_client.post_json(
            f"{self.config.central_base_url}/api/v1/evidence/upload",
            asset_payload,
            headers=self._auth_headers(),
            timeout=self.config.http_timeout_sec,
        )
        if result.ok:
            return SYNC_OK
        if result.status_code == 400:
            error_text = _result_error_text(result) or "bad evidence upload request"
            self._set_error(f"evidence upload failed: {error_text}")
            return SYNC_DROP if item.attempts >= 3 else SYNC_RETRY
        if result.status_code == 404 and _is_incident_not_found(result):
            self._set_error(f"evidence upload dropped: {_result_error_text(result)}")
            return SYNC_DROP
        if result.status_code:
            self._set_error(f"evidence upload failed ({result.status_code}): {_result_error_text(result)}")
        return SYNC_RETRY

    def _run_session(self, session: SessionSpec) -> None:
        try:
            if self.config.detector_mode == "front_runtime":
                self.front_runtime_runner(self, session)
            else:
                self._run_motion_session(session)
        except Exception as exc:  # pragma: no cover - runtime safety
            self._set_error(str(exc))
            self.logger.exception("session failed: session_id=%s", session.session_id)
            with self._lock:
                self._status = "error"
        finally:
            with self._lock:
                if self._status != "error":
                    self._status = "idle"
                self._session_thread = None

    def _run_motion_session(self, session: SessionSpec) -> None:
        builder = EvidenceBuilder(self.config)
        detector = MotionDetector(
            self.config.motion_threshold,
            self.config.motion_min_area_ratio,
            self.config.motion_cooldown_sec,
        )
        capture = self._open_capture()
        if capture is None or not capture.isOpened():
            raise RuntimeError("Could not open configured capture source.")

        with self._lock:
            self._status = "running"

        pending_sequences = []
        frame_counter = 0
        started_at = time.monotonic()

        try:
            while not self._shutdown.is_set() and not self._session_stop.is_set():
                ret, frame = capture.read()
                if not ret or frame is None:
                    if self.config.source_mode == "video" and self.config.video_path:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    time.sleep(0.05)
                    continue

                for sequence in list(pending_sequences):
                    if builder.advance_sequence(sequence, frame):
                        pending_sequences.remove(sequence)
                        self._finalize_incident(builder, sequence)

                event, metrics = detector.analyze(frame)
                if event:
                    pending_sequences.append(
                        builder.start_sequence(
                            session_id=session.session_id,
                            node_id=self.config.node_id,
                            camera_label=self.config.camera_label,
                            event=event,
                            frame=frame,
                        )
                    )
                    with self._lock:
                        self._banner_text = f"{event['type_label']} detected"
                        self._banner_expires = time.monotonic() + self.config.annotated_banner_ttl_sec

                builder.remember_frame(frame)

                frame_counter += 1
                elapsed = max(0.001, time.monotonic() - started_at)
                fps = frame_counter / elapsed
                with self._lock:
                    self._fps = fps
                banner = self._current_banner()
                annotated = annotate_frame(
                    frame,
                    node_name=self.config.display_name,
                    camera_label=self.config.camera_label,
                    session_id=session.session_id,
                    fps=fps,
                    metrics=metrics,
                    banner_text=banner,
                )
                self._publish_frames(frame, annotated)

            for sequence in list(pending_sequences):
                self._finalize_incident(builder, sequence)
        finally:
            capture.release()

    def _finalize_incident(self, builder: EvidenceBuilder, sequence) -> None:
        manifest, assets = builder.finalize_sequence(sequence)
        asset_payloads = []
        for asset in assets:
            if asset["asset_type"] == "manifest":
                continue
            asset_payloads.append(
                {
                    "asset_type": asset["asset_type"],
                    "file_path": str(asset["file_path"]),
                    "filename": asset["filename"],
                }
            )
        self.record_finalized_incident(manifest, asset_payloads)

    def mark_session_running(self) -> None:
        with self._lock:
            self._status = "running"
            self._last_error = ""

    def should_stop_requested(self) -> bool:
        return self._shutdown.is_set() or self._session_stop.is_set()

    def publish_detector_frames(
        self,
        raw_frame,
        annotated_frame,
        *,
        processing_fps: float | None = None,
    ) -> None:
        with self._lock:
            self._status = "running"
            if processing_fps is not None:
                self._fps = float(processing_fps)
        self._publish_frames(raw_frame, annotated_frame)

    def publish_preview_frames(self, raw_frame, annotated_frame=None) -> None:
        self._publish_frames(raw_frame, annotated_frame if annotated_frame is not None else raw_frame)

    def record_finalized_incident(
        self,
        manifest: IncidentManifest,
        assets: list[dict],
    ) -> None:
        self.sync_queue.enqueue(
            "manifest",
            manifest.incident_id,
            self.config.node_id,
            {"manifest_payload": manifest.to_dict()},
        )
        for asset in sorted(assets, key=_asset_sync_priority):
            if asset["asset_type"] == "frame":
                continue
            self.sync_queue.enqueue(
                "asset",
                manifest.incident_id,
                self.config.node_id,
                _asset_queue_payload(
                    asset,
                    manifest=manifest,
                ),
            )

        with self._lock:
            self._remember_incident_locked(manifest.incident_id)
        self._sync_wake.set()

    def record_detected_incident(self, manifest: IncidentManifest) -> None:
        self.sync_queue.enqueue(
            "manifest",
            manifest.incident_id,
            self.config.node_id,
            {"manifest_payload": manifest.to_dict()},
        )
        with self._lock:
            self._remember_incident_locked(manifest.incident_id)
        self._sync_wake.set()

    def _remember_incident_locked(self, incident_id: str) -> None:
        if not incident_id or incident_id in self._incident_ids:
            return
        self._incident_ids.add(incident_id)
        self._incident_count += 1

    def update_sound_telemetry(self, payload: dict) -> None:
        with self._lock:
            if "updated_at" not in payload:
                payload = {"updated_at": utc_now_iso(), **payload}
            self._sound_state.update(payload)

    def _publish_frames(self, raw_frame, annotated_frame) -> None:
        publish_interval = 1.0 / self.config.preview_fps
        now = time.monotonic()
        if now - self._last_publish_monotonic < publish_interval:
            return
        self._last_publish_monotonic = now

        raw_jpeg = self._encode_frame(raw_frame)
        annotated_jpeg = self._encode_frame(annotated_frame)
        with self._lock:
            self._raw_jpeg = raw_jpeg
            self._annotated_jpeg = annotated_jpeg
            self._raw_seq += 1
            self._annotated_seq += 1
            self._last_frame_at = utc_now_iso()

    def _clear_frames_locked(self) -> None:
        self._raw_jpeg = None
        self._annotated_jpeg = None
        self._raw_seq += 1
        self._annotated_seq += 1
        self._last_frame_at = ""

    def _encode_frame(self, frame):
        output = frame
        height, width = frame.shape[:2]
        if width > self.config.preview_width:
            scale = self.config.preview_width / float(width)
            output = cv2.resize(
                frame,
                (self.config.preview_width, max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        success, encoded = cv2.imencode(
            ".jpg",
            output,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        return encoded.tobytes() if success else None

    def stream_generator(self, mode: str):
        last_seq = -1
        while not self._shutdown.is_set():
            with self._lock:
                if mode == "raw":
                    jpeg_bytes = self._raw_jpeg
                    seq = self._raw_seq
                else:
                    jpeg_bytes = self._annotated_jpeg
                    seq = self._annotated_seq
            if jpeg_bytes is not None and seq != last_seq:
                last_seq = seq
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg_bytes
                    + b"\r\n"
                )
            else:
                time.sleep(0.03)

    def _open_capture(self):
        if self.config.source_mode == "video":
            if self.config.video_path is None:
                return None
            return cv2.VideoCapture(str(self.config.video_path))
        return cv2.VideoCapture(self.config.camera_index)

    def _current_banner(self) -> str:
        with self._lock:
            if time.monotonic() <= self._banner_expires:
                return self._banner_text
            self._banner_text = ""
            return ""

    def _auth_headers(self) -> dict[str, str]:
        return {
            "X-Node-Id": self.config.node_id,
            "X-Api-Key": self.config.api_key,
        }

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = str(message)


def _is_incident_not_found(result) -> bool:
    payload = result.json_data
    if isinstance(payload, dict):
        error_text = str(payload.get("error") or payload.get("message") or "")
    else:
        error_text = result.text
    return "incident not found" in error_text.lower()


def _queue_item_session_id(item: SyncQueueItem) -> str:
    payload = item.payload
    if item.item_type == "manifest":
        manifest = payload.get("manifest_payload")
        if isinstance(manifest, dict):
            return str(manifest.get("session_id", "")).strip()
    return str(payload.get("session_id", "")).strip()


def _queue_item_manifest_status(item: SyncQueueItem) -> str:
    if item.item_type != "manifest":
        return ""
    manifest = item.payload.get("manifest_payload")
    if not isinstance(manifest, dict):
        return ""
    return str(manifest.get("sync_status") or "").strip().lower()


def _asset_sync_priority(asset: dict) -> int:
    asset_type = str(asset.get("asset_type") or "").lower()
    if asset_type == "gif":
        return 0
    if asset_type == "poster":
        return 1
    return 2


def _asset_queue_payload(asset: dict, *, manifest: IncidentManifest) -> dict:
    payload = {
        "incident_id": manifest.incident_id,
        "session_id": manifest.session_id,
        "asset_type": asset["asset_type"],
        "file_path": str(asset["file_path"]),
        "filename": asset["filename"],
    }
    if "frame_paths" in asset:
        payload["frame_paths"] = [str(path) for path in (asset.get("frame_paths") or [])]
    return payload


def _asset_payload_type(payload: dict) -> str:
    return str(payload.get("asset_type") or "").strip().lower()


def _write_missing_gif_asset(payload: dict, gif_path: Path) -> bool:
    frame_paths = [
        Path(str(path))
        for path in (payload.get("frame_paths") or [])
        if str(path).strip()
    ]
    frame_paths = [path for path in frame_paths if path.exists() and path.is_file()]
    if not frame_paths:
        return False

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    if PIL_AVAILABLE and _write_gif_with_pillow(frame_paths, gif_path):
        return True
    return _write_gif_with_opencv(frame_paths, gif_path)


def _write_gif_with_pillow(frame_paths: list[Path], gif_path: Path) -> bool:
    try:
        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frames.append(image.convert("P", palette=Image.ADAPTIVE))
        if not frames:
            return False
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=220,
            loop=0,
            optimize=False,
        )
        return gif_path.exists()
    except Exception:
        return False


def _write_gif_with_opencv(frame_paths: list[Path], gif_path: Path) -> bool:
    try:
        if not hasattr(cv2, "Animation") or not hasattr(cv2, "imwriteanimation"):
            return False
        if hasattr(cv2, "haveImageWriter") and not cv2.haveImageWriter(".gif"):
            return False
        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                frames.append(frame)
        if not frames:
            return False
        animation = cv2.Animation()
        animation.frames = frames
        animation.durations = [220] * len(frames)
        animation.loop_count = 0
        return bool(cv2.imwriteanimation(str(gif_path), animation))
    except Exception:
        return False


def _result_error_text(result) -> str:
    payload = result.json_data
    if isinstance(payload, dict):
        return str(payload.get("error") or payload.get("message") or result.text or "").strip()
    return str(result.text or "").strip()
