"""Runtime controller for the standalone node agent."""

from __future__ import annotations

import logging
from queue import Empty, Full, Queue
import shutil
import socket
import threading
import time
from urllib.parse import urlparse

import cv2

from central_dashboard.node_agent.detector import EvidenceBuilder, MotionDetector, annotate_frame
from central_dashboard.node_agent.front_runtime import run_front_runtime_session
from central_dashboard.node_agent.upload_worker import IncidentUploadJob, IncidentUploadWorker
from central_dashboard.shared.dto import (
    CommandAck,
    IncidentManifest,
    NodeDescriptor,
    NodeHeartbeat,
    SessionSpec,
    utc_now_iso,
)
from central_dashboard.shared.http import StdlibHttpClient
from edge_node_runtime.sound_monitor import DEFAULT_SOUND_SNAPSHOT


ACTIVE_UPLOAD_STATES = {"starting", "running"}


class NodeRuntime:
    """Supervises node registration, session execution, local feeds, and sync."""

    def __init__(self, config, *, http_client=None, front_runtime_runner=None) -> None:
        self.config = config
        self.logger = logging.getLogger(f"central_dashboard.node_agent.{config.node_id}")
        self.http_client = http_client or StdlibHttpClient()
        self.front_runtime_runner = front_runtime_runner or run_front_runtime_session
        self._closed = False
        self.config.evidence_root.mkdir(parents=True, exist_ok=True)
        self._cleanup_local_evidence_storage("node startup")

        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._session_stop = threading.Event()
        self._background_started = False
        self._session_thread: threading.Thread | None = None
        self._preview_queue: Queue[tuple[object, object, object] | None] = Queue(
            maxsize=1
        )

        self._raw_jpeg = None
        self._annotated_jpeg = None
        self._debug_jpeg = None
        self._raw_seq = 0
        self._annotated_seq = 0
        self._debug_seq = 0
        self._last_publish_monotonic = 0.0
        self._last_frame_at = ""

        self._status = "idle"
        self._session: SessionSpec | None = None
        self._fps = 0.0
        self._incident_count = 0
        self._incident_ids: set[str] = set()
        self._last_error = ""
        self._dropped_upload_count = 0
        self._last_dropped_upload_at = ""
        self._last_dropped_upload_error = ""
        self._banner_text = ""
        self._banner_expires = 0.0
        self._warmup_deadline_monotonic = 0.0
        self._warmup_total_sec = 0.0
        self._registration_ok = False
        self._sound_state = dict(DEFAULT_SOUND_SNAPSHOT)
        self.upload_worker = IncidentUploadWorker(
            node_id=self.config.node_id,
            central_base_url=self.config.central_base_url,
            http_client=self.http_client,
            auth_headers=self._auth_headers,
            is_active_session=self._is_active_session_id,
            set_error=self._set_error,
            record_drop=self._record_dropped_upload,
            timeout_sec=self.config.http_timeout_sec,
            logger=self.logger,
        )
        self._preview_thread = threading.Thread(
            target=self._preview_loop,
            daemon=True,
            name=f"node-preview-{self.config.node_id}",
        )
        self._preview_thread.start()

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
        ):
            thread = threading.Thread(target=target, daemon=True, name=name)
            thread.start()

    def shutdown(self) -> None:
        self.logger.info("shutdown requested")
        self._shutdown.set()
        self.stop_session()
        self._clear_preview_queue()
        try:
            self._preview_queue.put_nowait(None)
        except Full:
            self._clear_preview_queue()
            try:
                self._preview_queue.put_nowait(None)
            except Full:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.shutdown()
        self.upload_worker.stop()
        self._preview_thread.join(timeout=2.0)

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
            upload_state = {
                "dropped_upload_count": self._dropped_upload_count,
                "last_dropped_upload_at": self._last_dropped_upload_at,
                "last_dropped_upload_error": self._last_dropped_upload_error,
            }
            warmup_state = self._warmup_snapshot_locked(time.monotonic())
            stream_state = {
                "raw_seq": self._raw_seq,
                "annotated_seq": self._annotated_seq,
                "debug_seq": self._debug_seq,
                "has_raw_frame": self._raw_jpeg is not None,
                "has_annotated_frame": self._annotated_jpeg is not None,
                "has_debug_frame": self._debug_jpeg is not None,
                "last_frame_at": self._last_frame_at,
                "preview_fps": self.config.preview_fps,
            }
        return NodeHeartbeat(
            node_id=self.config.node_id,
            state=state,
            session_id=session_id,
            fps=fps,
            sync_backlog=self.upload_worker.backlog_count(),
            incident_count=incident_count,
            last_error=last_error,
            extra={
                "profile": self.config.profile,
                "detector_mode": self.config.detector_mode,
                "sound": dict(self._sound_state),
                "stream": stream_state,
                "uploads": upload_state,
                "warmup": warmup_state,
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
        self._clear_upload_backlog(f"session start accepted for {session.session_id}")
        self._cleanup_local_evidence_storage(f"session start accepted for {session.session_id}")
        warmup_delay_sec = self._configured_startup_detection_delay_sec()
        now = time.monotonic()
        with self._lock:
            self._session = session
            self._session_stop = threading.Event()
            self._status = "starting"
            self._last_error = ""
            self._fps = 0.0
            self._warmup_deadline_monotonic = now + warmup_delay_sec if warmup_delay_sec > 0 else 0.0
            self._warmup_total_sec = warmup_delay_sec
            self._clear_frames_locked()

        self.logger.info(
            "session start accepted: session_id=%s startup_detection_delay=%.1fs",
            session.session_id,
            warmup_delay_sec,
        )
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
            self._clear_detection_warmup_locked()
            self._clear_frames_locked()
        self._clear_upload_backlog("session stopped")
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
                    "ready" if (
                        stream.get("has_annotated_frame")
                        or stream.get("has_raw_frame")
                        or stream.get("has_debug_frame")
                    ) else "waiting",
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
        self.upload_worker.wait_until_idle(timeout=0.1)
        return 0

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
            if not self._session_stop.is_set():
                self.upload_worker.wait_until_idle(timeout=2.0)
            with self._lock:
                if self._status != "error":
                    self._status = "completed" if not self._session_stop.is_set() else "idle"
                self._session_thread = None
            self._clear_upload_backlog("session thread completed")

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

                warmup_remaining = self.detection_warmup_remaining_sec()
                if warmup_remaining > 0:
                    warmup_label = self.detection_warmup_label(warmup_remaining)
                    annotated = annotate_frame(
                        frame,
                        node_name=self.config.display_name,
                        camera_label=self.config.camera_label,
                        session_id=session.session_id,
                        fps=0.0,
                        metrics={},
                        banner_text=warmup_label,
                    )
                    self.publish_preview_frames(frame, annotated)
                    continue

                if frame_counter == 0:
                    self.mark_session_running()
                    started_at = time.monotonic()

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

    def detection_warmup_remaining_sec(self) -> float:
        with self._lock:
            return self._warmup_remaining_locked(time.monotonic())

    def detection_warmup_active(self) -> bool:
        return self.detection_warmup_remaining_sec() > 0

    def detection_warmup_label(self, remaining_sec: float | None = None) -> str:
        remaining = self.detection_warmup_remaining_sec() if remaining_sec is None else remaining_sec
        return f"Warming up {max(1, int(remaining + 0.999))}s"

    def should_stop_requested(self) -> bool:
        return self._shutdown.is_set() or self._session_stop.is_set()

    def publish_detector_frames(
        self,
        raw_frame,
        annotated_frame,
        *,
        processing_fps: float | None = None,
        debug_frame=None,
    ) -> None:
        with self._lock:
            self._status = "running"
            if processing_fps is not None:
                self._fps = float(processing_fps)
        self._publish_frames(raw_frame, annotated_frame, debug_frame)

    def publish_preview_frames(
        self,
        raw_frame,
        annotated_frame=None,
        debug_frame=None,
    ) -> None:
        display_frame = annotated_frame if annotated_frame is not None else raw_frame
        self._publish_frames(
            raw_frame,
            display_frame,
            debug_frame if debug_frame is not None else display_frame,
        )

    def record_finalized_incident(
        self,
        manifest: IncidentManifest,
        assets: list[dict],
    ) -> None:
        if self.detection_warmup_active():
            self.logger.info(
                "ignored finalized incident during startup warmup: incident=%s session=%s",
                manifest.incident_id,
                manifest.session_id,
            )
            return
        if not self._incident_matches_active_session(manifest):
            self.logger.info(
                "ignored finalized incident outside active session: incident=%s session=%s",
                manifest.incident_id,
                manifest.session_id,
            )
            return
        with self._lock:
            self._remember_incident_locked(manifest.incident_id)
        self.upload_worker.enqueue(
            IncidentUploadJob(
                manifest=manifest,
                assets=tuple(asset for asset in assets if asset.get("asset_type") != "frame"),
            )
        )

    def record_detected_incident(self, manifest: IncidentManifest) -> None:
        if self.detection_warmup_active():
            self.logger.info(
                "ignored detected incident during startup warmup: incident=%s session=%s",
                manifest.incident_id,
                manifest.session_id,
            )
            return
        if not self._incident_matches_active_session(manifest):
            self.logger.info(
                "ignored detected incident outside active session: incident=%s session=%s",
                manifest.incident_id,
                manifest.session_id,
            )
            return
        with self._lock:
            self._remember_incident_locked(manifest.incident_id)
        self.upload_worker.enqueue(IncidentUploadJob(manifest=manifest, assets=()))

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

    def _publish_frames(self, raw_frame, annotated_frame, debug_frame=None) -> None:
        publish_interval = 1.0 / self.config.preview_fps
        now = time.monotonic()
        if now - self._last_publish_monotonic < publish_interval:
            return
        self._last_publish_monotonic = now

        self._queue_preview_frames(
            raw_frame,
            annotated_frame,
            debug_frame if debug_frame is not None else annotated_frame,
        )

    def _queue_preview_frames(self, raw_frame, annotated_frame, debug_frame=None) -> None:
        item = (
            raw_frame,
            annotated_frame,
            debug_frame if debug_frame is not None else annotated_frame,
        )
        try:
            self._preview_queue.put_nowait(item)
        except Full:
            try:
                self._preview_queue.get_nowait()
                self._preview_queue.task_done()
            except Empty:
                pass
            try:
                self._preview_queue.put_nowait(item)
            except Full:
                pass

    def _preview_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._preview_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if item is None:
                    return
                raw_frame, annotated_frame, debug_frame = item
                raw_jpeg = self._encode_frame(raw_frame)
                annotated_jpeg = self._encode_frame(annotated_frame)
                debug_jpeg = self._encode_frame(debug_frame)
                with self._lock:
                    self._raw_jpeg = raw_jpeg
                    self._annotated_jpeg = annotated_jpeg
                    self._debug_jpeg = debug_jpeg
                    self._raw_seq += 1
                    self._annotated_seq += 1
                    self._debug_seq += 1
                    self._last_frame_at = utc_now_iso()
            finally:
                self._preview_queue.task_done()

    def _clear_preview_queue(self) -> int:
        cleared = 0
        while True:
            try:
                self._preview_queue.get_nowait()
            except Empty:
                return cleared
            self._preview_queue.task_done()
            cleared += 1

    def _clear_frames_locked(self) -> None:
        self._raw_jpeg = None
        self._annotated_jpeg = None
        self._debug_jpeg = None
        self._raw_seq += 1
        self._annotated_seq += 1
        self._debug_seq += 1
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
                elif mode == "debug":
                    jpeg_bytes = self._debug_jpeg
                    seq = self._debug_seq
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

    def _configured_startup_detection_delay_sec(self) -> float:
        source_mode = str(self.config.source_mode or "").strip().lower()
        if source_mode != "webcam":
            return 0.0
        return max(0.0, float(getattr(self.config, "startup_detection_delay_sec", 0.0) or 0.0))

    def _warmup_remaining_locked(self, now_monotonic: float) -> float:
        if self._session is None or self._warmup_deadline_monotonic <= 0:
            return 0.0
        return max(0.0, self._warmup_deadline_monotonic - now_monotonic)

    def _warmup_snapshot_locked(self, now_monotonic: float) -> dict:
        remaining = self._warmup_remaining_locked(now_monotonic)
        active = remaining > 0 and self._status in {"starting", "running"}
        return {
            "active": active,
            "remaining_sec": round(remaining, 3),
            "total_sec": round(max(0.0, self._warmup_total_sec), 3),
            "source_mode": self.config.source_mode,
        }

    def _clear_detection_warmup_locked(self) -> None:
        self._warmup_deadline_monotonic = 0.0
        self._warmup_total_sec = 0.0

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

    def _record_dropped_upload(self, incident_id: str, item_type: str, reason: str) -> None:
        message = f"{item_type} upload dropped for incident {incident_id}: {reason}"
        now_iso = utc_now_iso()
        with self._lock:
            self._dropped_upload_count += 1
            self._last_dropped_upload_at = now_iso
            self._last_dropped_upload_error = message
            self._last_error = message
        self.logger.warning(message)

    def _active_upload_session_id(self) -> str:
        with self._lock:
            if self._session and self._status in ACTIVE_UPLOAD_STATES:
                return self._session.session_id
        return ""

    def _is_active_session_id(self, session_id: str) -> bool:
        return bool(session_id and session_id == self._active_upload_session_id())

    def _incident_matches_active_session(self, manifest: IncidentManifest) -> bool:
        active_session_id = self._active_upload_session_id()
        return bool(active_session_id and manifest.session_id == active_session_id)

    def _clear_upload_backlog(self, reason: str) -> int:
        purged = self.upload_worker.clear()
        if purged:
            self.logger.info(
                "cleared %s pending upload job(s): %s",
                purged,
                reason,
            )
        return purged

    def _cleanup_local_evidence_storage(self, reason: str) -> int:
        root = self.config.evidence_root
        if not root.exists():
            return 0
        removed = 0
        for child in root.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed += 1
            except OSError as exc:
                self.logger.warning("could not clean local evidence path %s: %s", child, exc)
        if removed:
            self.logger.info(
                "cleaned %s local evidence item(s): %s",
                removed,
                reason,
            )
        return removed
