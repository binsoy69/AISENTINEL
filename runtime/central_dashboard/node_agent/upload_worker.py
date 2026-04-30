"""Bounded asynchronous incident uploader for node agents."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Callable

from central_dashboard.shared.dto import IncidentManifest


SYNC_OK = "ok"
SYNC_RETRY = "retry"
SYNC_DROP = "drop"
UPLOAD_QUEUE_MAX_SIZE = 32


@dataclass(frozen=True, slots=True)
class IncidentUploadJob:
    manifest: IncidentManifest
    assets: tuple[dict, ...] = ()


class IncidentUploadWorker:
    """Uploads whole incident jobs without blocking the detector loop."""

    def __init__(
        self,
        *,
        node_id: str,
        central_base_url: str,
        http_client,
        auth_headers: Callable[[], dict[str, str]],
        is_active_session: Callable[[str], bool],
        set_error: Callable[[str], None],
        record_drop: Callable[[str, str, str], None],
        timeout_sec: float,
        logger: logging.Logger,
        max_queue_size: int = UPLOAD_QUEUE_MAX_SIZE,
    ) -> None:
        self.node_id = node_id
        self.central_base_url = central_base_url.rstrip("/")
        self.http_client = http_client
        self.auth_headers = auth_headers
        self.is_active_session = is_active_session
        self.set_error = set_error
        self.record_drop = record_drop
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.logger = logger
        self._error_lock = threading.Lock()
        self._last_error = ""
        self._queue: Queue[IncidentUploadJob | None] = Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"node-upload-{node_id}",
        )
        self._thread.start()

    def enqueue(self, job: IncidentUploadJob) -> bool:
        session_id = job.manifest.session_id
        incident_id = job.manifest.incident_id
        if not self.is_active_session(session_id):
            self.record_drop(
                incident_id,
                "incident",
                f"session {session_id or '-'} is not active",
            )
            return False
        try:
            self._queue.put_nowait(job)
            return True
        except Full:
            self.record_drop(incident_id, "incident", "upload queue is full")
            return False

    def backlog_count(self) -> int:
        return self._queue.qsize()

    def clear(self) -> int:
        cleared = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                return cleared
            self._queue.task_done()
            if item is not None:
                cleared += 1

    def wait_until_idle(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def stop(self) -> None:
        self._shutdown.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            self.clear()
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._shutdown.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if job is None:
                    return
                self._process_job(job)
            finally:
                self._queue.task_done()

    def _process_job(self, job: IncidentUploadJob) -> None:
        manifest = job.manifest
        if not self.is_active_session(manifest.session_id):
            self.record_drop(
                manifest.incident_id,
                "incident",
                f"session {manifest.session_id or '-'} is no longer active",
            )
            return

        if not self._attempt(
            manifest.incident_id,
            "manifest",
            lambda attempt: self._upload_manifest(manifest),
        ):
            return

        for asset in sorted(job.assets, key=_asset_upload_priority):
            if str(asset.get("asset_type") or "").lower() == "frame":
                continue
            if not self.is_active_session(manifest.session_id):
                self.record_drop(
                    manifest.incident_id,
                    str(asset.get("asset_type") or "asset"),
                    f"session {manifest.session_id or '-'} is no longer active",
                )
                return
            self._attempt(
                manifest.incident_id,
                str(asset.get("asset_type") or "asset"),
                lambda attempt, asset=asset: self._upload_asset(manifest, asset),
            )

    def _attempt(self, incident_id: str, item_type: str, upload_fn) -> bool:
        last_error = ""
        with self._error_lock:
            self._last_error = ""
        for attempt in (1, 2):
            try:
                disposition = upload_fn(attempt)
            except Exception as exc:  # pragma: no cover - runtime network safety
                disposition = SYNC_RETRY
                self._set_error(f"{item_type} upload failed: {exc}")
            last_error = self._last_error_text()
            if disposition == SYNC_OK:
                return True
            if disposition == SYNC_DROP:
                break
            if attempt == 1:
                self.logger.warning(
                    "upload attempt failed, retrying immediately: incident=%s type=%s reason=%s",
                    incident_id,
                    item_type,
                    last_error or "unknown upload failure",
                )
        self.record_drop(
            incident_id,
            item_type,
            last_error or "upload failed after two attempts",
        )
        return False

    def _upload_manifest(self, manifest: IncidentManifest) -> str:
        result = self.http_client.post_json(
            f"{self.central_base_url}/api/v1/incidents",
            manifest.to_dict(),
            headers=self.auth_headers(),
            timeout=self.timeout_sec,
        )
        if result.ok:
            return SYNC_OK
        if result.status_code == 409 and _is_stale_session_rejection(result):
            self._set_error(f"manifest upload dropped: {_result_error_text(result)}")
            return SYNC_DROP
        if result.status_code:
            self._set_error(
                f"manifest upload failed ({result.status_code}): {_result_error_text(result)}"
            )
        else:
            self._set_error(
                f"manifest upload retrying: {_result_error_text(result) or 'network timeout or dashboard unavailable'}"
            )
        return SYNC_RETRY

    def _upload_asset(self, manifest: IncidentManifest, asset: dict) -> str:
        asset_type = str(asset.get("asset_type") or "").strip().lower()
        file_path = Path(asset["file_path"])
        if not file_path.exists():
            self._set_error(f"evidence upload dropped: missing file {file_path}")
            return SYNC_DROP

        fields = {
            "incident_id": manifest.incident_id,
            "session_id": manifest.session_id,
            "node_id": self.node_id,
            "asset_type": asset_type,
            "filename": str(asset.get("filename") or file_path.name),
            "content_sha256": "",
            "size_bytes": str(file_path.stat().st_size),
        }
        result = self.http_client.post_file(
            f"{self.central_base_url}/api/v1/evidence/upload",
            fields,
            file_field="file",
            file_path=file_path,
            filename=fields["filename"],
            headers=self.auth_headers(),
            timeout=self.timeout_sec,
        )
        if result.ok:
            return SYNC_OK
        if result.status_code == 404 and _is_incident_not_found(result):
            self._set_error(f"evidence upload dropped: {_result_error_text(result)}")
            return SYNC_DROP
        if result.status_code == 409 and _is_stale_session_rejection(result):
            self._set_error(f"evidence upload dropped: {_result_error_text(result)}")
            return SYNC_DROP
        if result.status_code == 0:
            self._set_error(
                f"evidence upload retrying: {_result_error_text(result) or 'network timeout or dashboard unavailable'}"
            )
            return SYNC_RETRY
        if result.status_code:
            self._set_error(
                f"evidence upload failed ({result.status_code}): {_result_error_text(result)}"
            )
        return SYNC_RETRY

    def _set_error(self, message: str) -> None:
        with self._error_lock:
            self._last_error = str(message)
        self.set_error(message)

    def _last_error_text(self) -> str:
        with self._error_lock:
            return self._last_error


def _asset_upload_priority(asset: dict) -> int:
    asset_type = str(asset.get("asset_type") or "").lower()
    if asset_type == "poster":
        return 0
    if asset_type == "gif":
        return 1
    return 2


def _is_incident_not_found(result) -> bool:
    return "incident not found" in _result_error_text(result).lower()


def _is_stale_session_rejection(result) -> bool:
    error_text = _result_error_text(result).lower()
    return (
        "stale" in error_text
        or "active running session" in error_text
        or "not accepting node uploads" in error_text
    )


def _result_error_text(result) -> str:
    payload = result.json_data
    if isinstance(payload, dict):
        return str(payload.get("error") or payload.get("message") or result.text or "").strip()
    return str(result.text or "").strip()
