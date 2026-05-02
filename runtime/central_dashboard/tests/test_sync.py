from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest
import sys

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.node_agent.config import NodeAgentConfig, load_node_agent_config
from central_dashboard.node_agent.state import NodeRuntime
from central_dashboard.node_agent.upload_worker import IncidentUploadJob, IncidentUploadWorker
from central_dashboard.shared.dto import IncidentManifest, SessionSpec
from central_dashboard.shared.http import HttpResult


class FakeHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post_json(self, url: str, payload: dict, *, headers=None, timeout=5.0):
        self.requests.append(("json", url, payload, timeout))
        if self.responses:
            return self.responses.pop(0)
        return HttpResult(200, {"ok": True}, '{"ok": true}')

    def post_file(
        self,
        url: str,
        fields: dict,
        *,
        file_field: str,
        file_path,
        filename: str,
        headers=None,
        timeout=5.0,
    ):
        self.requests.append(("file", url, dict(fields), Path(file_path).read_bytes(), timeout))
        if self.responses:
            return self.responses.pop(0)
        return HttpResult(200, {"ok": True}, '{"ok": true}')


class BlockingHttpClient(FakeHttpClient):
    def __init__(self):
        super().__init__([HttpResult(200, {"ok": True}, '{"ok": true}')])
        self.started = threading.Event()
        self.release = threading.Event()

    def post_json(self, url: str, payload: dict, *, headers=None, timeout=5.0):
        self.started.set()
        self.release.wait(timeout=2.0)
        return super().post_json(url, payload, headers=headers, timeout=timeout)


def build_config(tmpdir: Path) -> NodeAgentConfig:
    return NodeAgentConfig(
        config_path=tmpdir / "node.ini",
        node_id="front",
        display_name="Front Node",
        camera_label="Front Camera",
        profile="front",
        host="front.test",
        port=8091,
        api_key="front-key",
        central_base_url="http://central.test:8090",
        registration_interval_sec=10.0,
        heartbeat_interval_sec=10.0,
        http_timeout_sec=5.0,
        source_mode="webcam",
        camera_index=0,
        video_path=None,
        preview_width=640,
        preview_fps=15.0,
        jpeg_quality=75,
        detector_mode="motion",
        runtime_config_path=None,
        motion_threshold=5.0,
        motion_min_area_ratio=0.001,
        motion_cooldown_sec=0.05,
        annotated_banner_ttl_sec=1.0,
        evidence_root=tmpdir / "evidence",
        pre_event_frames=2,
        post_event_frames=2,
    )


def activate_runtime_session(runtime: NodeRuntime, session_id: str = "session-1") -> None:
    with runtime._lock:
        runtime._session = SessionSpec(session_id=session_id)
        runtime._status = "running"


def manifest(session_id: str = "session-1", incident_id: str = "incident-1") -> IncidentManifest:
    return IncidentManifest(
        incident_id=incident_id,
        session_id=session_id,
        node_id="front",
        camera_label="Front Camera",
        behavior_type="object",
        type_label="Using Phone",
        student_numbers=[5],
        created_at="2026-04-24T01:00:00Z",
        display_time="09:00 AM",
        frame_count=1,
        summary="Student #05 using phone detected",
        sync_status="ready",
        sync_attempts=0,
        asset_names=["poster.jpg"],
    )


class NodeRuntimeAsyncUploadTests(unittest.TestCase):
    def test_finalized_incident_enqueue_does_not_wait_for_http(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            http_client = BlockingHttpClient()
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)
            started_at = time.monotonic()

            runtime.record_finalized_incident(manifest(), [])

            elapsed = time.monotonic() - started_at
            self.assertLess(elapsed, 0.2)
            self.assertTrue(http_client.started.wait(timeout=1.0))
            http_client.release.set()
            self.assertTrue(runtime.upload_worker.wait_until_idle(timeout=2.0))
            self.assertEqual(len(http_client.requests), 1)
            runtime.close()

    def test_failed_manifest_upload_retries_once_then_counts_drop(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            http_client = FakeHttpClient(
                [
                    HttpResult(503, {"error": "down"}, "down"),
                    HttpResult(503, {"error": "down"}, "down"),
                ]
            )
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)

            runtime.record_detected_incident(replace(manifest(), sync_status="recording", asset_names=[]))

            self.assertTrue(runtime.upload_worker.wait_until_idle(timeout=2.0))
            heartbeat = runtime.heartbeat()
            self.assertEqual(len(http_client.requests), 2)
            self.assertEqual(heartbeat.sync_backlog, 0)
            self.assertEqual(heartbeat.extra["uploads"]["dropped_upload_count"], 1)
            self.assertIn("incident-1", heartbeat.extra["uploads"]["last_dropped_upload_error"])
            runtime.close()

    def test_stale_manifest_conflict_drops_without_retry(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            http_client = FakeHttpClient(
                [
                    HttpResult(
                        409,
                        {"error": "Stale incident upload rejected: session session-1 is not the active running session."},
                        '{"error": "stale"}',
                    )
                ]
            )
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)

            runtime.record_detected_incident(replace(manifest(), sync_status="recording", asset_names=[]))

            self.assertTrue(runtime.upload_worker.wait_until_idle(timeout=2.0))
            self.assertEqual(len(http_client.requests), 1)
            self.assertEqual(runtime.heartbeat().extra["uploads"]["dropped_upload_count"], 1)
            runtime.close()

    def test_finalized_incident_uploads_multipart_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            poster_path = tmpdir / "poster.jpg"
            poster_path.write_bytes(b"poster")
            http_client = FakeHttpClient([HttpResult(200, {"ok": True}, '{"ok": true}') for _ in range(2)])
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)

            runtime.record_finalized_incident(
                manifest(),
                [{"asset_type": "poster", "file_path": poster_path, "filename": "poster.jpg"}],
            )

            self.assertTrue(runtime.upload_worker.wait_until_idle(timeout=2.0))
            self.assertEqual([request[0] for request in http_client.requests], ["json", "file"])
            self.assertNotIn("content_base64", http_client.requests[1][2])
            self.assertEqual(http_client.requests[1][2]["asset_type"], "poster")
            runtime.close()

    def test_upload_queue_full_drops_new_jobs(self):
        drops = []
        client = BlockingHttpClient()
        worker = IncidentUploadWorker(
            node_id="front",
            central_base_url="http://central.test:8090",
            http_client=client,
            auth_headers=lambda: {},
            is_active_session=lambda session_id: True,
            set_error=lambda message: None,
            record_drop=lambda incident_id, item_type, reason: drops.append((incident_id, item_type, reason)),
            timeout_sec=5.0,
            logger=__import__("logging").getLogger(__name__),
            max_queue_size=1,
        )
        try:
            self.assertTrue(worker.enqueue(IncidentUploadJob(manifest=manifest(incident_id="one"))))
            self.assertTrue(client.started.wait(timeout=1.0))
            self.assertTrue(worker.enqueue(IncidentUploadJob(manifest=manifest(incident_id="two"))))
            self.assertFalse(worker.enqueue(IncidentUploadJob(manifest=manifest(incident_id="three"))))
            self.assertEqual(drops[0][0], "three")
            self.assertIn("full", drops[0][2])
        finally:
            client.release.set()
            worker.stop()

    def test_runtime_does_not_create_queue_sqlite_file(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            queue_path = tmpdir / "queue.sqlite3"
            config = build_config(tmpdir)
            runtime = NodeRuntime(config)
            try:
                self.assertFalse(queue_path.exists())
            finally:
                runtime.close()


class NodeRuntimePreviewQueueTests(unittest.TestCase):
    def test_preview_queue_keeps_newest_pending_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            runtime = NodeRuntime(build_config(Path(tmpdir_str)))
            runtime._shutdown.set()
            runtime._clear_preview_queue()
            first = np.zeros((4, 4, 3), dtype=np.uint8)
            latest = np.full((4, 4, 3), 255, dtype=np.uint8)

            runtime._queue_preview_frames(first, first)
            runtime._queue_preview_frames(latest, latest)

            queued_raw, _queued_annotated, _queued_debug = runtime._preview_queue.get_nowait()
            self.assertTrue(np.array_equal(queued_raw, latest))
            runtime._preview_queue.task_done()
            runtime.close()

    def test_lower_preview_defaults_load_when_preview_section_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config_path = tmpdir / "node.ini"
            config_path.write_text(
                """
[agent]
node_id = front
api_key = front-key
central_base_url = http://central.test:8090

[capture]
source_mode = webcam

[detector]
mode = motion
""".strip(),
                encoding="utf-8",
            )

            config = load_node_agent_config(config_path)

            self.assertEqual(config.preview_width, 640)
            self.assertEqual(config.preview_fps, 6.0)
            self.assertEqual(config.jpeg_quality, 60)


if __name__ == "__main__":
    unittest.main()
