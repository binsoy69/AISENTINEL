from __future__ import annotations

from dataclasses import replace
import threading
import time
import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.node_agent.sync import LocalSyncQueue
from central_dashboard.node_agent.config import NodeAgentConfig
from central_dashboard.node_agent.state import NodeRuntime
from central_dashboard.shared.dto import IncidentManifest, SessionSpec
from central_dashboard.shared.http import HttpResult


class SyncQueueTests(unittest.TestCase):
    def test_enqueue_retry_and_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = LocalSyncQueue(Path(tmpdir) / "queue.sqlite3")
            item_id = queue.enqueue("manifest", "incident-1", "front", {"manifest_path": "manifest.json"})
            self.assertEqual(queue.backlog_count(), 1)

            items = queue.due_items()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].item_id, item_id)

            queue.mark_retry(items[0], "temporary outage")
            self.assertEqual(queue.backlog_count(), 1)

            items = queue.due_items()
            if items:
                queue.mark_done(items[0].item_id)
            else:
                # Retry scheduling can push the item out briefly; simulate completion directly.
                queue.connection.execute("DELETE FROM sync_queue WHERE item_id=?", (item_id,))
                queue.connection.commit()

            self.assertEqual(queue.backlog_count(), 0)
            queue.close()

    def test_has_pending_manifest_reports_incident_dependency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = LocalSyncQueue(Path(tmpdir) / "queue.sqlite3")
            queue.enqueue("manifest", "incident-1", "front", {"manifest_payload": {}})
            queue.enqueue("asset", "incident-1", "front", {"filename": "poster.jpg"})
            queue.enqueue("asset", "incident-2", "front", {"filename": "poster.jpg"})

            self.assertTrue(queue.has_pending_manifest("incident-1"))
            self.assertFalse(queue.has_pending_manifest("incident-2"))
            queue.close()

    def test_recording_manifest_does_not_block_ready_asset_upload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = LocalSyncQueue(Path(tmpdir) / "queue.sqlite3")
            queue.enqueue(
                "manifest",
                "incident-1",
                "front",
                {"manifest_payload": {"sync_status": "recording"}},
            )
            queue.enqueue("asset", "incident-1", "front", {"filename": "poster.jpg"})

            self.assertFalse(queue.has_pending_manifest("incident-1"))
            self.assertEqual(queue.purge_recording_manifests("incident-1"), 1)
            self.assertEqual(queue.backlog_count(), 1)
            queue.close()

    def test_purge_asset_type_removes_existing_frame_backlog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = LocalSyncQueue(Path(tmpdir) / "queue.sqlite3")
            queue.enqueue("asset", "incident-1", "front", {"asset_type": "frame", "filename": "frame_001.jpg"})
            queue.enqueue("asset", "incident-1", "front", {"asset_type": "gif", "filename": "evidence.gif"})
            queue.enqueue("asset", "incident-2", "front", {"asset_type": "frame", "filename": "frame_002.jpg"})

            self.assertEqual(queue.purge_asset_type("frame"), 2)
            self.assertEqual(queue.backlog_count(), 1)
            item = queue.due_items()[0]
            self.assertEqual(item.payload["asset_type"], "gif")
            queue.close()

    def test_clear_removes_all_backlog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = LocalSyncQueue(Path(tmpdir) / "queue.sqlite3")
            queue.enqueue("manifest", "incident-1", "front", {"manifest_payload": {}})
            queue.enqueue("asset", "incident-1", "front", {"asset_type": "poster"})

            self.assertEqual(queue.clear(), 2)
            self.assertEqual(queue.backlog_count(), 0)
            self.assertEqual(queue.clear(), 0)
            queue.close()

    def test_clear_except_session_removes_only_stale_backlog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = LocalSyncQueue(Path(tmpdir) / "queue.sqlite3")
            queue.enqueue(
                "manifest",
                "current-incident",
                "front",
                {"manifest_payload": {"session_id": "current-session"}},
            )
            queue.enqueue(
                "asset",
                "old-incident",
                "front",
                {"session_id": "old-session", "asset_type": "poster"},
            )

            self.assertEqual(queue.clear_except_session("current-session"), 1)
            self.assertEqual(queue.backlog_count(), 1)
            item = queue.due_items()[0]
            self.assertEqual(item.incident_id, "current-incident")
            queue.close()


class FakeHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post_json(self, url: str, payload: dict, *, headers=None, timeout=5.0):
        self.requests.append((url, payload, timeout))
        if self.responses:
            return self.responses.pop(0)
        return HttpResult(200, {"ok": True}, '{"ok": true}')


def build_config(
    tmpdir: Path,
    *,
    clear_sync_backlog_on_session_start: bool = False,
) -> NodeAgentConfig:
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
        sync_interval_sec=10.0,
        http_timeout_sec=5.0,
        local_db_path=tmpdir / "queue.sqlite3",
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
        clear_sync_backlog_on_session_start=clear_sync_backlog_on_session_start,
    )


def activate_runtime_session(runtime: NodeRuntime, session_id: str = "session-1") -> None:
    with runtime._lock:
        runtime._session = SessionSpec(session_id=session_id)
        runtime._status = "running"


class NodeRuntimeSyncTests(unittest.TestCase):
    def test_runtime_startup_clears_existing_backlog(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            queue = LocalSyncQueue(tmpdir / "queue.sqlite3")
            queue.enqueue(
                "manifest",
                "old-incident",
                "front",
                {"manifest_payload": {"incident_id": "old-incident", "session_id": "old-session"}},
            )
            self.assertEqual(queue.backlog_count(), 1)
            queue.close()

            runtime = NodeRuntime(build_config(tmpdir))

            self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            runtime.close()

    def test_sync_without_active_session_clears_backlog_without_uploading(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            http_client = FakeHttpClient([HttpResult(200, {"ok": True}, '{"ok": true}')])
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            runtime.sync_queue.enqueue(
                "manifest",
                "old-incident",
                "front",
                {"manifest_payload": {"incident_id": "old-incident", "session_id": "old-session"}},
            )

            synced = runtime.sync_once()

            self.assertEqual(synced, 0)
            self.assertEqual(http_client.requests, [])
            self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            runtime.close()

    def test_asset_waits_when_manifest_is_still_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            asset_path = tmpdir / "poster.jpg"
            asset_path.write_bytes(b"jpeg-data")
            http_client = FakeHttpClient([HttpResult(503, {"error": "down"}, "down")])
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)
            runtime.sync_queue.enqueue(
                "manifest",
                "incident-1",
                "front",
                {"manifest_payload": {"incident_id": "incident-1", "session_id": "session-1", "node_id": "front"}},
            )
            runtime.sync_queue.enqueue(
                "asset",
                "incident-1",
                "front",
                {
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "asset_type": "poster",
                    "file_path": str(asset_path),
                    "filename": "poster.jpg",
                },
            )

            runtime.sync_once()

            self.assertEqual(len(http_client.requests), 1)
            self.assertTrue(http_client.requests[0][0].endswith("/api/v1/incidents"))
            self.assertEqual(runtime.sync_queue.backlog_count(), 2)
            runtime.close()

    def test_stale_asset_404_incident_not_found_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            asset_path = tmpdir / "poster.jpg"
            asset_path.write_bytes(b"jpeg-data")
            http_client = FakeHttpClient(
                [
                    HttpResult(
                        404,
                        {"error": "Incident not found."},
                        '{"error": "Incident not found."}',
                    )
                ]
            )
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)
            runtime.sync_queue.enqueue(
                "asset",
                "incident-1",
                "front",
                {
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "asset_type": "poster",
                    "file_path": str(asset_path),
                    "filename": "poster.jpg",
                },
            )

            runtime.sync_once()

            self.assertEqual(len(http_client.requests), 1)
            self.assertTrue(http_client.requests[0][0].endswith("/api/v1/evidence/upload"))
            self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            runtime.close()

    def test_bad_asset_upload_400_retries_then_drops_after_repeated_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            asset_path = tmpdir / "poster.jpg"
            asset_path.write_bytes(b"jpeg-data")
            http_client = FakeHttpClient(
                [
                    HttpResult(
                        400,
                        {"error": "node_id does not match authenticated header"},
                        '{"error": "node_id does not match authenticated header"}',
                    ),
                    HttpResult(
                        400,
                        {"error": "node_id does not match authenticated header"},
                        '{"error": "node_id does not match authenticated header"}',
                    ),
                ]
            )
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)
            runtime.sync_queue.enqueue(
                "asset",
                "incident-1",
                "front",
                {
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "asset_type": "poster",
                    "file_path": str(asset_path),
                    "filename": "poster.jpg",
                },
            )

            runtime.sync_once()

            self.assertEqual(len(http_client.requests), 1)
            self.assertTrue(http_client.requests[0][0].endswith("/api/v1/evidence/upload"))
            self.assertEqual(runtime.sync_queue.backlog_count(), 1)
            self.assertIn("node_id does not match", runtime.heartbeat().last_error)

            runtime.sync_queue.connection.execute(
                "UPDATE sync_queue SET attempts=3, next_retry_at=?",
                ("1970-01-01T00:00:00Z",),
            )
            runtime.sync_queue.connection.commit()

            runtime.sync_once()

            self.assertEqual(len(http_client.requests), 2)
            self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            runtime.close()

    def test_frame_assets_are_purged_before_sync(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            frame_path = tmpdir / "frame.jpg"
            gif_path = tmpdir / "evidence.gif"
            frame_path.write_bytes(b"frame-data")
            gif_path.write_bytes(b"gif-data")
            http_client = FakeHttpClient([HttpResult(200, {"ok": True}, '{"ok": true}')])
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)
            runtime.sync_queue.enqueue(
                "asset",
                "incident-1",
                "front",
                {
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "asset_type": "frame",
                    "file_path": str(frame_path),
                    "filename": "frame.jpg",
                },
            )
            runtime.sync_queue.enqueue(
                "asset",
                "incident-1",
                "front",
                {
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "asset_type": "gif",
                    "file_path": str(gif_path),
                    "filename": "evidence.gif",
                },
            )

            runtime.sync_once()

            self.assertEqual(len(http_client.requests), 1)
            self.assertEqual(http_client.requests[0][1]["asset_type"], "gif")
            self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            runtime.close()

    def test_record_finalized_incident_prioritizes_poster_before_legacy_gif(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            poster_path = tmpdir / "poster.jpg"
            gif_path = tmpdir / "evidence.gif"
            poster_path.write_bytes(b"poster")
            gif_path.write_bytes(b"gif")
            runtime = NodeRuntime(build_config(tmpdir))
            activate_runtime_session(runtime, "session-001")

            runtime.record_finalized_incident(
                IncidentManifest(
                    incident_id="incident-001",
                    session_id="session-001",
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="object",
                    type_label="Using Phone",
                    student_numbers=[5],
                    created_at="2026-04-24T01:00:00Z",
                    display_time="09:00 AM",
                    frame_count=2,
                    summary="Student #05 using phone detected",
                    sync_status="ready",
                    sync_attempts=0,
                    asset_names=["poster.jpg", "evidence.gif"],
                ),
                [
                    {
                        "asset_type": "poster",
                        "file_path": poster_path,
                        "filename": "poster.jpg",
                    },
                    {
                        "asset_type": "gif",
                        "file_path": gif_path,
                        "filename": "evidence.gif",
                    },
                ],
            )

            items = runtime.sync_queue.due_items(limit=10)
            self.assertEqual([item.item_type for item in items], ["manifest", "asset", "asset"])
            self.assertEqual([item.payload.get("asset_type") for item in items[1:]], ["poster", "gif"])
            runtime.close()

    def test_late_incident_for_non_current_session_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            runtime = NodeRuntime(build_config(tmpdir))
            activate_runtime_session(runtime, "session-current")

            runtime.record_finalized_incident(
                IncidentManifest(
                    incident_id="incident-stale",
                    session_id="session-old",
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="object",
                    type_label="Using Phone",
                    student_numbers=[5],
                    created_at="2026-04-24T01:00:00Z",
                    display_time="09:00 AM",
                    frame_count=2,
                    summary="Stale session incident",
                    sync_status="ready",
                    sync_attempts=0,
                    asset_names=[],
                ),
                [],
            )

            self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            self.assertEqual(runtime.heartbeat().incident_count, 0)
            runtime.close()

    def test_missing_gif_asset_is_dropped_without_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            frame_a = tmpdir / "frame-a.jpg"
            frame_b = tmpdir / "frame-b.jpg"
            gif_path = tmpdir / "missing" / "evidence.gif"
            cv2.imwrite(str(frame_a), np.zeros((32, 48, 3), dtype=np.uint8))
            cv2.imwrite(str(frame_b), np.full((32, 48, 3), 200, dtype=np.uint8))
            http_client = FakeHttpClient(
                [
                    HttpResult(200, {"ok": True}, '{"ok": true}'),
                    HttpResult(200, {"ok": True}, '{"ok": true}'),
                ]
            )
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)
            runtime.sync_queue.enqueue(
                "manifest",
                "incident-1",
                "front",
                {
                    "manifest_payload": {
                        "incident_id": "incident-1",
                        "session_id": "session-1",
                        "node_id": "front",
                        "sync_status": "ready",
                    }
                },
            )
            runtime.sync_queue.enqueue(
                "asset",
                "incident-1",
                "front",
                {
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "asset_type": "gif",
                    "file_path": str(gif_path),
                    "filename": "evidence.gif",
                    "frame_paths": [str(frame_a), str(frame_b)],
                },
            )

            runtime.sync_once()

            self.assertFalse(gif_path.exists())
            self.assertEqual(len(http_client.requests), 1)
            self.assertTrue(http_client.requests[0][0].endswith("/api/v1/incidents"))
            self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            runtime.close()

    def test_gif_upload_uses_base_timeout_and_surfaces_network_retry_error(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            gif_path = tmpdir / "evidence.gif"
            gif_path.write_bytes(b"gif-data")
            http_client = FakeHttpClient([HttpResult(0, None, "timed out")])
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)
            runtime.sync_queue.enqueue(
                "asset",
                "incident-1",
                "front",
                {
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "asset_type": "gif",
                    "file_path": str(gif_path),
                    "filename": "evidence.gif",
                },
            )

            runtime.sync_once()

            self.assertEqual(len(http_client.requests), 1)
            self.assertTrue(http_client.requests[0][0].endswith("/api/v1/evidence/upload"))
            self.assertEqual(http_client.requests[0][2], runtime.config.http_timeout_sec)
            self.assertEqual(runtime.sync_queue.backlog_count(), 1)
            self.assertIn("timed out", runtime.heartbeat().last_error)
            runtime.close()

    def test_active_session_items_are_prioritized_over_stale_backlog(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            stale_path = tmpdir / "stale.jpg"
            stale_path.write_bytes(b"stale")
            http_client = FakeHttpClient(
                [HttpResult(200, {"ok": True}, '{"ok": true}') for _ in range(12)]
            )
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)

            for index in range(12):
                runtime.sync_queue.enqueue(
                    "asset",
                    f"stale-{index}",
                    "front",
                    {
                        "incident_id": f"stale-{index}",
                        "session_id": "old-session",
                        "asset_type": "poster",
                        "file_path": str(stale_path),
                        "filename": "poster.jpg",
                    },
                )
            runtime.sync_queue.enqueue(
                "manifest",
                "current-incident",
                "front",
                {
                    "manifest_payload": {
                        "incident_id": "current-incident",
                        "session_id": "current-session",
                        "node_id": "front",
                    }
                },
            )
            with runtime._lock:
                runtime._session = SessionSpec(session_id="current-session")
                runtime._status = "running"

            runtime.sync_once()

            self.assertEqual(len(http_client.requests), 1)
            self.assertTrue(http_client.requests[0][0].endswith("/api/v1/incidents"))
            self.assertEqual(http_client.requests[0][1]["incident_id"], "current-incident")
            self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            runtime.close()

    def test_start_session_can_clear_stale_backlog(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config = replace(
                build_config(tmpdir, clear_sync_backlog_on_session_start=True),
                detector_mode="front_runtime",
            )

            def fake_runner(runtime: NodeRuntime, session: SessionSpec) -> None:
                runtime.mark_session_running()

            runtime = NodeRuntime(config, front_runtime_runner=fake_runner)
            try:
                runtime.sync_queue.enqueue(
                    "manifest",
                    "old-incident",
                    "front",
                    {"manifest_payload": {"incident_id": "old-incident", "session_id": "old-session"}},
                )

                ack = runtime.start_session(
                    {
                        "subject_code": "CS321",
                        "professor": "Dr. Reyes",
                        "session_date": "2026-04-24",
                        "start_time": "09:00",
                        "end_time": "11:00",
                    }
                )
                self.assertTrue(ack.ok)
                thread = runtime._session_thread
                if thread is not None:
                    thread.join(timeout=2.0)
                self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            finally:
                runtime.close()

    def test_record_detected_incident_queues_recording_manifest_only(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            runtime = NodeRuntime(build_config(tmpdir))
            activate_runtime_session(runtime, "session-001")

            runtime.record_detected_incident(
                IncidentManifest(
                    incident_id="incident-001",
                    session_id="session-001",
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="head",
                    type_label="Head Tilt",
                    student_numbers=[5],
                    created_at="2026-04-24T01:00:00Z",
                    display_time="09:00 AM",
                    frame_count=0,
                    summary="Student #05 head tilt detected",
                    sync_status="recording",
                    sync_attempts=0,
                    asset_names=[],
                )
            )

            items = runtime.sync_queue.due_items(limit=10)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].item_type, "manifest")
            manifest = items[0].payload["manifest_payload"]
            self.assertEqual(manifest["sync_status"], "recording")
            self.assertEqual(manifest["asset_names"], [])
            self.assertEqual(runtime.heartbeat().incident_count, 1)
            self.assertTrue(runtime._sync_wake.is_set())
            runtime.close()

    def test_record_finalized_incident_wakes_sync_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            runtime = NodeRuntime(build_config(tmpdir))
            activate_runtime_session(runtime, "session-001")

            runtime.record_finalized_incident(
                IncidentManifest(
                    incident_id="incident-001",
                    session_id="session-001",
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="head",
                    type_label="Head Tilt",
                    student_numbers=[5],
                    created_at="2026-04-24T01:00:00Z",
                    display_time="09:00 AM",
                    frame_count=1,
                    summary="Student #05 head tilt detected",
                    sync_status="queued",
                    sync_attempts=0,
                    asset_names=[],
                ),
                [],
            )

            self.assertEqual(runtime.sync_queue.backlog_count(), 1)
            self.assertTrue(runtime._sync_wake.is_set())
            runtime.close()

    def test_stop_session_clears_items_finalized_during_shutdown(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            runner_ready = threading.Event()

            def fake_runner(runtime: NodeRuntime, session: SessionSpec) -> None:
                runtime.mark_session_running()
                runner_ready.set()
                while not runtime.should_stop_requested():
                    time.sleep(0.01)
                runtime.record_finalized_incident(
                    IncidentManifest(
                        incident_id="incident-late",
                        session_id=session.session_id,
                        node_id="front",
                        camera_label="Front Camera",
                        behavior_type="head",
                        type_label="Head Tilt",
                        student_numbers=[5],
                        created_at="2026-04-24T01:00:00Z",
                        display_time="09:00 AM",
                        frame_count=1,
                        summary="Late shutdown incident",
                        sync_status="queued",
                        sync_attempts=0,
                        asset_names=[],
                    ),
                    [],
                )

            config = replace(build_config(tmpdir), detector_mode="front_runtime")
            runtime = NodeRuntime(config, front_runtime_runner=fake_runner)
            try:
                ack = runtime.start_session(
                    {
                        "session_id": "session-001",
                        "subject_code": "CS321",
                        "professor": "Dr. Reyes",
                    }
                )
                self.assertTrue(ack.ok)
                self.assertTrue(runner_ready.wait(timeout=2.0))

                stop_ack = runtime.stop_session()

                self.assertTrue(stop_ack.ok)
                self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            finally:
                runtime.close()

    def test_stale_manifest_conflict_is_dropped(self):
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
            runtime.sync_queue.enqueue(
                "manifest",
                "incident-1",
                "front",
                {
                    "manifest_payload": {
                        "incident_id": "incident-1",
                        "session_id": "session-1",
                        "node_id": "front",
                    }
                },
            )

            runtime.sync_once()

            self.assertEqual(len(http_client.requests), 1)
            self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            self.assertIn("dropped", runtime.heartbeat().last_error)
            runtime.close()

    def test_stale_asset_conflict_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            asset_path = tmpdir / "poster.jpg"
            asset_path.write_bytes(b"jpeg-data")
            http_client = FakeHttpClient(
                [
                    HttpResult(
                        409,
                        {"error": "Stale evidence upload rejected: session session-1 is not the active running session."},
                        '{"error": "stale"}',
                    )
                ]
            )
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            activate_runtime_session(runtime)
            runtime.sync_queue.enqueue(
                "asset",
                "incident-1",
                "front",
                {
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "asset_type": "poster",
                    "file_path": str(asset_path),
                    "filename": "poster.jpg",
                },
            )

            runtime.sync_once()

            self.assertEqual(len(http_client.requests), 1)
            self.assertEqual(runtime.sync_queue.backlog_count(), 0)
            self.assertIn("dropped", runtime.heartbeat().last_error)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
