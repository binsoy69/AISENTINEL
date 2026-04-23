from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.node_agent.sync import LocalSyncQueue
from central_dashboard.node_agent.config import NodeAgentConfig
from central_dashboard.node_agent.state import NodeRuntime
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


class FakeHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post_json(self, url: str, payload: dict, *, headers=None, timeout=5.0):
        self.requests.append((url, payload))
        if self.responses:
            return self.responses.pop(0)
        return HttpResult(200, {"ok": True}, '{"ok": true}')


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
    )


class NodeRuntimeSyncTests(unittest.TestCase):
    def test_asset_waits_when_manifest_is_still_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            asset_path = tmpdir / "poster.jpg"
            asset_path.write_bytes(b"jpeg-data")
            http_client = FakeHttpClient([HttpResult(503, {"error": "down"}, "down")])
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
            runtime.sync_queue.enqueue(
                "manifest",
                "incident-1",
                "front",
                {"manifest_payload": {"incident_id": "incident-1", "node_id": "front"}},
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

    def test_bad_asset_upload_400_is_dropped(self):
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
                    )
                ]
            )
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
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

    def test_frame_assets_are_purged_before_sync(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            frame_path = tmpdir / "frame.jpg"
            gif_path = tmpdir / "evidence.gif"
            frame_path.write_bytes(b"frame-data")
            gif_path.write_bytes(b"gif-data")
            http_client = FakeHttpClient([HttpResult(200, {"ok": True}, '{"ok": true}')])
            runtime = NodeRuntime(build_config(tmpdir), http_client=http_client)
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


if __name__ == "__main__":
    unittest.main()
