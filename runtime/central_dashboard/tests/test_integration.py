from __future__ import annotations

import base64
from dataclasses import replace
import tempfile
import time
import unittest
from pathlib import Path
import sys

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.central_service.app import create_app as create_central_app
from central_dashboard.central_service.config import BrowserAuthConfig, CentralServiceConfig, KnownNodeConfig
from central_dashboard.central_service.db import connect_db, init_db
from central_dashboard.central_service.repositories import CentralRepository
from central_dashboard.central_service.services import CentralServiceManager
from central_dashboard.node_agent.app import create_app as create_node_app
from central_dashboard.node_agent.config import NodeAgentConfig
from central_dashboard.node_agent.state import NodeRuntime
from central_dashboard.shared.dto import IncidentManifest, NodeDescriptor
from central_dashboard.shared.http import HttpResult

try:
    from .test_support import InProcessHttpClient
except ImportError:  # pragma: no cover - direct script execution
    from test_support import InProcessHttpClient


class FakeCapture:
    def __init__(self, frames):
        self.frames = [frame.copy() for frame in frames]
        self.index = 0
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        frame = self.frames[self.index % len(self.frames)].copy()
        self.index += 1
        time.sleep(0.01)
        return True, frame

    def release(self):
        self.opened = False

    def set(self, prop, value):
        self.index = 0


class PartialStopHttpClient:
    def post_json(self, url: str, payload: dict, *, headers=None, timeout=5.0) -> HttpResult:
        action = str(payload.get("action") or "")
        session_id = str((payload.get("session") or {}).get("session_id") or "")
        if "front.test" in url:
            state = "idle" if action == "stop" else "running"
            return HttpResult(
                200,
                {
                    "ok": True,
                    "node_id": "front",
                    "action": action,
                    "session_id": session_id,
                    "state": state,
                    "message": "ok",
                },
                '{"ok": true}',
            )
        return HttpResult(503, {"error": "node unavailable"}, "node unavailable")


def build_node_config(tmpdir: Path, *, node_id: str, display_name: str, camera_label: str, host: str, port: int) -> NodeAgentConfig:
    return NodeAgentConfig(
        config_path=tmpdir / f"{node_id}.ini",
        node_id=node_id,
        display_name=display_name,
        camera_label=camera_label,
        profile=node_id,
        host=host,
        port=port,
        api_key=f"{node_id}-key",
        central_base_url="http://central.test:8090",
        registration_interval_sec=10.0,
        heartbeat_interval_sec=10.0,
        http_timeout_sec=5.0,
        source_mode="video",
        camera_index=0,
        video_path=tmpdir / f"{node_id}.mp4",
        preview_width=640,
        preview_fps=30.0,
        jpeg_quality=75,
        detector_mode="motion",
        runtime_config_path=None,
        motion_threshold=5.0,
        motion_min_area_ratio=0.001,
        motion_cooldown_sec=0.05,
        annotated_banner_ttl_sec=1.0,
        evidence_root=tmpdir / node_id / "evidence",
        pre_event_frames=2,
        post_event_frames=2,
    )


class CentralNodeIntegrationTests(unittest.TestCase):
    def test_pending_incident_updates_to_ready_with_snapshot_without_duplicate_row(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            connection = connect_db(tmpdir / "central" / "central.sqlite3")
            init_db(connection)
            repo = CentralRepository(connection)
            config = CentralServiceConfig(
                config_path=tmpdir / "central.ini",
                host="127.0.0.1",
                port=8090,
                db_path=tmpdir / "central" / "central.sqlite3",
                evidence_root=tmpdir / "central" / "evidence",
                node_offline_after_sec=120,
                proxy_timeout_sec=5.0,
                stream_timeout_sec=5.0,
                browser_auth=BrowserAuthConfig(
                    username="admin",
                    password="admin123",
                    secret_key="test-secret",
                    session_ttl_minutes=60,
                ),
                known_nodes={
                    "front": KnownNodeConfig("front", "Front Node", "Front Camera", "front-key"),
                },
            )
            manager = CentralServiceManager(config, repo)
            session_id = manager.create_session(
                {
                    "subject_code": "CS321",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-24",
                    "start_time": "09:00",
                    "end_time": "11:00",
                }
            )["session"]["session_id"]
            repo.update_session_status(session_id, "running")

            manager.upsert_incident(
                IncidentManifest(
                    incident_id="incident-001",
                    session_id=session_id,
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="object",
                    type_label="Using Phone",
                    student_numbers=[5],
                    created_at="2026-04-24T01:00:00Z",
                    display_time="09:00 AM",
                    frame_count=0,
                    summary="Student #05 using phone detected",
                    sync_status="recording",
                    sync_attempts=0,
                    asset_names=[],
                ).to_dict()
            )
            pending_snapshot = manager.dashboard_snapshot()
            pending_incidents = pending_snapshot["incidents"]
            self.assertEqual(len(pending_incidents), 1)
            self.assertEqual(pending_incidents[0]["sync_status"], "recording")
            self.assertEqual(pending_incidents[0]["gif_url"], "")

            review_result = manager.update_review_status("incident-001", "verified")
            self.assertTrue(review_result["ok"])
            self.assertEqual(review_result["incident"]["review_status"], "verified")

            manager.upsert_incident(
                IncidentManifest(
                    incident_id="incident-001",
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
                ).to_dict()
            )
            asset_result = manager.store_asset(
                {
                    "incident_id": "incident-001",
                    "session_id": session_id,
                    "node_id": "front",
                    "asset_type": "poster",
                    "filename": "poster.jpg",
                    "content_base64": base64.b64encode(b"poster-data").decode("ascii"),
                    "content_sha256": "",
                    "size_bytes": 11,
                }
            )
            self.assertTrue(asset_result["ok"])

            ready_snapshot = manager.dashboard_snapshot()
            ready_incidents = ready_snapshot["incidents"]
            self.assertEqual(len(ready_incidents), 1)
            self.assertEqual(ready_incidents[0]["incident_id"], "incident-001")
            self.assertEqual(ready_incidents[0]["sync_status"], "ready")
            self.assertEqual(ready_incidents[0]["review_status"], "verified")
            self.assertTrue(ready_incidents[0]["poster_url"])
            self.assertEqual(ready_incidents[0]["gif_url"], "")

            manager.upsert_incident(
                IncidentManifest(
                    incident_id="incident-001",
                    session_id=session_id,
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="object",
                    type_label="Using Phone",
                    student_numbers=[5],
                    created_at="2026-04-24T01:00:00Z",
                    display_time="09:00 AM",
                    frame_count=0,
                    summary="Late recording retry should not downgrade ready evidence",
                    sync_status="recording",
                    sync_attempts=1,
                    asset_names=[],
                ).to_dict()
            )
            retry_snapshot = manager.dashboard_snapshot()
            retry_incidents = retry_snapshot["incidents"]
            self.assertEqual(len(retry_incidents), 1)
            self.assertEqual(retry_incidents[0]["sync_status"], "ready")
            self.assertEqual(retry_incidents[0]["frame_count"], 1)
            self.assertTrue(retry_incidents[0]["poster_url"])
            self.assertEqual(retry_incidents[0]["gif_url"], "")
            connection.close()

    def test_central_rejects_stale_node_uploads_outside_running_session(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            connection = connect_db(tmpdir / "central" / "central.sqlite3")
            init_db(connection)
            repo = CentralRepository(connection)
            config = CentralServiceConfig(
                config_path=tmpdir / "central.ini",
                host="127.0.0.1",
                port=8090,
                db_path=tmpdir / "central" / "central.sqlite3",
                evidence_root=tmpdir / "central" / "evidence",
                node_offline_after_sec=120,
                proxy_timeout_sec=5.0,
                stream_timeout_sec=5.0,
                browser_auth=BrowserAuthConfig(
                    username="admin",
                    password="admin123",
                    secret_key="test-secret",
                    session_ttl_minutes=60,
                ),
                known_nodes={
                    "front": KnownNodeConfig("front", "Front Node", "Front Camera", "front-key"),
                },
            )
            manager = CentralServiceManager(config, repo)

            stale_manifest_result = manager.upsert_incident(
                IncidentManifest(
                    incident_id="stale-incident",
                    session_id="old-session",
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="object",
                    type_label="Using Phone",
                    student_numbers=[5],
                    created_at="2026-04-24T01:00:00Z",
                    display_time="09:00 AM",
                    frame_count=1,
                    summary="Stale incident",
                    sync_status="ready",
                    sync_attempts=0,
                    asset_names=[],
                ).to_dict()
            )
            self.assertEqual(stale_manifest_result["status_code"], 409)
            self.assertEqual(repo.list_incidents(), [])

            stale_asset_result = manager.store_asset(
                {
                    "incident_id": "stale-incident",
                    "session_id": "old-session",
                    "node_id": "front",
                    "asset_type": "poster",
                    "filename": "poster.jpg",
                    "content_base64": base64.b64encode(b"stale").decode("ascii"),
                    "content_sha256": "",
                    "size_bytes": 5,
                }
            )
            self.assertEqual(stale_asset_result["status_code"], 409)
            self.assertFalse(any(config.evidence_root.rglob("*")))

            session_id = manager.create_session(
                {
                    "subject_code": "CS321",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-24",
                    "start_time": "09:00",
                    "end_time": "11:00",
                }
            )["session"]["session_id"]
            repo.update_session_status(session_id, "running")

            valid_manifest_result = manager.upsert_incident(
                IncidentManifest(
                    incident_id="valid-incident",
                    session_id=session_id,
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="object",
                    type_label="Using Phone",
                    student_numbers=[5],
                    created_at="2026-04-24T01:00:00Z",
                    display_time="09:00 AM",
                    frame_count=1,
                    summary="Valid incident",
                    sync_status="ready",
                    sync_attempts=0,
                    asset_names=["poster.jpg"],
                ).to_dict()
            )
            self.assertTrue(valid_manifest_result["ok"])
            valid_asset_result = manager.store_asset(
                {
                    "incident_id": "valid-incident",
                    "session_id": session_id,
                    "node_id": "front",
                    "asset_type": "poster",
                    "filename": "poster.jpg",
                    "content_base64": base64.b64encode(b"poster").decode("ascii"),
                    "content_sha256": "",
                    "size_bytes": 6,
                }
            )
            self.assertTrue(valid_asset_result["ok"])
            connection.close()

    def test_stop_session_marks_session_stopped_even_when_one_node_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            connection = connect_db(tmpdir / "central" / "central.sqlite3")
            init_db(connection)
            repo = CentralRepository(connection)
            config = CentralServiceConfig(
                config_path=tmpdir / "central.ini",
                host="127.0.0.1",
                port=8090,
                db_path=tmpdir / "central" / "central.sqlite3",
                evidence_root=tmpdir / "central" / "evidence",
                node_offline_after_sec=120,
                proxy_timeout_sec=5.0,
                stream_timeout_sec=5.0,
                browser_auth=BrowserAuthConfig(
                    username="admin",
                    password="admin123",
                    secret_key="test-secret",
                    session_ttl_minutes=60,
                ),
                known_nodes={
                    "front": KnownNodeConfig("front", "Front Node", "Front Camera", "front-key"),
                    "mid": KnownNodeConfig("mid", "Mid Node", "Mid Camera", "mid-key"),
                },
            )
            manager = CentralServiceManager(config, repo, http_client=PartialStopHttpClient())
            repo.upsert_node_registration(
                NodeDescriptor(
                    node_id="front",
                    display_name="Front Node",
                    camera_label="Front Camera",
                    base_url="http://front.test:8091",
                    agent_base_url="http://front.test:8091",
                )
            )
            repo.upsert_node_registration(
                NodeDescriptor(
                    node_id="mid",
                    display_name="Mid Node",
                    camera_label="Mid Camera",
                    base_url="http://mid.test:8092",
                    agent_base_url="http://mid.test:8092",
                )
            )

            create_result = manager.create_session(
                {
                    "subject_code": "CS321",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-07",
                    "start_time": "09:00",
                    "end_time": "11:00",
                }
            )
            session_id = create_result["session"]["session_id"]
            start_result = manager.dispatch_session_command(session_id, "start")
            self.assertEqual(start_result["session"]["status"], "degraded")
            self.assertEqual(repo.get_active_session()["session_id"], session_id)

            stop_result = manager.dispatch_session_command(session_id, "stop")
            self.assertTrue(stop_result["ok"])
            self.assertEqual(stop_result["session"]["status"], "stopped")
            self.assertTrue(stop_result["session"]["stopped_at"])
            self.assertIsNone(repo.get_active_session())
            self.assertEqual(1, len([item for item in stop_result["results"] if item["ok"]]))
            connection.close()

    def test_registration_session_control_sync_review_and_stream_proxy(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            http_client = InProcessHttpClient()

            central_config = CentralServiceConfig(
                config_path=tmpdir / "central.ini",
                host="127.0.0.1",
                port=8090,
                db_path=tmpdir / "central" / "central.sqlite3",
                evidence_root=tmpdir / "central" / "evidence",
                node_offline_after_sec=120,
                proxy_timeout_sec=5.0,
                stream_timeout_sec=5.0,
                browser_auth=BrowserAuthConfig(
                    username="admin",
                    password="admin123",
                    secret_key="test-secret",
                    session_ttl_minutes=60,
                ),
                known_nodes={
                    "front": KnownNodeConfig("front", "Front Node", "Front Camera", "front-key"),
                    "mid": KnownNodeConfig("mid", "Mid Node", "Mid Camera", "mid-key"),
                },
            )

            central_app = create_central_app(central_config, http_client=http_client)
            http_client.register_app("http://central.test:8090", central_app)
            central_client = central_app.test_client()
            self.addCleanup(central_app.extensions["central_connection"].close)

            front_config = build_node_config(
                tmpdir,
                node_id="front",
                display_name="Front Node",
                camera_label="Front Camera",
                host="front.test",
                port=8091,
            )
            mid_config = build_node_config(
                tmpdir,
                node_id="mid",
                display_name="Mid Node",
                camera_label="Mid Camera",
                host="mid.test",
                port=8092,
            )

            front_runtime = NodeRuntime(front_config, http_client=http_client)
            mid_runtime = NodeRuntime(mid_config, http_client=http_client)
            self.addCleanup(front_runtime.close)
            self.addCleanup(mid_runtime.close)

            base_frames = [
                np.zeros((180, 320, 3), dtype=np.uint8),
                np.zeros((180, 320, 3), dtype=np.uint8),
                np.full((180, 320, 3), 255, dtype=np.uint8),
                np.full((180, 320, 3), 255, dtype=np.uint8),
            ]
            front_runtime._open_capture = lambda: FakeCapture(base_frames)  # type: ignore[attr-defined]
            mid_runtime._open_capture = lambda: FakeCapture(base_frames)  # type: ignore[attr-defined]

            front_app = create_node_app(front_config, runtime=front_runtime, start_background=False)
            mid_app = create_node_app(mid_config, runtime=mid_runtime, start_background=False)
            http_client.register_app("http://front.test:8091", front_app)
            http_client.register_app("http://mid.test:8092", mid_app)

            self.assertTrue(front_runtime.register_once())
            self.assertTrue(mid_runtime.register_once())
            self.assertTrue(front_runtime.heartbeat_once())
            self.assertTrue(mid_runtime.heartbeat_once())

            central_client.post(
                "/login",
                data={"username": "admin", "password": "admin123"},
                follow_redirects=True,
            )
            invalid_create_response = central_client.post(
                "/api/v1/sessions",
                json={
                    "subject_code": "",
                    "professor": "Dr. Reyes",
                },
            )
            self.assertEqual(invalid_create_response.status_code, 400)
            self.assertIn("Subject code is required", invalid_create_response.get_json()["error"])
            page_response = central_client.get("/dashboard")
            self.assertEqual(page_response.status_code, 200)
            self.assertIn(b"Records &amp; Analytics", page_response.data)
            self.assertIn(b"Seat Map", page_response.data)

            create_response = central_client.post(
                "/api/v1/sessions",
                json={
                    "subject_code": "CS321",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-07",
                    "start_time": "09:00",
                    "end_time": "11:00",
                },
            )
            self.assertEqual(create_response.status_code, 200)
            session_id = create_response.get_json()["session"]["session_id"]

            start_response = central_client.post(f"/api/v1/sessions/{session_id}/start", json={})
            self.assertEqual(start_response.status_code, 200)
            self.assertTrue(all(item["ok"] for item in start_response.get_json()["results"]))

            time.sleep(0.35)
            for _ in range(4):
                front_runtime.sync_once()
                mid_runtime.sync_once()

            front_runtime.update_sound_telemetry(
                {
                    "enabled": True,
                    "current_db": 60.8,
                    "threshold_db": 55.0,
                    "over_threshold": True,
                    "status": "alert",
                    "updated_at": "2026-04-22T12:00:00Z",
                    "last_error": "",
                }
            )
            self.assertTrue(front_runtime.heartbeat_once())
            noise_poster_path = tmpdir / "front" / "noise-001" / "poster.jpg"
            noise_poster_path.parent.mkdir(parents=True, exist_ok=True)
            noise_poster_path.write_bytes(b"noise-jpeg")
            front_runtime.record_finalized_incident(
                IncidentManifest(
                    incident_id="noise-001",
                    session_id=session_id,
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="noise",
                    type_label="Noise Threshold Exceeded",
                    student_numbers=[],
                    created_at="2026-04-22T12:00:00Z",
                    display_time="12:00 PM",
                    frame_count=1,
                    summary="Estimated noise 60.8 dB exceeded 55.0 dB threshold.",
                    sync_status="queued",
                    sync_attempts=0,
                    asset_names=["poster.jpg"],
                ),
                [
                    {
                        "asset_type": "poster",
                        "file_path": noise_poster_path,
                        "filename": "poster.jpg",
                    }
                ],
            )
            cheat_poster_path = tmpdir / "front" / "cheat-001" / "poster.jpg"
            cheat_poster_path.parent.mkdir(parents=True, exist_ok=True)
            cheat_poster_path.write_bytes(b"cheat-jpeg")
            front_runtime.record_finalized_incident(
                IncidentManifest(
                    incident_id="cheat-001",
                    session_id=session_id,
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="object",
                    type_label="Phone Detected",
                    student_numbers=[5],
                    created_at="2026-04-22T12:00:01Z",
                    display_time="12:00 PM",
                    frame_count=1,
                    summary="Student #05 phone detected",
                    sync_status="queued",
                    sync_attempts=0,
                    asset_names=["poster.jpg"],
                ),
                [
                    {
                        "asset_type": "poster",
                        "file_path": cheat_poster_path,
                        "filename": "poster.jpg",
                    },
                ],
            )
            for _ in range(10):
                front_runtime.sync_once()
                mid_runtime.sync_once()

            dashboard_response = central_client.get("/api/v1/dashboard")
            self.assertEqual(dashboard_response.status_code, 200)
            payload = dashboard_response.get_json()
            self.assertEqual(len(payload["nodes"]), 2)
            for node in payload["nodes"]:
                self.assertIn("debug", node["stream_urls"])
                self.assertEqual(
                    node["stream_urls"]["debug"],
                    f"/api/v1/streams/{node['node_id']}/debug",
                )
            self.assertGreaterEqual(len(payload["incidents"]), 2)
            self.assertTrue(payload["nodes"][0]["extra"]["sound"]["enabled"] or payload["nodes"][1]["extra"]["sound"]["enabled"])
            noise_incident = next(item for item in payload["incidents"] if item["behavior_type"] == "noise")
            self.assertTrue(noise_incident["poster_url"])
            self.assertEqual(noise_incident["gif_url"], "")
            noise_poster_response = central_client.get(noise_incident["poster_url"])
            self.assertEqual(noise_poster_response.status_code, 200)
            self.assertEqual(noise_poster_response.data, b"noise-jpeg")
            noise_poster_response.close()
            cheat_incident = next(item for item in payload["incidents"] if item["incident_id"] == "cheat-001")
            self.assertTrue(cheat_incident["poster_url"])
            self.assertEqual(cheat_incident["gif_url"], "")
            cheat_poster_response = central_client.get(cheat_incident["poster_url"])
            self.assertEqual(cheat_poster_response.status_code, 200)
            self.assertEqual(cheat_poster_response.data, b"cheat-jpeg")
            cheat_poster_response.close()
            self.assertIn("sessions_history", payload)
            self.assertEqual(payload["sessions_history"][0]["session_id"], session_id)
            self.assertGreaterEqual(payload["sessions_history"][0]["incident_count"], 2)

            stop_response = central_client.post(f"/api/v1/sessions/{session_id}/stop", json={})
            self.assertEqual(stop_response.status_code, 200)

            first_incident = payload["incidents"][0]
            review_response = central_client.post(
                f"/api/v1/incidents/{first_incident['incident_id']}/review",
                json={"review_status": "verified"},
            )
            self.assertEqual(review_response.status_code, 200)
            self.assertEqual(review_response.get_json()["incident"]["review_status"], "verified")

            clear_records_response = central_client.post(
                "/api/v1/incidents/clear",
                json={"session_id": session_id},
            )
            self.assertEqual(clear_records_response.status_code, 200)
            self.assertGreaterEqual(clear_records_response.get_json()["cleared_incidents"], 2)

            cleared_snapshot = central_client.get("/api/v1/dashboard").get_json()
            self.assertEqual(cleared_snapshot["incidents"], [])
            self.assertEqual(cleared_snapshot["sessions_history"][0]["incident_count"], 0)

            http_client.set_stream_payload(
                "http://front.test:8091/agent/v1/stream/annotated",
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\nstream-bytes\r\n",
            )
            stream_response = central_client.get("/api/v1/streams/front/annotated")
            self.assertEqual(stream_response.status_code, 200)
            self.assertIn(b"stream-bytes", stream_response.data)
            http_client.set_stream_payload(
                "http://front.test:8091/agent/v1/stream/debug",
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\ndebug-stream-bytes\r\n",
            )
            debug_stream_response = central_client.get("/api/v1/streams/front/debug")
            self.assertEqual(debug_stream_response.status_code, 200)
            self.assertIn(b"debug-stream-bytes", debug_stream_response.data)
            invalid_stream_response = central_client.get("/api/v1/streams/front/diagnostic")
            self.assertEqual(invalid_stream_response.status_code, 404)

            front_runtime.close()
            mid_runtime.close()
            central_app.extensions["central_connection"].close()

    def test_startup_reset_clear_and_shutdown_session_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            http_client = InProcessHttpClient()

            central_config = CentralServiceConfig(
                config_path=tmpdir / "central.ini",
                host="127.0.0.1",
                port=8090,
                db_path=tmpdir / "central" / "central.sqlite3",
                evidence_root=tmpdir / "central" / "evidence",
                node_offline_after_sec=120,
                proxy_timeout_sec=5.0,
                stream_timeout_sec=5.0,
                browser_auth=BrowserAuthConfig(
                    username="admin",
                    password="admin123",
                    secret_key="test-secret",
                    session_ttl_minutes=60,
                ),
                known_nodes={
                    "front": KnownNodeConfig("front", "Front Node", "Front Camera", "front-key"),
                    "mid": KnownNodeConfig("mid", "Mid Node", "Mid Camera", "mid-key"),
                },
            )

            central_app = create_central_app(central_config, http_client=http_client)
            central_client = central_app.test_client()
            central_client.post(
                "/login",
                data={"username": "admin", "password": "admin123"},
                follow_redirects=True,
            )

            create_response = central_client.post(
                "/api/v1/sessions",
                json={
                    "subject_code": "CS321",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-07",
                    "start_time": "09:00",
                    "end_time": "11:00",
                },
            )
            self.assertEqual(create_response.status_code, 200)
            session_id = create_response.get_json()["session"]["session_id"]
            self.assertEqual(central_app.extensions["central_manager"].dashboard_snapshot()["active_session"]["session_id"], session_id)
            central_app.extensions["central_connection"].close()

            restarted_app = create_central_app(central_config, http_client=http_client)
            restarted_client = restarted_app.test_client()
            restarted_client.post(
                "/login",
                data={"username": "admin", "password": "admin123"},
                follow_redirects=True,
            )

            restarted_snapshot = restarted_app.extensions["central_manager"].dashboard_snapshot()
            self.assertIsNone(restarted_snapshot["active_session"])
            self.assertEqual(restarted_snapshot["sessions_history"][0]["status"], "stopped")

            create_response = restarted_client.post(
                "/api/v1/sessions",
                json={
                    "subject_code": "CS322",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-08",
                    "start_time": "13:00",
                    "end_time": "15:00",
                },
            )
            self.assertEqual(create_response.status_code, 200)
            clear_response = restarted_client.post("/api/v1/sessions/current/clear", json={})
            self.assertEqual(clear_response.status_code, 200)
            cleared_session = clear_response.get_json()["session"]
            self.assertEqual(cleared_session["status"], "cleared")
            self.assertIsNone(restarted_app.extensions["central_manager"].dashboard_snapshot()["active_session"])

            create_response = restarted_client.post(
                "/api/v1/sessions",
                json={
                    "subject_code": "CS322",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-08",
                    "start_time": "15:00",
                    "end_time": "17:00",
                },
            )
            self.assertEqual(create_response.status_code, 200)
            second_cleared_id = create_response.get_json()["session"]["session_id"]
            self.assertTrue(second_cleared_id)
            clear_response = restarted_client.post("/api/v1/sessions/current/clear", json={})
            self.assertEqual(clear_response.status_code, 200)

            delete_subject_response = restarted_client.post(
                "/api/v1/sessions/subjects/delete",
                json={"subject_code": "CS322"},
            )
            self.assertEqual(delete_subject_response.status_code, 200)
            self.assertEqual(delete_subject_response.get_json()["deleted_sessions"], 2)
            post_subject_delete_snapshot = restarted_app.extensions["central_manager"].dashboard_snapshot()
            self.assertFalse(any(item["subject_code"] == "CS322" for item in post_subject_delete_snapshot["sessions_history"]))

            create_response = restarted_client.post(
                "/api/v1/sessions",
                json={
                    "subject_code": "CS323",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-09",
                    "start_time": "16:00",
                    "end_time": "18:00",
                },
            )
            self.assertEqual(create_response.status_code, 200)
            deleted_session_id = create_response.get_json()["session"]["session_id"]
            delete_session_response = restarted_client.delete(f"/api/v1/sessions/{deleted_session_id}")
            self.assertEqual(delete_session_response.status_code, 200)
            self.assertEqual(delete_session_response.get_json()["deleted_session"]["session_id"], deleted_session_id)
            deleted_snapshot = restarted_app.extensions["central_manager"].dashboard_snapshot()
            self.assertIsNone(deleted_snapshot["active_session"])
            self.assertFalse(any(item["session_id"] == deleted_session_id for item in deleted_snapshot["sessions_history"]))

            create_response = restarted_client.post(
                "/api/v1/sessions",
                json={
                    "subject_code": "CS324",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-09",
                    "start_time": "16:00",
                    "end_time": "18:00",
                },
            )
            self.assertEqual(create_response.status_code, 200)
            shutdown_session_id = create_response.get_json()["session"]["session_id"]
            restarted_app.extensions["central_shutdown_handler"]()
            self.assertTrue(restarted_app.extensions["central_shutdown_done"])
            shutdown_snapshot = restarted_app.extensions["central_manager"].dashboard_snapshot()
            self.assertIsNone(shutdown_snapshot["active_session"])
            shutdown_entry = next(item for item in shutdown_snapshot["sessions_history"] if item["session_id"] == shutdown_session_id)
            self.assertEqual(shutdown_entry["status"], "stopped")
            restarted_app.extensions["central_connection"].close()


if __name__ == "__main__":
    unittest.main()
