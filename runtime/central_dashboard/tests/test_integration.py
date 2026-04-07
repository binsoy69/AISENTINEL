from __future__ import annotations

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
from central_dashboard.node_agent.app import create_app as create_node_app
from central_dashboard.node_agent.config import NodeAgentConfig
from central_dashboard.node_agent.state import NodeRuntime

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
        sync_interval_sec=10.0,
        http_timeout_sec=5.0,
        local_db_path=tmpdir / node_id / "queue.sqlite3",
        source_mode="video",
        camera_index=0,
        video_path=tmpdir / f"{node_id}.mp4",
        preview_width=640,
        preview_fps=30.0,
        jpeg_quality=75,
        detector_mode="motion",
        motion_threshold=5.0,
        motion_min_area_ratio=0.001,
        motion_cooldown_sec=0.05,
        annotated_banner_ttl_sec=1.0,
        evidence_root=tmpdir / node_id / "evidence",
        pre_event_frames=2,
        post_event_frames=2,
    )


class CentralNodeIntegrationTests(unittest.TestCase):
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
            stop_response = central_client.post(f"/api/v1/sessions/{session_id}/stop", json={})
            self.assertEqual(stop_response.status_code, 200)

            for _ in range(4):
                front_runtime.sync_once()
                mid_runtime.sync_once()

            dashboard_response = central_client.get("/api/v1/dashboard")
            self.assertEqual(dashboard_response.status_code, 200)
            payload = dashboard_response.get_json()
            self.assertEqual(len(payload["nodes"]), 2)
            self.assertGreaterEqual(len(payload["incidents"]), 2)

            first_incident = payload["incidents"][0]
            review_response = central_client.post(
                f"/api/v1/incidents/{first_incident['incident_id']}/review",
                json={"review_status": "verified"},
            )
            self.assertEqual(review_response.status_code, 200)
            self.assertEqual(review_response.get_json()["incident"]["review_status"], "verified")

            http_client.set_stream_payload(
                "http://front.test:8091/agent/v1/stream/annotated",
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\nstream-bytes\r\n",
            )
            stream_response = central_client.get("/api/v1/streams/front/annotated")
            self.assertEqual(stream_response.status_code, 200)
            self.assertIn(b"stream-bytes", stream_response.data)

            front_runtime.close()
            mid_runtime.close()
            central_app.extensions["central_connection"].close()


if __name__ == "__main__":
    unittest.main()
