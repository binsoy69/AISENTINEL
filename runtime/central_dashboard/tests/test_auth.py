from __future__ import annotations

import base64
import http.client
from io import BytesIO
import tempfile
import unittest
from pathlib import Path
import sys

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.central_service.app import create_app as create_central_app
from central_dashboard.central_service.config import BrowserAuthConfig, CentralServiceConfig, KnownNodeConfig
from central_dashboard.central_service.proxy import relay_stream_chunks
from central_dashboard.node_agent.app import create_app as create_node_app
from central_dashboard.node_agent.config import NodeAgentConfig
from central_dashboard.shared.dto import SessionSpec


class AuthTests(unittest.TestCase):
    def test_central_node_api_requires_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config = CentralServiceConfig(
                config_path=tmpdir / "central.ini",
                host="127.0.0.1",
                port=8090,
                db_path=tmpdir / "central.sqlite3",
                evidence_root=tmpdir / "evidence",
                node_offline_after_sec=10,
                proxy_timeout_sec=5.0,
                stream_timeout_sec=5.0,
                browser_auth=BrowserAuthConfig("admin", "admin123", "secret", 60),
                known_nodes={"front": KnownNodeConfig("front", "Front Node", "Front Camera", "front-key")},
            )
            app = create_central_app(config)
            client = app.test_client()
            response = client.post("/api/v1/nodes/register", json={"node_id": "front"})
            self.assertEqual(response.status_code, 401)
            app.extensions["central_connection"].close()

    def test_central_node_payload_node_id_uses_authenticated_header(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config = CentralServiceConfig(
                config_path=tmpdir / "central.ini",
                host="127.0.0.1",
                port=8090,
                db_path=tmpdir / "central.sqlite3",
                evidence_root=tmpdir / "evidence",
                node_offline_after_sec=10,
                proxy_timeout_sec=5.0,
                stream_timeout_sec=5.0,
                browser_auth=BrowserAuthConfig("admin", "admin123", "secret", 60),
                known_nodes={"front": KnownNodeConfig("front", "Front Node", "Front Camera", "front-key")},
            )
            app = create_central_app(config)
            client = app.test_client()
            headers = {"X-Node-Id": "front", "X-Api-Key": "front-key"}
            repository = app.extensions["central_repository"]
            repository.create_session(
                SessionSpec(
                    session_id="session-1",
                    subject_code="CS321",
                    professor="Dr. Reyes",
                )
            )
            repository.update_session_status("session-1", "running")

            manifest_response = client.post(
                "/api/v1/incidents",
                json={
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "node_id": "wrong-node",
                    "camera_label": "Front Camera",
                    "behavior_type": "noise",
                    "type_label": "Noise Threshold Exceeded",
                    "created_at": "2026-04-23T04:04:00Z",
                },
                headers=headers,
            )
            self.assertEqual(manifest_response.status_code, 200)

            upload_response = client.post(
                "/api/v1/evidence/upload",
                json={
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "node_id": "wrong-node",
                    "asset_type": "poster",
                    "filename": "poster.jpg",
                    "content_base64": base64.b64encode(b"jpeg-data").decode("ascii"),
                    "content_sha256": "",
                    "size_bytes": 9,
                },
                headers=headers,
            )
            self.assertEqual(upload_response.status_code, 200)
            relative_path = upload_response.get_json()["relative_path"]
            self.assertTrue(relative_path.startswith("session-1/front/incident-1/"))

            nested_upload_response = client.post(
                "/api/v1/evidence/upload",
                json={
                    "incident_id": "incident-1",
                    "asset_type": "poster",
                    "filename": "objects/events/incident-1/frames/f11_event.jpg",
                    "content_base64": (
                        "data:image/jpeg;base64,"
                        + base64.b64encode(b"event-frame").decode("ascii")
                    ),
                    "content_sha256": "",
                    "size_bytes": 11,
                },
                headers=headers,
            )
            self.assertEqual(nested_upload_response.status_code, 200)
            self.assertEqual(
                nested_upload_response.get_json()["relative_path"],
                "session-1/front/incident-1/poster.jpg",
            )

            multipart_response = client.post(
                "/api/v1/evidence/upload",
                data={
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "asset_type": "gif",
                    "file": (BytesIO(b"gif-data"), "from-form.gif"),
                },
                headers=headers,
                content_type="multipart/form-data",
            )
            self.assertEqual(multipart_response.status_code, 200)
            self.assertEqual(
                multipart_response.get_json()["relative_path"],
                "session-1/front/incident-1/evidence.gif",
            )

            bad_upload_response = client.post(
                "/api/v1/evidence/upload",
                json={
                    "incident_id": "incident-1",
                    "session_id": "session-1",
                    "asset_type": "poster",
                    "filename": "bad.jpg",
                    "content_base64": "not base64",
                    "content_sha256": "",
                    "size_bytes": 9,
                },
                headers=headers,
            )
            self.assertEqual(bad_upload_response.status_code, 400)
            self.assertIn("Invalid evidence content", bad_upload_response.get_json()["error"])

            app.extensions["central_connection"].close()

    def test_node_agent_routes_require_api_key(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config = NodeAgentConfig(
                config_path=tmpdir / "front.ini",
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
                source_mode="video",
                camera_index=0,
                video_path=tmpdir / "demo.mp4",
                preview_width=640,
                preview_fps=10.0,
                jpeg_quality=70,
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
            app = create_node_app(config, start_background=False)
            client = app.test_client()
            response = client.get("/agent/v1/status")
            self.assertEqual(response.status_code, 401)
            app.extensions["node_runtime"].close()

    def test_stream_relay_stops_on_incomplete_read(self):
        class BrokenStream:
            def __init__(self):
                self.closed = False

            def read(self, _chunk_size):
                raise http.client.IncompleteRead(b"partial")

            def close(self):
                self.closed = True

        stream = BrokenStream()
        self.assertEqual(list(relay_stream_chunks(stream)), [])
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
