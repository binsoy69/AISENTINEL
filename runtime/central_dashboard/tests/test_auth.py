from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.central_service.app import create_app as create_central_app
from central_dashboard.central_service.config import BrowserAuthConfig, CentralServiceConfig, KnownNodeConfig
from central_dashboard.node_agent.app import create_app as create_node_app
from central_dashboard.node_agent.config import NodeAgentConfig


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
                sync_interval_sec=10.0,
                http_timeout_sec=5.0,
                local_db_path=tmpdir / "queue.sqlite3",
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


if __name__ == "__main__":
    unittest.main()
