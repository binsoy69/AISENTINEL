from __future__ import annotations

import io
import json
import unittest
import uuid
from pathlib import Path
import sys

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.central_service.config import BrowserAuthConfig, CentralServiceConfig, KnownNodeConfig
from editable_dashboard.editable_service.app import create_app


ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_ROOT = ROOT / "data" / "test_tmp"


def build_config(tmpdir: Path) -> CentralServiceConfig:
    return CentralServiceConfig(
        config_path=tmpdir / "editable_dashboard.ini",
        host="127.0.0.1",
        port=8095,
        db_path=tmpdir / "editable" / "editable.sqlite3",
        evidence_root=tmpdir / "editable" / "evidence",
        node_offline_after_sec=120,
        proxy_timeout_sec=5.0,
        stream_timeout_sec=5.0,
        browser_auth=BrowserAuthConfig(
            username="admin",
            password="admin123",
            secret_key="editable-test-secret",
            session_ttl_minutes=60,
        ),
        known_nodes={
            "front": KnownNodeConfig("front", "Front Node", "Front Camera", "front-key"),
            "mid": KnownNodeConfig("mid", "Mid Node", "Mid Camera", "mid-key"),
        },
    )


class EditableDashboardIntegrationTests(unittest.TestCase):
    def test_login_session_simulated_nodes_and_editable_incident_crud(self):
        tmpdir = TEST_DATA_ROOT / uuid.uuid4().hex
        tmpdir.mkdir(parents=True, exist_ok=True)
        app = create_app(build_config(tmpdir))
        client = app.test_client()
        self.addCleanup(app.extensions["central_connection"].close)

        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            follow_redirects=True,
        )
        self.assertEqual(login_response.status_code, 200)
        self.assertIn(b"Records &amp; Analytics", login_response.data)

        snapshot = client.get("/api/v1/dashboard").get_json()
        self.assertTrue(snapshot["editable_demo"])
        self.assertIsNone(snapshot["active_session"])
        self.assertEqual(len(snapshot["nodes"]), 2)
        self.assertTrue(all(node["online"] for node in snapshot["nodes"]))
        self.assertFalse(any(node["extra"]["stream"]["has_annotated_frame"] for node in snapshot["nodes"]))

        create_response = client.post(
            "/api/v1/sessions",
            json={
                "subject_code": "DEMO101",
                "professor": "Dr. Reyes",
                "session_date": "2026-05-04",
                "start_time": "09:00",
                "end_time": "11:00",
            },
        )
        self.assertEqual(create_response.status_code, 200)
        session_id = create_response.get_json()["session"]["session_id"]

        start_response = client.post(f"/api/v1/sessions/{session_id}/start", json={})
        self.assertEqual(start_response.status_code, 200)
        self.assertEqual(start_response.get_json()["session"]["status"], "running")
        running_snapshot = client.get("/api/v1/dashboard").get_json()
        self.assertEqual(running_snapshot["active_session"]["status"], "running")
        self.assertTrue(all(node["state"] == "running" for node in running_snapshot["nodes"]))

        add_response = client.post(
            "/api/v1/editable/incidents",
            data={
                "records": json.dumps(
                    [
                        {
                            "incident_id": "incident-demo-001",
                            "timestamp": "2026-05-04T09:15",
                            "student_numbers": [5],
                                "type_label": "Using Phone",
                            "camera_label": "Front Camera",
                            "file_field": "asset_001",
                        }
                    ]
                ),
                "asset_001": (io.BytesIO(b"GIF89a-demo"), "evidence.gif"),
            },
        )
        self.assertEqual(add_response.status_code, 200)
        add_payload = add_response.get_json()
        self.assertEqual(add_payload["saved_count"], 1)
        added = add_payload["incidents"][0]
        self.assertEqual(added["incident_id"], "incident-demo-001")
        self.assertEqual(added["student_numbers"], [5])
        self.assertEqual(added["type_label"], "Using Phone")
        self.assertTrue(added["gif_url"])
        evidence_response = client.get(added["gif_url"])
        self.assertEqual(evidence_response.status_code, 200)
        self.assertEqual(evidence_response.data, b"GIF89a-demo")
        evidence_response.close()

        edit_response = client.post(
            "/api/v1/editable/incidents",
            data={
                "records": json.dumps(
                    [
                        {
                            "incident_id": "incident-demo-001",
                            "timestamp": "2026-05-04T09:30",
                            "student_numbers": [5, 6],
                            "type_label": "Passing Papers",
                            "camera_label": "Mid Camera",
                        }
                    ]
                )
            },
        )
        self.assertEqual(edit_response.status_code, 200)
        edited = edit_response.get_json()["incidents"][0]
        self.assertEqual(edited["student_numbers"], [5, 6])
        self.assertEqual(edited["type_label"], "Passing Papers")
        self.assertEqual(edited["camera_label"], "Mid Camera")
        self.assertTrue(edited["gif_url"])

        delete_response = client.delete("/api/v1/editable/incidents/incident-demo-001")
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.get_json()["incidents"], [])
        self.assertEqual(client.get("/api/v1/dashboard").get_json()["incidents"], [])

    def test_static_assets_expose_edit_mode_and_hidden_preview_defaults(self):
        script = (ROOT / "editable_service" / "static" / "dashboard.js").read_text(encoding="utf-8")
        stylesheet = (ROOT / "editable_service" / "static" / "app.css").read_text(encoding="utf-8")

        self.assertIn("const EDITABLE_DASHBOARD = true", script)
        self.assertIn('const DEFAULT_EDITABLE_TYPES = ["Using Phone", "Cheat Sheet", "Head Tilt", "Hands Under the Table", "Passing Papers", "Noise Threshold Exceeded"]', script)
        self.assertIn("return stored === null ? true", script)
        self.assertIn("function beginEditableMode", script)
        self.assertIn("function saveEditableChanges", script)
        self.assertIn("data-editable-add-record", script)
        self.assertIn("data-editable-delete", script)
        self.assertIn("/api/v1/editable/incidents", script)
        self.assertIn("Editable demo mode keeps camera previews disconnected", script)
        self.assertIn(".editable-add-record-button", stylesheet)
        self.assertIn(".editable-add-record-button[hidden]", stylesheet)
        self.assertIn(".editable-file-control", stylesheet)


if __name__ == "__main__":
    unittest.main()
