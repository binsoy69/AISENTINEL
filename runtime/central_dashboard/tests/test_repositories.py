from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.central_service.db import connect_db, init_db
from central_dashboard.central_service.repositories import CentralRepository
from central_dashboard.shared.dto import IncidentManifest, NodeDescriptor, NodeHeartbeat, SessionSpec


class RepositoryTests(unittest.TestCase):
    def test_registration_heartbeat_session_and_review_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "central.sqlite3"
            connection = connect_db(db_path)
            init_db(connection)
            repo = CentralRepository(connection)

            repo.upsert_node_registration(
                NodeDescriptor(
                    node_id="front",
                    display_name="Front Node",
                    camera_label="Front Camera",
                    base_url="http://front.test:8091",
                    agent_base_url="http://front.test:8091",
                )
            )
            repo.update_node_heartbeat(
                NodeHeartbeat(
                    node_id="front",
                    state="running",
                    session_id="session-1",
                    fps=14.5,
                    sync_backlog=2,
                    incident_count=1,
                )
            )

            repo.create_session(
                SessionSpec(
                    session_id="session-1",
                    subject_code="CS321",
                    professor="Dr. Reyes",
                    session_date="2026-04-07",
                    start_time="09:00",
                    end_time="11:00",
                )
            )

            repo.upsert_incident(
                IncidentManifest(
                    incident_id="incident-1",
                    session_id="session-1",
                    node_id="front",
                    camera_label="Front Camera",
                    behavior_type="motion",
                    type_label="Movement Spike",
                    student_numbers=[],
                    created_at="2026-04-07T01:00:00Z",
                    display_time="9:00 AM",
                    summary="Synthetic motion incident.",
                )
            )
            updated = repo.update_review_status("incident-1", "verified")

            self.assertIsNotNone(updated)
            self.assertEqual(updated["review_status"], "verified")
            self.assertEqual(repo.get_active_session()["session_id"], "session-1")
            nodes = repo.node_status_snapshot(
                {"front": type("Known", (), {"display_name": "Front Node", "camera_label": "Front Camera"})()},
                offline_after_sec=9999,
            )
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0]["fps"], 14.5)
            connection.close()


if __name__ == "__main__":
    unittest.main()
