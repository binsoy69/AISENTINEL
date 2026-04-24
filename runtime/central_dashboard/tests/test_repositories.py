from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
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
            history = repo.list_sessions_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["session_id"], "session-1")
            self.assertEqual(history[0]["incident_count"], 1)
            nodes = repo.node_status_snapshot(
                {"front": type("Known", (), {"display_name": "Front Node", "camera_label": "Front Camera"})()},
                offline_after_sec=9999,
            )
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0]["fps"], 14.5)
            connection.close()

    def test_node_status_uses_central_receive_time_not_node_clock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "central.sqlite3"
            connection = connect_db(db_path)
            init_db(connection)
            repo = CentralRepository(connection)

            with patch(
                "central_dashboard.central_service.repositories.utc_now_iso",
                return_value="1970-01-01T00:00:00Z",
            ):
                repo.upsert_node_registration(
                    NodeDescriptor(
                        node_id="front",
                        display_name="Front Node",
                        camera_label="Front Camera",
                        base_url="http://front.test:8091",
                        agent_base_url="http://front.test:8091",
                        registered_at="2999-01-01T00:00:00Z",
                    )
                )
                repo.update_node_heartbeat(
                    NodeHeartbeat(
                        node_id="front",
                        state="running",
                        updated_at="2999-01-01T00:00:00Z",
                    )
                )

            nodes = repo.node_status_snapshot(
                {"front": type("Known", (), {"display_name": "Front Node", "camera_label": "Front Camera"})()},
                offline_after_sec=2,
            )
            self.assertEqual(nodes[0]["last_seen_at"], "1970-01-01T00:00:00Z")
            self.assertFalse(nodes[0]["online"])
            connection.close()

    def test_node_status_treats_persisted_future_last_seen_as_offline(self):
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
            connection.execute(
                "UPDATE nodes SET last_seen_at=?, state=?, last_error=? WHERE node_id=?",
                (
                    "2999-01-01T00:00:00Z",
                    "idle",
                    "<html><title>500 Internal Server Error</title></html>",
                    "front",
                ),
            )
            connection.commit()

            nodes = repo.node_status_snapshot(
                {"front": type("Known", (), {"display_name": "Front Node", "camera_label": "Front Camera"})()},
                offline_after_sec=12,
            )
            self.assertFalse(nodes[0]["online"])
            connection.close()

    def test_bulk_stop_or_clear_removes_active_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "central.sqlite3"
            connection = connect_db(db_path)
            init_db(connection)
            repo = CentralRepository(connection)

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

            self.assertEqual(repo.get_active_session()["session_id"], "session-1")
            self.assertEqual(repo.update_all_active_session_statuses("cleared"), 1)
            self.assertIsNone(repo.get_active_session())
            self.assertEqual(repo.get_session("session-1")["status"], "cleared")
            connection.close()

    def test_delete_incidents_for_session_updates_history_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "central.sqlite3"
            connection = connect_db(db_path)
            init_db(connection)
            repo = CentralRepository(connection)

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
                    student_numbers=[1],
                    created_at="2026-04-07T01:00:00Z",
                    display_time="9:00 AM",
                    summary="Synthetic motion incident.",
                )
            )

            self.assertEqual(repo.delete_incidents_for_session("session-1"), 1)
            self.assertEqual(repo.list_incidents(), [])
            self.assertEqual(repo.list_sessions_history()[0]["incident_count"], 0)
            connection.close()

    def test_delete_session_and_subject_sessions_remove_history_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "central.sqlite3"
            connection = connect_db(db_path)
            init_db(connection)
            repo = CentralRepository(connection)

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
            repo.create_session(
                SessionSpec(
                    session_id="session-2",
                    subject_code="CS321",
                    professor="Dr. Reyes",
                    session_date="2026-04-08",
                    start_time="13:00",
                    end_time="15:00",
                )
            )
            repo.create_session(
                SessionSpec(
                    session_id="session-3",
                    subject_code="CS322",
                    professor="Dr. Reyes",
                    session_date="2026-04-09",
                    start_time="16:00",
                    end_time="18:00",
                )
            )

            self.assertEqual(repo.delete_session("session-3"), 1)
            self.assertIsNone(repo.get_session("session-3"))
            self.assertEqual(repo.delete_sessions_by_subject("CS321"), 2)
            self.assertEqual(repo.list_sessions_history(), [])
            connection.close()


if __name__ == "__main__":
    unittest.main()
