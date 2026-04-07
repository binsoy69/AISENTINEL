"""SQLite repositories for central dashboard state."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3

from central_dashboard.shared.dto import (
    IncidentManifest,
    NodeDescriptor,
    NodeHeartbeat,
    SessionSpec,
    utc_now_iso,
)


def _parse_iso(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


class CentralRepository:
    """Persistence layer for the central service."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def upsert_node_registration(self, descriptor: NodeDescriptor) -> None:
        self.connection.execute(
            """
            INSERT INTO nodes (
                node_id, display_name, camera_label, profile, base_url, agent_base_url,
                capabilities_json, registered_at, last_seen_at, state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                display_name=excluded.display_name,
                camera_label=excluded.camera_label,
                profile=excluded.profile,
                base_url=excluded.base_url,
                agent_base_url=excluded.agent_base_url,
                capabilities_json=excluded.capabilities_json,
                registered_at=excluded.registered_at,
                last_seen_at=excluded.last_seen_at,
                state='registered'
            """,
            (
                descriptor.node_id,
                descriptor.display_name,
                descriptor.camera_label,
                descriptor.profile,
                descriptor.base_url,
                descriptor.agent_base_url,
                json.dumps(descriptor.capabilities),
                descriptor.registered_at,
                descriptor.registered_at,
                "registered",
            ),
        )
        self.connection.commit()

    def update_node_heartbeat(self, heartbeat: NodeHeartbeat) -> None:
        self.connection.execute(
            """
            UPDATE nodes
            SET last_seen_at=?, state=?, session_id=?, fps=?, sync_backlog=?,
                incident_count=?, last_error=?, extra_json=?
            WHERE node_id=?
            """,
            (
                heartbeat.updated_at,
                heartbeat.state,
                heartbeat.session_id,
                heartbeat.fps,
                heartbeat.sync_backlog,
                heartbeat.incident_count,
                heartbeat.last_error,
                json.dumps(heartbeat.extra),
                heartbeat.node_id,
            ),
        )
        self.connection.commit()

    def list_registered_nodes(self) -> dict[str, dict]:
        rows = self.connection.execute("SELECT * FROM nodes").fetchall()
        return {row["node_id"]: dict(row) for row in rows}

    def create_session(self, spec: SessionSpec) -> None:
        self.connection.execute(
            "UPDATE sessions SET status='superseded' WHERE status IN ('created', 'running', 'degraded') AND session_id != ?",
            (spec.session_id,),
        )
        self.connection.execute(
            """
            INSERT INTO sessions (
                session_id, subject_code, professor, session_date, start_time, end_time,
                notes, created_at, status, started_at, stopped_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'created', '', '')
            ON CONFLICT(session_id) DO UPDATE SET
                subject_code=excluded.subject_code,
                professor=excluded.professor,
                session_date=excluded.session_date,
                start_time=excluded.start_time,
                end_time=excluded.end_time,
                notes=excluded.notes
            """,
            (
                spec.session_id,
                spec.subject_code,
                spec.professor,
                spec.session_date,
                spec.start_time,
                spec.end_time,
                spec.notes,
                spec.created_at,
            ),
        )
        self.connection.commit()

    def update_session_status(self, session_id: str, status: str) -> None:
        started_at = utc_now_iso() if status in {"running", "degraded"} else None
        stopped_at = utc_now_iso() if status in {"stopped", "cleared"} else None
        row = self.connection.execute(
            "SELECT started_at FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        current_started = row["started_at"] if row else ""
        self.connection.execute(
            """
            UPDATE sessions
            SET status=?,
                started_at=?,
                stopped_at=?
            WHERE session_id=?
            """,
            (
                status,
                current_started or started_at or "",
                stopped_at or "",
                session_id,
            ),
        )
        self.connection.commit()

    def update_all_active_session_statuses(self, status: str) -> int:
        session_rows = self.connection.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE status IN ('created', 'running', 'degraded')
            """
        ).fetchall()
        for row in session_rows:
            self.update_session_status(row["session_id"], status)
        return len(session_rows)

    def get_session(self, session_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_active_session(self) -> dict | None:
        row = self.connection.execute(
            """
            SELECT * FROM sessions
            WHERE status IN ('created', 'running', 'degraded')
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        return dict(row) if row else None

    def list_sessions_history(self, limit: int = 20) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT
                sessions.*,
                COUNT(incidents.incident_id) AS incident_count
            FROM sessions
            LEFT JOIN incidents ON incidents.session_id = sessions.session_id
            GROUP BY sessions.session_id
            ORDER BY sessions.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_sessions_by_subject(self, subject_code: str) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT * FROM sessions
            WHERE subject_code=?
            ORDER BY created_at DESC
            """,
            (subject_code,),
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> int:
        cursor = self.connection.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        self.connection.commit()
        return int(cursor.rowcount or 0)

    def delete_sessions_by_subject(self, subject_code: str) -> int:
        cursor = self.connection.execute("DELETE FROM sessions WHERE subject_code=?", (subject_code,))
        self.connection.commit()
        return int(cursor.rowcount or 0)

    def upsert_incident(self, manifest: IncidentManifest) -> None:
        payload = manifest.to_dict()
        now_iso = utc_now_iso()
        self.connection.execute(
            """
            INSERT INTO incidents (
                incident_id, session_id, node_id, camera_label, behavior_type, type_label,
                student_numbers_json, created_at, display_time, review_status, poster_path,
                gif_path, frame_count, summary, sync_status, sync_attempts, asset_names_json,
                manifest_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(incident_id) DO UPDATE SET
                session_id=excluded.session_id,
                node_id=excluded.node_id,
                camera_label=excluded.camera_label,
                behavior_type=excluded.behavior_type,
                type_label=excluded.type_label,
                student_numbers_json=excluded.student_numbers_json,
                created_at=excluded.created_at,
                display_time=excluded.display_time,
                review_status=excluded.review_status,
                poster_path=CASE WHEN excluded.poster_path != '' THEN excluded.poster_path ELSE incidents.poster_path END,
                gif_path=CASE WHEN excluded.gif_path != '' THEN excluded.gif_path ELSE incidents.gif_path END,
                frame_count=excluded.frame_count,
                summary=excluded.summary,
                sync_status=excluded.sync_status,
                sync_attempts=excluded.sync_attempts,
                asset_names_json=excluded.asset_names_json,
                manifest_json=excluded.manifest_json,
                updated_at=excluded.updated_at
            """,
            (
                manifest.incident_id,
                manifest.session_id,
                manifest.node_id,
                manifest.camera_label,
                manifest.behavior_type,
                manifest.type_label,
                json.dumps(manifest.student_numbers),
                manifest.created_at,
                manifest.display_time,
                manifest.review_status,
                manifest.poster_path,
                manifest.gif_path,
                manifest.frame_count,
                manifest.summary,
                manifest.sync_status,
                manifest.sync_attempts,
                json.dumps(manifest.asset_names),
                json.dumps(payload),
                now_iso,
            ),
        )
        self.connection.commit()

    def attach_asset_path(self, incident_id: str, asset_type: str, relative_path: str) -> None:
        column = "poster_path" if asset_type == "poster" else "gif_path" if asset_type == "gif" else None
        if column is None:
            return
        self.connection.execute(
            f"UPDATE incidents SET {column}=?, updated_at=? WHERE incident_id=?",
            (relative_path, utc_now_iso(), incident_id),
        )
        self.connection.commit()

    def update_review_status(self, incident_id: str, review_status: str) -> dict | None:
        self.connection.execute(
            "UPDATE incidents SET review_status=?, updated_at=? WHERE incident_id=?",
            (review_status, utc_now_iso(), incident_id),
        )
        self.connection.commit()
        return self.get_incident(incident_id)

    def get_incident(self, incident_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM incidents WHERE incident_id=?",
            (incident_id,),
        ).fetchone()
        return self._row_to_incident(row) if row else None

    def list_incidents(self, limit: int = 120) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT * FROM incidents
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_incident(row) for row in rows]

    def delete_incidents_for_session(self, session_id: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS total FROM incidents WHERE session_id=?",
            (session_id,),
        ).fetchone()
        self.connection.execute("DELETE FROM incidents WHERE session_id=?", (session_id,))
        self.connection.commit()
        return int(row["total"] if row else 0)

    def _row_to_incident(self, row: sqlite3.Row) -> dict:
        item = dict(row)
        item["student_numbers"] = json.loads(item.pop("student_numbers_json"))
        item["asset_names"] = json.loads(item.pop("asset_names_json"))
        item["manifest"] = json.loads(item.pop("manifest_json"))
        return item

    def node_status_snapshot(self, known_nodes: dict, *, offline_after_sec: int) -> list[dict]:
        registered = self.list_registered_nodes()
        deadline = datetime.now(timezone.utc) - timedelta(seconds=offline_after_sec)
        snapshot = []
        for node_id, node_cfg in known_nodes.items():
            row = registered.get(node_id, {})
            last_seen_at = row.get("last_seen_at") or row.get("registered_at") or ""
            seen_dt = _parse_iso(last_seen_at)
            online = bool(seen_dt and seen_dt >= deadline)
            snapshot.append(
                {
                    "node_id": node_id,
                    "display_name": row.get("display_name") or node_cfg.display_name,
                    "camera_label": row.get("camera_label") or node_cfg.camera_label,
                    "profile": row.get("profile", ""),
                    "base_url": row.get("base_url", ""),
                    "agent_base_url": row.get("agent_base_url", ""),
                    "capabilities": json.loads(row.get("capabilities_json", "[]")),
                    "registered_at": row.get("registered_at", ""),
                    "last_seen_at": last_seen_at,
                    "state": row.get("state", "unregistered"),
                    "session_id": row.get("session_id", ""),
                    "fps": float(row.get("fps") or 0.0),
                    "sync_backlog": int(row.get("sync_backlog") or 0),
                    "incident_count": int(row.get("incident_count") or 0),
                    "last_error": row.get("last_error", ""),
                    "extra": json.loads(row.get("extra_json", "{}")),
                    "online": online,
                }
            )
        return snapshot
