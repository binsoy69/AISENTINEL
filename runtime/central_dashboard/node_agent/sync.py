"""Persistent sync queue used by the node agent."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3
from pathlib import Path

from central_dashboard.shared.dto import SyncQueueItem, make_id, utc_now_iso


def _payload_session_id(payload: dict) -> str:
    manifest = payload.get("manifest_payload")
    if isinstance(manifest, dict):
        return str(manifest.get("session_id") or "")
    return str(payload.get("session_id") or "")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_queue (
            item_id TEXT PRIMARY KEY,
            item_type TEXT NOT NULL,
            incident_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            next_retry_at TEXT NOT NULL,
            last_error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


class LocalSyncQueue:
    """Durable queue for manifest and evidence uploads."""

    def __init__(self, db_path: Path) -> None:
        self.connection = _connect(db_path)

    def enqueue(self, item_type: str, incident_id: str, node_id: str, payload: dict) -> str:
        item_id = make_id("queue")
        now_iso = utc_now_iso()
        self.connection.execute(
            """
            INSERT INTO sync_queue (
                item_id, item_type, incident_id, node_id, payload_json, attempts,
                next_retry_at, last_error, created_at
            ) VALUES (?, ?, ?, ?, ?, 0, ?, '', ?)
            """,
            (item_id, item_type, incident_id, node_id, json.dumps(payload), now_iso, now_iso),
        )
        self.connection.commit()
        return item_id

    def due_items(self, limit: int = 8) -> list[SyncQueueItem]:
        rows = self.connection.execute(
            """
            SELECT * FROM sync_queue
            WHERE next_retry_at <= ?
            ORDER BY created_at ASC, rowid ASC
            LIMIT ?
            """,
            (utc_now_iso(), limit),
        ).fetchall()
        return [
            SyncQueueItem(
                item_id=row["item_id"],
                item_type=row["item_type"],
                incident_id=row["incident_id"],
                node_id=row["node_id"],
                payload=json.loads(row["payload_json"]),
                attempts=int(row["attempts"]),
                next_retry_at=row["next_retry_at"],
                last_error=row["last_error"],
            )
            for row in rows
        ]

    def mark_done(self, item_id: str) -> None:
        self.connection.execute("DELETE FROM sync_queue WHERE item_id=?", (item_id,))
        self.connection.commit()

    def clear(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM sync_queue").fetchone()
        count = int(row["count"])
        if count:
            self.connection.execute("DELETE FROM sync_queue")
            self.connection.commit()
        return count

    def clear_except_session(self, session_id: str) -> int:
        target_session_id = str(session_id or "")
        rows = self.connection.execute("SELECT item_id, payload_json FROM sync_queue").fetchall()
        item_ids = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                item_ids.append(row["item_id"])
                continue
            if _payload_session_id(payload) != target_session_id:
                item_ids.append(row["item_id"])

        if not item_ids:
            return 0

        self.connection.executemany(
            "DELETE FROM sync_queue WHERE item_id=?",
            [(item_id,) for item_id in item_ids],
        )
        self.connection.commit()
        return len(item_ids)

    def has_pending_manifest(self, incident_id: str) -> bool:
        rows = self.connection.execute(
            """
            SELECT payload_json FROM sync_queue
            WHERE incident_id=? AND item_type='manifest'
            """,
            (incident_id,),
        ).fetchall()
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                return True
            manifest = payload.get("manifest_payload")
            if not isinstance(manifest, dict):
                return True
            if str(manifest.get("sync_status") or "").lower() != "recording":
                return True
        return False

    def purge_recording_manifests(self, incident_id: str) -> int:
        rows = self.connection.execute(
            """
            SELECT item_id, payload_json FROM sync_queue
            WHERE incident_id=? AND item_type='manifest'
            """,
            (incident_id,),
        ).fetchall()
        item_ids = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                continue
            manifest = payload.get("manifest_payload")
            if isinstance(manifest, dict) and str(manifest.get("sync_status") or "").lower() == "recording":
                item_ids.append(row["item_id"])

        if not item_ids:
            return 0

        self.connection.executemany(
            "DELETE FROM sync_queue WHERE item_id=?",
            [(item_id,) for item_id in item_ids],
        )
        self.connection.commit()
        return len(item_ids)

    def purge_asset_type(self, asset_type: str) -> int:
        rows = self.connection.execute(
            "SELECT item_id, payload_json FROM sync_queue WHERE item_type='asset'"
        ).fetchall()
        item_ids = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                continue
            if str(payload.get("asset_type") or "") == asset_type:
                item_ids.append(row["item_id"])

        if not item_ids:
            return 0

        self.connection.executemany(
            "DELETE FROM sync_queue WHERE item_id=?",
            [(item_id,) for item_id in item_ids],
        )
        self.connection.commit()
        return len(item_ids)

    def mark_retry(self, item: SyncQueueItem, error_message: str) -> None:
        delay_sec = min(60, max(2, 2 ** min(item.attempts + 1, 5)))
        next_retry_at = (
            datetime.now(timezone.utc) + timedelta(seconds=delay_sec)
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self.connection.execute(
            """
            UPDATE sync_queue
            SET attempts=?, next_retry_at=?, last_error=?
            WHERE item_id=?
            """,
            (item.attempts + 1, next_retry_at, str(error_message), item.item_id),
        )
        self.connection.commit()

    def backlog_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM sync_queue").fetchone()
        return int(row["count"])

    def close(self) -> None:
        self.connection.close()
