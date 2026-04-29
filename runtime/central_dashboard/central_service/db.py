"""SQLite helpers for the central dashboard service."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def init_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            camera_label TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT '',
            base_url TEXT NOT NULL DEFAULT '',
            agent_base_url TEXT NOT NULL DEFAULT '',
            capabilities_json TEXT NOT NULL DEFAULT '[]',
            registered_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL DEFAULT 'unknown',
            session_id TEXT NOT NULL DEFAULT '',
            fps REAL NOT NULL DEFAULT 0.0,
            sync_backlog INTEGER NOT NULL DEFAULT 0,
            incident_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            extra_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            subject_code TEXT NOT NULL,
            professor TEXT NOT NULL,
            session_date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'created',
            started_at TEXT NOT NULL DEFAULT '',
            stopped_at TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS incidents (
            incident_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            camera_label TEXT NOT NULL,
            behavior_type TEXT NOT NULL,
            type_label TEXT NOT NULL,
            student_numbers_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            display_time TEXT NOT NULL DEFAULT '',
            review_status TEXT NOT NULL DEFAULT 'unverified',
            poster_path TEXT NOT NULL DEFAULT '',
            gif_path TEXT NOT NULL DEFAULT '',
            frame_count INTEGER NOT NULL DEFAULT 0,
            summary TEXT NOT NULL DEFAULT '',
            sync_status TEXT NOT NULL DEFAULT 'pending',
            sync_attempts INTEGER NOT NULL DEFAULT 0,
            asset_names_json TEXT NOT NULL DEFAULT '[]',
            manifest_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_subject_created
            ON sessions(subject_code, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_sessions_status_created
            ON sessions(status, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_incidents_session_created
            ON incidents(session_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_incidents_subject_lookup
            ON incidents(session_id, node_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_incidents_created
            ON incidents(created_at DESC);
        """
    )
    connection.commit()
