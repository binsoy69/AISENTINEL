"""Service layer for the editable demo dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
import posixpath
import shutil
from pathlib import Path
from typing import Mapping

from werkzeug.datastructures import MultiDict

from central_dashboard.shared.dto import IncidentManifest, SessionSpec, utc_now_iso


ALLOWED_MEDIA_EXTENSIONS = {".gif", ".jpg", ".jpeg", ".png"}
DEFAULT_EDITABLE_TYPE_LABEL = "Using Phone"


class EditableDashboardManager:
    """Coordinates isolated demo sessions, simulated nodes, and manual evidence."""

    def __init__(self, config, repository) -> None:
        self.config = config
        self.repository = repository
        self.config.evidence_root.mkdir(parents=True, exist_ok=True)

    def create_session(self, payload: dict) -> dict:
        session = SessionSpec.from_dict(payload)
        if not session.subject_code:
            return {"ok": False, "error": "Subject code is required before creating a session.", "status_code": 400}
        if not session.professor:
            return {"ok": False, "error": "Professor is required before creating a session.", "status_code": 400}
        self.repository.create_session(session)
        return {"ok": True, "session": session.to_dict()}

    def dispatch_session_command(self, session_id: str, action: str) -> dict:
        session_row = self.repository.get_session(str(session_id or "").strip())
        if session_row is None:
            return {"ok": False, "error": "Session not found.", "status_code": 404}

        status = "stopped" if action == "stop" else "running"
        self.repository.update_session_status(session_row["session_id"], status)
        session = self.repository.get_session(session_row["session_id"])
        return {
            "ok": True,
            "session": session,
            "results": [
                {
                    "ok": True,
                    "node_id": node_id,
                    "action": action,
                    "session_id": session_row["session_id"],
                    "state": "idle" if action == "stop" else "running",
                    "message": "Demo node state updated.",
                }
                for node_id in self.config.known_nodes
            ],
        }

    def clear_current_session(self) -> dict:
        session_row = self.repository.get_active_session()
        if session_row is None:
            return {"ok": False, "error": "No active session to clear.", "status_code": 404}
        self.repository.update_session_status(session_row["session_id"], "cleared")
        return {"ok": True, "session": self.repository.get_session(session_row["session_id"]), "results": []}

    def clear_incidents(self, session_id: str | None = None) -> dict:
        active_session = self.repository.get_active_session()
        target_session_id = str(session_id or (active_session.get("session_id") if active_session else "")).strip()
        if not target_session_id:
            return {"ok": False, "error": "Select a session first before clearing records.", "status_code": 400}
        session_row = self.repository.get_session(target_session_id)
        if session_row is None:
            return {"ok": False, "error": "Session not found.", "status_code": 404}

        cleared_incidents = self.repository.delete_incidents_for_session(target_session_id)
        evidence_dir = self.safe_evidence_path(target_session_id)
        cleared_evidence = False
        if evidence_dir.exists() and evidence_dir.is_dir():
            shutil.rmtree(evidence_dir, ignore_errors=True)
            cleared_evidence = True
        return {
            "ok": True,
            "session": session_row,
            "cleared_incidents": cleared_incidents,
            "cleared_evidence": cleared_evidence,
        }

    def delete_session(self, session_id: str) -> dict:
        target_session_id = str(session_id or "").strip()
        session_row = self.repository.get_session(target_session_id)
        if session_row is None:
            return {"ok": False, "error": "Session not found.", "status_code": 404}
        cleared_incidents, cleared_evidence = self._clear_session_storage(target_session_id)
        deleted_sessions = self.repository.delete_session(target_session_id)
        return {
            "ok": deleted_sessions > 0,
            "deleted_session": session_row,
            "deleted_sessions": deleted_sessions,
            "cleared_incidents": cleared_incidents,
            "cleared_evidence": cleared_evidence,
            "results": [],
        }

    def delete_subject_sessions(self, subject_code: str) -> dict:
        target_subject_code = str(subject_code or "").strip()
        if not target_subject_code:
            return {"ok": False, "error": "Subject code is required to delete stored sessions.", "status_code": 400}
        sessions = self.repository.list_sessions_by_subject(target_subject_code)
        if not sessions:
            return {"ok": False, "error": "No stored sessions were found for that subject code.", "status_code": 404}

        cleared_incidents = 0
        cleared_evidence = 0
        for session in sessions:
            incident_count, had_evidence = self._clear_session_storage(session["session_id"])
            cleared_incidents += incident_count
            cleared_evidence += int(had_evidence)
        deleted_sessions = self.repository.delete_sessions_by_subject(target_subject_code)
        return {
            "ok": deleted_sessions > 0,
            "subject_code": target_subject_code,
            "deleted_sessions": deleted_sessions,
            "cleared_incidents": cleared_incidents,
            "cleared_evidence": cleared_evidence,
            "results": [],
        }

    def save_editable_incidents(self, records: list[dict], files: MultiDict) -> dict:
        active_session = self.repository.get_active_session()
        if active_session is None:
            return {"ok": False, "error": "Create a demo session before saving evidence.", "status_code": 400}

        session_id = active_session["session_id"]
        saved: list[dict] = []
        for record in records:
            saved.append(self._save_editable_incident(session_id, record, files))

        return {
            "ok": True,
            "saved_count": len(saved),
            "incidents": self.session_incidents(session_id)["incidents"],
        }

    def delete_editable_incident(self, incident_id: str) -> dict:
        target_id = str(incident_id or "").strip()
        deleted = self.repository.delete_incident(target_id)
        if deleted is None:
            return {"ok": False, "error": "Incident not found.", "status_code": 404}
        self._remove_incident_storage(deleted)
        return {
            "ok": True,
            "deleted_incident": deleted,
            "incidents": self.session_incidents(deleted["session_id"])["incidents"],
        }

    def update_review_status(self, incident_id: str, review_status: str) -> dict:
        normalized = str(review_status or "unverified").strip() or "unverified"
        updated = self.repository.update_review_status(incident_id, normalized)
        if updated is None:
            return {"ok": False, "error": "Incident not found.", "status_code": 404}
        return {"ok": True, "incident": self._public_incident(incident_id)}

    def dashboard_snapshot(self) -> dict:
        active_session = self.repository.get_active_session()
        active_session_id = str(active_session.get("session_id") or "").strip() if active_session else ""
        incidents = self.session_incidents(active_session_id)["incidents"] if active_session_id else []
        return {
            "active_session": active_session,
            "nodes": self._simulated_nodes(active_session),
            "incidents": incidents,
            "sessions_history": self.repository.list_sessions_history(),
            "editable_demo": True,
        }

    def session_incidents(self, session_id: str) -> dict:
        target_session_id = str(session_id or "").strip()
        if not target_session_id or self.repository.get_session(target_session_id) is None:
            return {"ok": False, "error": "Session not found.", "status_code": 404}
        incidents = [
            self._public_incident(row["incident_id"])
            for row in self.repository.list_incidents(limit=None, session_id=target_session_id)
        ]
        return {"ok": True, "session_id": target_session_id, "incidents": incidents}

    def safe_evidence_path(self, relative_path: str) -> Path:
        candidate = (self.config.evidence_root / relative_path).resolve(strict=False)
        root = self.config.evidence_root.resolve(strict=False)
        if candidate != root and root not in candidate.parents:
            raise ValueError("Evidence path is outside the configured evidence root.")
        return candidate

    def _save_editable_incident(self, session_id: str, record: dict, files: MultiDict) -> dict:
        incident_id = str(record.get("incident_id") or "").strip() or self._manual_incident_id()
        existing = self.repository.get_incident(incident_id)
        timestamp = str(record.get("timestamp") or record.get("created_at") or "").strip()
        created_at = _normalize_timestamp(timestamp)
        type_label = str(record.get("type_label") or record.get("cheating_type") or DEFAULT_EDITABLE_TYPE_LABEL).strip() or DEFAULT_EDITABLE_TYPE_LABEL
        camera_label = str(record.get("camera_label") or record.get("camera") or "").strip()
        node_id, camera_label = self._resolve_node(camera_label, str(record.get("node_id") or ""))

        poster_path = str(existing.get("poster_path") or "") if existing else ""
        gif_path = str(existing.get("gif_path") or "") if existing else ""
        asset_names = list(existing.get("asset_names") or []) if existing else []
        uploaded = files.get(str(record.get("file_field") or ""))
        if uploaded and uploaded.filename:
            poster_path, gif_path, asset_names = self._store_uploaded_media(
                uploaded,
                session_id=session_id,
                node_id=node_id,
                incident_id=incident_id,
            )

        manifest = IncidentManifest(
            incident_id=incident_id,
            session_id=session_id,
            node_id=node_id,
            camera_label=camera_label,
            behavior_type=_behavior_type_from_label(type_label),
            type_label=type_label,
            student_numbers=_parse_student_numbers(record.get("student_numbers") or record.get("seat_numbers") or record.get("seat_no")),
            created_at=created_at,
            display_time=_display_time(created_at),
            review_status=str(record.get("review_status") or (existing or {}).get("review_status") or "unverified"),
            poster_path=poster_path,
            gif_path=gif_path,
            frame_count=1 if poster_path or gif_path else 0,
            summary=str(record.get("summary") or "").strip() or _summary_for(type_label, record),
            sync_status="ready",
            sync_attempts=0,
            asset_names=asset_names,
        )
        self.repository.upsert_incident(manifest)
        self.repository.set_incident_asset_paths(incident_id, poster_path=poster_path, gif_path=gif_path)
        return self._public_incident(incident_id)

    def _store_uploaded_media(self, uploaded, *, session_id: str, node_id: str, incident_id: str) -> tuple[str, str, list[str]]:
        suffix = Path(uploaded.filename or "").suffix.lower()
        if suffix not in ALLOWED_MEDIA_EXTENSIONS:
            raise ValueError("Evidence media must be a GIF, JPG, or PNG file.")
        is_gif = suffix == ".gif"
        filename = "evidence.gif" if is_gif else f"poster{suffix if suffix else '.jpg'}"
        relative_path = posixpath.join(session_id, node_id, incident_id, filename)
        file_path = self.safe_evidence_path(relative_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        uploaded.save(file_path)
        return ("", relative_path, [filename]) if is_gif else (relative_path, "", [filename])

    def _resolve_node(self, camera_label: str, node_id: str) -> tuple[str, str]:
        if node_id in self.config.known_nodes:
            known = self.config.known_nodes[node_id]
            return known.node_id, camera_label or known.camera_label
        normalized_camera = camera_label.lower()
        for known in self.config.known_nodes.values():
            if normalized_camera in {known.camera_label.lower(), known.display_name.lower(), known.node_id.lower()}:
                return known.node_id, known.camera_label
        first = next(iter(self.config.known_nodes.values()), None)
        if first is None:
            return node_id or "demo", camera_label or "Demo Camera"
        return first.node_id, camera_label or first.camera_label

    def _simulated_nodes(self, active_session: Mapping | None) -> list[dict]:
        now = utc_now_iso()
        session_id = str(active_session.get("session_id") or "").strip() if active_session else ""
        session_status = str(active_session.get("status") or "").lower() if active_session else ""
        state = "running" if session_status in {"running", "degraded"} else "idle"
        nodes = []
        for known in self.config.known_nodes.values():
            nodes.append(
                {
                    "node_id": known.node_id,
                    "display_name": known.display_name,
                    "camera_label": known.camera_label,
                    "profile": known.node_id,
                    "base_url": "",
                    "agent_base_url": "",
                    "capabilities": ["raw", "annotated", "debug"],
                    "registered_at": now,
                    "last_seen_at": now,
                    "state": state,
                    "session_id": session_id if state == "running" else "",
                    "fps": 0.0,
                    "sync_backlog": 0,
                    "incident_count": len(self.repository.list_incidents(limit=None, session_id=session_id)) if session_id else 0,
                    "last_error": "",
                    "extra": {
                        "stream": {
                            "has_raw_frame": False,
                            "has_annotated_frame": False,
                            "has_debug_frame": False,
                        },
                        "sound": {"enabled": False},
                    },
                    "online": True,
                    "dropped_upload_count": 0,
                    "last_dropped_upload_at": "",
                    "last_dropped_upload_error": "",
                    "stream_urls": {
                        "raw": f"/api/v1/streams/{known.node_id}/raw",
                        "annotated": f"/api/v1/streams/{known.node_id}/annotated",
                        "debug": f"/api/v1/streams/{known.node_id}/debug",
                    },
                }
            )
        return nodes

    def _clear_session_storage(self, session_id: str) -> tuple[int, bool]:
        cleared_incidents = self.repository.delete_incidents_for_session(session_id)
        evidence_dir = self.safe_evidence_path(session_id)
        cleared_evidence = False
        if evidence_dir.exists() and evidence_dir.is_dir():
            shutil.rmtree(evidence_dir, ignore_errors=True)
            cleared_evidence = True
        return cleared_incidents, cleared_evidence

    def _remove_incident_storage(self, incident: dict) -> None:
        relative_dir = posixpath.join(
            str(incident.get("session_id") or ""),
            str(incident.get("node_id") or ""),
            str(incident.get("incident_id") or ""),
        )
        try:
            incident_dir = self.safe_evidence_path(relative_dir)
        except ValueError:
            return
        if incident_dir.exists() and incident_dir.is_dir():
            shutil.rmtree(incident_dir, ignore_errors=True)

    def _public_incident(self, incident_id: str) -> dict:
        incident = self.repository.get_incident(incident_id)
        if incident is None:
            raise FileNotFoundError(incident_id)
        poster_path = incident.get("poster_path", "")
        gif_path = incident.get("gif_path", "")
        item = dict(incident)
        item["poster_url"] = f"/api/v1/evidence/{poster_path}" if poster_path else ""
        item["gif_url"] = f"/api/v1/evidence/{gif_path}" if gif_path else ""
        return item

    @staticmethod
    def _manual_incident_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        return f"incident-manual-{stamp}"


def _normalize_timestamp(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return utc_now_iso()
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return utc_now_iso()
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _display_time(created_at: str) -> str:
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return ""
    return parsed.astimezone().strftime("%I:%M %p").lstrip("0")


def _parse_student_numbers(value) -> list[int]:
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = str(value or "").replace(";", ",").split(",")
    numbers = []
    for raw in raw_values:
        text = str(raw).strip()
        if not text:
            continue
        try:
            number = int(text)
        except ValueError:
            continue
        if number > 0 and number not in numbers:
            numbers.append(number)
    return numbers


def _behavior_type_from_label(label: str) -> str:
    normalized = str(label or "").lower()
    if "noise" in normalized:
        return "noise"
    if "phone" in normalized or "object" in normalized or "sheet" in normalized:
        return "object"
    if "paper" in normalized or "passing" in normalized:
        return "passing_papers"
    if "hand" in normalized:
        return "hands_under_table"
    if "head" in normalized:
        return "head_behavior"
    return "manual"


def _summary_for(type_label: str, record: dict) -> str:
    seats = _parse_student_numbers(record.get("student_numbers") or record.get("seat_numbers") or record.get("seat_no"))
    if seats:
        labels = ", ".join(f"#{seat:02d}" for seat in seats)
        return f"Manual demo evidence for seat {labels}: {type_label}."
    return f"Manual demo evidence: {type_label}."
