"""Service layer for the central dashboard app."""

from __future__ import annotations

import shutil
from pathlib import Path
import base64
import binascii
import logging
import posixpath

from central_dashboard.shared.dto import (
    CommandAck,
    EvidenceAsset,
    IncidentManifest,
    NodeDescriptor,
    NodeHeartbeat,
    SessionCommand,
    SessionSpec,
)
from central_dashboard.shared.http import StdlibHttpClient


class CentralServiceManager:
    """Coordinates persistence, node control, evidence storage, and dashboard views."""

    def __init__(self, config, repository, http_client=None) -> None:
        self.config = config
        self.repository = repository
        self.http_client = http_client or StdlibHttpClient()
        self.logger = logging.getLogger(__name__)
        self.config.evidence_root.mkdir(parents=True, exist_ok=True)

    def _dispatch_command_to_nodes(self, session_spec: SessionSpec, action: str) -> tuple[list[dict], int, list[dict]]:
        command = SessionCommand(action=action, session=session_spec)
        node_rows = self.repository.node_status_snapshot(
            self.config.known_nodes,
            offline_after_sec=self.config.node_offline_after_sec,
        )
        results = []
        ok_count = 0

        for node in node_rows:
            node_id = node["node_id"]
            agent_base_url = str(node.get("agent_base_url", "")).rstrip("/")
            if not agent_base_url:
                results.append(
                    CommandAck(
                        ok=False,
                        node_id=node_id,
                        action=action,
                        session_id=session_spec.session_id,
                        state="unregistered",
                        message="Node has not registered an agent URL yet.",
                    ).to_dict()
                )
                continue

            result = self.http_client.post_json(
                f"{agent_base_url}/agent/v1/session/{action}",
                command.to_dict(),
                headers=self.build_node_headers(node_id),
                timeout=self.config.proxy_timeout_sec,
            )
            if result.ok and isinstance(result.json_data, dict):
                ack = CommandAck.from_dict(result.json_data)
                if ack.ok:
                    ok_count += 1
                    self.logger.info(
                        "node command acknowledged: node=%s action=%s state=%s",
                        node_id,
                        action,
                        ack.state,
                    )
                else:
                    self.logger.warning(
                        "node command rejected: node=%s action=%s message=%s",
                        node_id,
                        action,
                        ack.message,
                    )
                results.append(ack.to_dict())
            else:
                self.logger.warning(
                    "node command failed: node=%s action=%s status=%s message=%s",
                    node_id,
                    action,
                    result.status_code,
                    result.text,
                )
                results.append(
                    CommandAck(
                        ok=False,
                        node_id=node_id,
                        action=action,
                        session_id=session_spec.session_id,
                        state=node.get("state", ""),
                        message=result.text or "Node command failed.",
                    ).to_dict()
                )
        return results, ok_count, node_rows

    def _stop_session_if_active(self, session_row: dict | None) -> list[dict]:
        if session_row is None:
            return []
        active_session = self.repository.get_active_session()
        if active_session is None or active_session.get("session_id") != session_row.get("session_id"):
            return []
        try:
            results, _ok_count, _node_rows = self._dispatch_command_to_nodes(SessionSpec.from_dict(session_row), "stop")
        except Exception:
            results = []
        return results

    def _clear_session_storage(self, session_id: str) -> tuple[int, bool]:
        cleared_incidents = self.repository.delete_incidents_for_session(session_id)
        evidence_dir = self.safe_evidence_path(session_id)
        cleared_evidence = False
        if evidence_dir.exists() and evidence_dir.is_dir():
            shutil.rmtree(evidence_dir, ignore_errors=True)
            cleared_evidence = True
        return cleared_incidents, cleared_evidence

    def reset_runtime_sessions_on_startup(self) -> dict:
        session = self.repository.get_active_session()
        if session is None:
            return {"ok": True, "stopped_sessions": 0, "results": []}

        session_spec = SessionSpec.from_dict(session)
        try:
            results, _ok_count, _node_rows = self._dispatch_command_to_nodes(session_spec, "stop")
        except Exception:
            results = []
        stopped_sessions = self.repository.update_all_active_session_statuses("stopped")
        return {"ok": True, "stopped_sessions": stopped_sessions, "results": results}

    def shutdown_active_session(self) -> dict:
        session = self.repository.get_active_session()
        if session is None:
            return {"ok": True, "session": None, "results": []}

        session_spec = SessionSpec.from_dict(session)
        try:
            results, _ok_count, _node_rows = self._dispatch_command_to_nodes(session_spec, "stop")
        except Exception:
            results = []
        try:
            self.repository.update_session_status(session_spec.session_id, "stopped")
            session = self.repository.get_session(session_spec.session_id)
        except Exception:
            session = None
        return {"ok": True, "session": session, "results": results}

    def build_node_headers(self, node_id: str) -> dict[str, str]:
        known = self.config.known_nodes[node_id]
        return {
            "X-Node-Id": known.node_id,
            "X-Api-Key": known.api_key,
        }

    def register_node(self, descriptor: NodeDescriptor) -> dict:
        known = self.config.known_nodes[descriptor.node_id]
        normalized = NodeDescriptor(
            node_id=descriptor.node_id,
            display_name=descriptor.display_name or known.display_name,
            camera_label=descriptor.camera_label or known.camera_label,
            base_url=descriptor.base_url,
            agent_base_url=descriptor.agent_base_url or descriptor.base_url,
            registered_at=descriptor.registered_at,
            capabilities=descriptor.capabilities,
            profile=descriptor.profile,
        )
        self.repository.upsert_node_registration(normalized)
        self.logger.info(
            "node registered: node=%s agent=%s profile=%s",
            normalized.node_id,
            normalized.agent_base_url,
            normalized.profile or "-",
        )
        return {"ok": True, "node": normalized.to_dict()}

    def record_heartbeat(self, heartbeat: NodeHeartbeat) -> dict:
        self.repository.update_node_heartbeat(heartbeat)
        stream = heartbeat.extra.get("stream", {}) if isinstance(heartbeat.extra, dict) else {}
        self.logger.info(
            "node heartbeat: node=%s state=%s session=%s fps=%.1f backlog=%s stream=%s error=%s",
            heartbeat.node_id,
            heartbeat.state,
            heartbeat.session_id or "-",
            heartbeat.fps,
            heartbeat.sync_backlog,
            "ready" if stream.get("has_annotated_frame") or stream.get("has_raw_frame") else "waiting",
            heartbeat.last_error or "-",
        )
        return {"ok": True, "heartbeat": heartbeat.to_dict()}

    def create_session(self, payload: dict) -> dict:
        session = SessionSpec.from_dict(payload)
        if not session.subject_code:
            return {"ok": False, "error": "Subject code is required before creating a session.", "status_code": 400}
        if not session.professor:
            return {"ok": False, "error": "Professor is required before creating a session.", "status_code": 400}
        self.repository.create_session(session)
        return {"ok": True, "session": session.to_dict()}

    def dispatch_session_command(self, session_id: str, action: str) -> dict:
        session_row = self.repository.get_session(session_id)
        if session_row is None:
            return {"ok": False, "error": "Session not found.", "status_code": 404}

        session_spec = SessionSpec.from_dict(session_row)
        results, ok_count, node_rows = self._dispatch_command_to_nodes(session_spec, action)
        if action in {"start", "restart"}:
            status = "running" if ok_count == len(node_rows) and node_rows else "degraded" if ok_count > 0 else "error"
        else:
            status = "stopped"

        self.repository.update_session_status(session_id, status)
        return {
            "ok": ok_count > 0 or action == "stop",
            "session": self.repository.get_session(session_id),
            "results": results,
        }

    def clear_current_session(self) -> dict:
        session_row = self.repository.get_active_session()
        if session_row is None:
            return {"ok": False, "error": "No active session to clear.", "status_code": 404}

        session_spec = SessionSpec.from_dict(session_row)
        results, _ok_count, _node_rows = self._dispatch_command_to_nodes(session_spec, "stop")
        self.repository.update_session_status(session_spec.session_id, "cleared")
        return {
            "ok": True,
            "session": self.repository.get_session(session_spec.session_id),
            "results": results,
        }

    def clear_incidents(self, session_id: str | None = None) -> dict:
        active_session = self.repository.get_active_session()
        target_session_id = str(session_id or (active_session.get("session_id") if active_session else "")).strip()
        if not target_session_id:
            return {"ok": False, "error": "Select a session first before clearing records.", "status_code": 400}

        session_row = self.repository.get_session(target_session_id)
        if session_row is None:
            return {"ok": False, "error": "Session not found.", "status_code": 404}

        cleared_incidents, cleared_evidence = self._clear_session_storage(target_session_id)

        return {
            "ok": True,
            "session": session_row,
            "cleared_incidents": cleared_incidents,
            "cleared_evidence": cleared_evidence,
        }

    def delete_session(self, session_id: str) -> dict:
        target_session_id = str(session_id or "").strip()
        if not target_session_id:
            return {"ok": False, "error": "Session not found.", "status_code": 404}

        session_row = self.repository.get_session(target_session_id)
        if session_row is None:
            return {"ok": False, "error": "Session not found.", "status_code": 404}

        results = self._stop_session_if_active(session_row)
        cleared_incidents, cleared_evidence = self._clear_session_storage(target_session_id)
        deleted_sessions = self.repository.delete_session(target_session_id)
        return {
            "ok": deleted_sessions > 0,
            "deleted_session": session_row,
            "deleted_sessions": deleted_sessions,
            "cleared_incidents": cleared_incidents,
            "cleared_evidence": cleared_evidence,
            "results": results,
        }

    def delete_subject_sessions(self, subject_code: str) -> dict:
        target_subject_code = str(subject_code or "").strip()
        if not target_subject_code:
            return {"ok": False, "error": "Subject code is required to delete stored sessions.", "status_code": 400}

        session_rows = self.repository.list_sessions_by_subject(target_subject_code)
        if not session_rows:
            return {"ok": False, "error": "No stored sessions were found for that subject code.", "status_code": 404}

        results = []
        active_session = self.repository.get_active_session()
        if active_session and str(active_session.get("subject_code", "")).strip() == target_subject_code:
            results = self._stop_session_if_active(active_session)

        cleared_incidents = 0
        cleared_evidence = 0
        for session_row in session_rows:
            incident_count, had_evidence = self._clear_session_storage(session_row["session_id"])
            cleared_incidents += incident_count
            cleared_evidence += int(had_evidence)

        deleted_sessions = self.repository.delete_sessions_by_subject(target_subject_code)
        return {
            "ok": deleted_sessions > 0,
            "subject_code": target_subject_code,
            "deleted_sessions": deleted_sessions,
            "cleared_incidents": cleared_incidents,
            "cleared_evidence": cleared_evidence,
            "results": results,
        }

    def upsert_incident(self, payload: dict) -> dict:
        manifest = IncidentManifest.from_dict(payload)
        self.repository.upsert_incident(manifest)
        return {"ok": True, "incident": self._public_incident(manifest.incident_id)}

    def store_asset(self, payload: dict) -> dict:
        try:
            asset = EvidenceAsset.from_dict(payload)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"Invalid evidence payload: {exc}", "status_code": 400}

        if not asset.incident_id:
            return {"ok": False, "error": "Missing required evidence fields.", "status_code": 400}

        incident = self.repository.get_incident(asset.incident_id)
        if incident is None:
            return {"ok": False, "error": "Incident not found.", "status_code": 404}

        if not asset.session_id:
            asset.session_id = str(incident.get("session_id", "")).strip()
        asset.asset_type = _normalize_asset_type(asset.asset_type, asset.filename)
        asset.filename = _normalize_asset_filename(asset.asset_type, asset.filename)

        if not asset.session_id or not asset.filename or not asset.content_base64:
            return {"ok": False, "error": "Missing required evidence fields.", "status_code": 400}

        try:
            content = base64.b64decode(asset.content_base64.encode("ascii"), validate=True)
        except (binascii.Error, ValueError) as exc:
            return {"ok": False, "error": f"Invalid evidence content: {exc}", "status_code": 400}

        relative_path = posixpath.join(
            asset.session_id,
            asset.node_id,
            asset.incident_id,
            asset.filename,
        )
        try:
            file_path = self.safe_evidence_path(relative_path)
        except ValueError:
            return {"ok": False, "error": "Invalid evidence path.", "status_code": 400}
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        self.repository.attach_asset_path(asset.incident_id, asset.asset_type, relative_path)
        return {"ok": True, "relative_path": relative_path}

    def update_review_status(self, incident_id: str, review_status: str) -> dict:
        normalized = str(review_status or "unverified").strip() or "unverified"
        updated = self.repository.update_review_status(incident_id, normalized)
        if updated is None:
            return {"ok": False, "error": "Incident not found.", "status_code": 404}
        return {"ok": True, "incident": self._public_incident(incident_id)}

    def dashboard_snapshot(self) -> dict:
        active_session = self.repository.get_active_session()
        nodes = self.repository.node_status_snapshot(
            self.config.known_nodes,
            offline_after_sec=self.config.node_offline_after_sec,
        )
        for node in nodes:
            node["stream_urls"] = {
                "raw": f"/api/v1/streams/{node['node_id']}/raw",
                "annotated": f"/api/v1/streams/{node['node_id']}/annotated",
            }
        incidents = [self._public_incident(row["incident_id"]) for row in self.repository.list_incidents()]
        return {
            "active_session": active_session,
            "nodes": nodes,
            "incidents": incidents,
            "sessions_history": self.repository.list_sessions_history(),
        }

    def open_node_stream(self, node_id: str, mode: str):
        nodes = self.repository.node_status_snapshot(
            self.config.known_nodes,
            offline_after_sec=self.config.node_offline_after_sec,
        )
        match = next((node for node in nodes if node["node_id"] == node_id), None)
        if match is None or not match.get("agent_base_url") or not match.get("online"):
            raise FileNotFoundError(f"Node stream not available for {node_id}")
        url = f"{str(match['agent_base_url']).rstrip('/')}/agent/v1/stream/{mode}"
        self.logger.info("opening node stream: node=%s mode=%s url=%s", node_id, mode, url)
        return self.http_client.open_stream(
            url,
            headers=self.build_node_headers(node_id),
            timeout=self.config.stream_timeout_sec,
        )

    def safe_evidence_path(self, relative_path: str) -> Path:
        candidate = (self.config.evidence_root / relative_path).resolve(strict=False)
        root = self.config.evidence_root.resolve(strict=False)
        if candidate != root and root not in candidate.parents:
            raise ValueError("Evidence path is outside the configured evidence root.")
        return candidate

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


def _normalize_asset_type(asset_type: str, filename: str) -> str:
    normalized = str(asset_type or "").strip().lower()
    if normalized in {"poster", "gif", "frame"}:
        return normalized
    lowered = str(filename or "").lower()
    return "gif" if lowered.endswith(".gif") else "poster"


def _normalize_asset_filename(asset_type: str, filename: str) -> str:
    if asset_type == "poster":
        return "poster.jpg"
    if asset_type == "gif":
        return "evidence.gif"

    normalized = str(filename or "").replace("\\", "/").strip()
    normalized = posixpath.normpath(normalized).lstrip("/")
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        return ""
    return normalized
