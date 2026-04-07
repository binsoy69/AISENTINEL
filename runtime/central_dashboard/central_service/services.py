"""Service layer for the central dashboard app."""

from __future__ import annotations

from pathlib import Path
import base64
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
        self.config.evidence_root.mkdir(parents=True, exist_ok=True)

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
        return {"ok": True, "node": normalized.to_dict()}

    def record_heartbeat(self, heartbeat: NodeHeartbeat) -> dict:
        self.repository.update_node_heartbeat(heartbeat)
        return {"ok": True, "heartbeat": heartbeat.to_dict()}

    def create_session(self, payload: dict) -> dict:
        session = SessionSpec.from_dict(payload)
        self.repository.create_session(session)
        return {"ok": True, "session": session.to_dict()}

    def dispatch_session_command(self, session_id: str, action: str) -> dict:
        session_row = self.repository.get_session(session_id)
        if session_row is None:
            return {"ok": False, "error": "Session not found.", "status_code": 404}

        session_spec = SessionSpec.from_dict(session_row)
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
                        session_id=session_id,
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
                results.append(ack.to_dict())
            else:
                results.append(
                    CommandAck(
                        ok=False,
                        node_id=node_id,
                        action=action,
                        session_id=session_id,
                        state=node.get("state", ""),
                        message=result.text or "Node command failed.",
                    ).to_dict()
                )

        if action in {"start", "restart"}:
            status = "running" if ok_count == len(node_rows) and node_rows else "degraded" if ok_count > 0 else "error"
        else:
            status = "stopped" if ok_count == len(node_rows) and node_rows else "degraded" if ok_count > 0 else "stopped"

        self.repository.update_session_status(session_id, status)
        return {
            "ok": ok_count > 0 or action == "stop",
            "session": self.repository.get_session(session_id),
            "results": results,
        }

    def upsert_incident(self, payload: dict) -> dict:
        manifest = IncidentManifest.from_dict(payload)
        self.repository.upsert_incident(manifest)
        return {"ok": True, "incident": self._public_incident(manifest.incident_id)}

    def store_asset(self, payload: dict) -> dict:
        asset = EvidenceAsset.from_dict(payload)
        incident = self.repository.get_incident(asset.incident_id)
        if incident is None:
            return {"ok": False, "error": "Incident not found.", "status_code": 404}

        content = base64.b64decode(asset.content_base64.encode("ascii"))
        relative_path = posixpath.join(
            asset.session_id,
            asset.node_id,
            asset.incident_id,
            asset.filename,
        )
        file_path = self.safe_evidence_path(relative_path)
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
        }

    def open_node_stream(self, node_id: str, mode: str):
        nodes = self.repository.node_status_snapshot(
            self.config.known_nodes,
            offline_after_sec=self.config.node_offline_after_sec,
        )
        match = next((node for node in nodes if node["node_id"] == node_id), None)
        if match is None or not match.get("agent_base_url"):
            raise FileNotFoundError(f"Node stream not available for {node_id}")
        url = f"{str(match['agent_base_url']).rstrip('/')}/agent/v1/stream/{mode}"
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
