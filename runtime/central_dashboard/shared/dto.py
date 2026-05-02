"""Dataclasses for shared node/central contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass(slots=True)
class NodeDescriptor:
    node_id: str
    display_name: str
    camera_label: str
    base_url: str
    agent_base_url: str
    registered_at: str = field(default_factory=utc_now_iso)
    capabilities: list[str] = field(default_factory=lambda: ["raw", "annotated", "debug"])
    profile: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "NodeDescriptor":
        return cls(
            node_id=str(payload.get("node_id", "")).strip(),
            display_name=str(payload.get("display_name", "")).strip(),
            camera_label=str(payload.get("camera_label", "")).strip(),
            base_url=str(payload.get("base_url", "")).rstrip("/"),
            agent_base_url=str(
                payload.get("agent_base_url") or payload.get("base_url", "")
            ).rstrip("/"),
            registered_at=str(payload.get("registered_at") or utc_now_iso()),
            capabilities=list(payload.get("capabilities") or ["raw", "annotated", "debug"]),
            profile=str(payload.get("profile", "")).strip(),
        )


@dataclass(slots=True)
class NodeHeartbeat:
    node_id: str
    state: str
    session_id: str = ""
    fps: float = 0.0
    sync_backlog: int = 0
    incident_count: int = 0
    last_error: str = ""
    updated_at: str = field(default_factory=utc_now_iso)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "NodeHeartbeat":
        return cls(
            node_id=str(payload.get("node_id", "")).strip(),
            state=str(payload.get("state", "idle")).strip() or "idle",
            session_id=str(payload.get("session_id", "")).strip(),
            fps=float(payload.get("fps") or 0.0),
            sync_backlog=int(payload.get("sync_backlog") or 0),
            incident_count=int(payload.get("incident_count") or 0),
            last_error=str(payload.get("last_error", "")).strip(),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            extra=dict(payload.get("extra") or {}),
        )


@dataclass(slots=True)
class SessionSpec:
    session_id: str = ""
    subject_code: str = ""
    professor: str = ""
    session_date: str = ""
    start_time: str = ""
    end_time: str = ""
    notes: str = ""
    created_at: str = field(default_factory=utc_now_iso)

    def ensure_id(self) -> "SessionSpec":
        if not self.session_id:
            self.session_id = make_id("session")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "SessionSpec":
        spec = cls(
            session_id=str(payload.get("session_id", "")).strip(),
            subject_code=str(payload.get("subject_code", "")).strip(),
            professor=str(payload.get("professor", "")).strip(),
            session_date=str(payload.get("session_date", "")).strip(),
            start_time=str(payload.get("start_time", "")).strip(),
            end_time=str(payload.get("end_time", "")).strip(),
            notes=str(payload.get("notes", "")).strip(),
            created_at=str(payload.get("created_at") or utc_now_iso()),
        )
        return spec.ensure_id()


@dataclass(slots=True)
class SessionCommand:
    action: str
    session: SessionSpec
    issued_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "session": self.session.to_dict(),
            "issued_at": self.issued_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SessionCommand":
        return cls(
            action=str(payload.get("action", "")).strip(),
            session=SessionSpec.from_dict(payload.get("session") or {}),
            issued_at=str(payload.get("issued_at") or utc_now_iso()),
        )


@dataclass(slots=True)
class IncidentManifest:
    incident_id: str
    session_id: str
    node_id: str
    camera_label: str
    behavior_type: str
    type_label: str
    student_numbers: list[int]
    created_at: str
    display_time: str
    review_status: str = "unverified"
    poster_path: str = ""
    gif_path: str = ""
    frame_count: int = 0
    summary: str = ""
    sync_status: str = "pending"
    sync_attempts: int = 0
    asset_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "IncidentManifest":
        return cls(
            incident_id=str(payload.get("incident_id", "")).strip()
            or make_id("incident"),
            session_id=str(payload.get("session_id", "")).strip(),
            node_id=str(payload.get("node_id", "")).strip(),
            camera_label=str(payload.get("camera_label", "")).strip(),
            behavior_type=str(payload.get("behavior_type", "")).strip() or "motion",
            type_label=str(payload.get("type_label", "")).strip() or "Movement Spike",
            student_numbers=[
                int(value) for value in (payload.get("student_numbers") or [])
            ],
            created_at=str(payload.get("created_at") or utc_now_iso()),
            display_time=str(payload.get("display_time", "")).strip(),
            review_status=str(payload.get("review_status", "unverified")).strip()
            or "unverified",
            poster_path=str(payload.get("poster_path", "")).strip(),
            gif_path=str(payload.get("gif_path", "")).strip(),
            frame_count=int(payload.get("frame_count") or 0),
            summary=str(payload.get("summary", "")).strip(),
            sync_status=str(payload.get("sync_status", "pending")).strip() or "pending",
            sync_attempts=int(payload.get("sync_attempts") or 0),
            asset_names=[str(value) for value in (payload.get("asset_names") or [])],
        )


@dataclass(slots=True)
class EvidenceAsset:
    incident_id: str
    session_id: str
    node_id: str
    asset_type: str
    filename: str
    content_base64: str
    content_sha256: str
    size_bytes: int
    relative_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "EvidenceAsset":
        filename = str(
            payload.get("filename")
            or payload.get("name")
            or payload.get("file_name")
            or ""
        ).strip()
        asset_type = str(payload.get("asset_type") or "").strip()
        if not asset_type:
            lowered = filename.lower()
            asset_type = "gif" if lowered.endswith(".gif") else "poster"

        content_base64 = str(
            payload.get("content_base64")
            or payload.get("file_base64")
            or payload.get("content")
            or payload.get("data")
            or ""
        ).strip()
        if "," in content_base64 and ";base64" in content_base64.split(",", 1)[0]:
            content_base64 = content_base64.split(",", 1)[1].strip()

        return cls(
            incident_id=str(payload.get("incident_id", "")).strip(),
            session_id=str(payload.get("session_id", "")).strip(),
            node_id=str(payload.get("node_id", "")).strip(),
            asset_type=asset_type,
            filename=filename,
            content_base64=content_base64,
            content_sha256=str(payload.get("content_sha256", "")).strip(),
            size_bytes=int(payload.get("size_bytes") or 0),
            relative_path=str(payload.get("relative_path", "")).strip(),
        )


@dataclass(slots=True)
class CommandAck:
    ok: bool
    node_id: str
    action: str
    session_id: str = ""
    message: str = ""
    state: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CommandAck":
        return cls(
            ok=bool(payload.get("ok")),
            node_id=str(payload.get("node_id", "")).strip(),
            action=str(payload.get("action", "")).strip(),
            session_id=str(payload.get("session_id", "")).strip(),
            message=str(payload.get("message", "")).strip(),
            state=str(payload.get("state", "")).strip(),
        )
