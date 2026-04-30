"""Incident manifest and evidence normalization helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path, PurePosixPath

import cv2

from central_dashboard.shared.dto import IncidentManifest, SessionSpec, make_id, utc_now_iso
from .sound_monitor import build_noise_summary


def normalize_front_runtime_incident(
    *,
    node_config,
    session: SessionSpec,
    evidence_root: Path,
    front_manifest: dict,
) -> tuple[IncidentManifest, list[dict]]:
    poster_relpath = best_poster_relpath(front_manifest)
    gif_relpath = str(front_manifest.get("gif_relpath") or "").strip()
    behavior_type = str(front_manifest.get("behavior_type", "")).strip() or "object"
    assets = []
    asset_names = []

    if poster_relpath:
        asset_names.append("poster.jpg")
        assets.append(
            {
                "asset_type": "poster",
                "file_path": local_evidence_path(evidence_root, poster_relpath),
                "filename": asset_names[-1],
            }
        )
    if gif_relpath and behavior_type.lower() != "noise":
        asset_names.append("evidence.gif")
        assets.append(
            {
                "asset_type": "gif",
                "file_path": local_evidence_path(evidence_root, gif_relpath),
                "filename": "evidence.gif",
            }
        )

    manifest = IncidentManifest(
        incident_id=str(front_manifest.get("id", "")).strip(),
        session_id=session.session_id,
        node_id=node_config.node_id,
        camera_label=node_config.camera_label,
        behavior_type=behavior_type,
        type_label=str(front_manifest.get("type_label", "")).strip() or "Incident",
        student_numbers=[
            int(value) for value in (front_manifest.get("student_numbers") or [])
        ],
        created_at=str(front_manifest.get("created_at") or utc_now_iso()),
        display_time=str(front_manifest.get("display_time", "")).strip(),
        review_status=str(front_manifest.get("review_status", "unverified")).strip()
        or "unverified",
        poster_path="",
        gif_path="",
        frame_count=int(front_manifest.get("frame_count") or (5 if gif_relpath else 1 if poster_relpath else 0)),
        summary=str(front_manifest.get("summary", "")).strip(),
        sync_status="ready",
        sync_attempts=0,
        asset_names=asset_names,
    )
    return manifest, assets


def normalize_front_runtime_detected_incident(
    *,
    node_config,
    session: SessionSpec,
    front_manifest: dict,
) -> IncidentManifest:
    return IncidentManifest(
        incident_id=str(front_manifest.get("id", "")).strip(),
        session_id=session.session_id,
        node_id=node_config.node_id,
        camera_label=node_config.camera_label,
        behavior_type=str(front_manifest.get("behavior_type", "")).strip()
        or "object",
        type_label=str(front_manifest.get("type_label", "")).strip()
        or "Incident",
        student_numbers=[
            int(value) for value in (front_manifest.get("student_numbers") or [])
        ],
        created_at=str(front_manifest.get("created_at") or utc_now_iso()),
        display_time=str(front_manifest.get("display_time", "")).strip(),
        review_status=str(front_manifest.get("review_status", "unverified")).strip()
        or "unverified",
        poster_path="",
        gif_path="",
        frame_count=int(front_manifest.get("frame_count") or 0),
        summary=str(front_manifest.get("summary", "")).strip(),
        sync_status=str(front_manifest.get("status") or "recording").strip()
        or "recording",
        sync_attempts=0,
        asset_names=[],
    )


def build_noise_incident_evidence(
    *,
    node_config,
    session: SessionSpec,
    source_label: str,
    estimated_db: float,
    threshold_db: float,
    frame,
) -> tuple[IncidentManifest, list[dict]]:
    incident_id = make_id("noise")
    assets = []
    asset_names = []
    frame_count = 0

    if frame is not None:
        incident_dir = node_config.evidence_root / session.session_id / incident_id
        incident_dir.mkdir(parents=True, exist_ok=True)
        poster_path = incident_dir / "poster.jpg"
        if cv2.imwrite(str(poster_path), frame):
            asset_names.append("poster.jpg")
            assets.append(
                {
                    "asset_type": "poster",
                    "file_path": poster_path,
                    "filename": "poster.jpg",
                }
            )
            frame_count = 1

    manifest = IncidentManifest(
        incident_id=incident_id,
        session_id=session.session_id,
        node_id=node_config.node_id,
        camera_label=node_config.camera_label or source_label,
        behavior_type="noise",
        type_label="Noise Threshold Exceeded",
        student_numbers=[],
        created_at=utc_now_iso(),
        display_time=datetime.now().strftime("%I:%M %p").lstrip("0"),
        review_status="unverified",
        poster_path="",
        gif_path="",
        frame_count=frame_count,
        summary=build_noise_summary(estimated_db, threshold_db),
        sync_status="ready",
        sync_attempts=0,
        asset_names=asset_names,
    )
    return manifest, assets


def best_poster_relpath(front_manifest: dict) -> str:
    poster_relpath = str(front_manifest.get("poster_relpath") or "").strip()
    if poster_relpath:
        return poster_relpath

    frame_relpaths = [
        str(value).strip()
        for value in (front_manifest.get("frame_relpaths") or [])
        if str(value).strip()
    ]
    if not frame_relpaths:
        return ""

    event_matches = [
        value for value in frame_relpaths if "_event" in PurePosixPath(value).name
    ]
    return (
        event_matches[0]
        if event_matches
        else frame_relpaths[len(frame_relpaths) // 2]
    )


def local_evidence_path(evidence_root: Path, relpath: str) -> Path:
    rel = PurePosixPath(relpath)
    return evidence_root.joinpath(*rel.parts)

