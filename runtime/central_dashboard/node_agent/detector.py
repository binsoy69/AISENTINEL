"""Local frame detector and evidence helpers for the standalone node agent."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import time

import cv2

from central_dashboard.shared.dto import IncidentManifest, make_id, utc_now_iso


try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional on target systems
    PIL_AVAILABLE = False


def display_clock_now() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


class MotionDetector:
    """Simple local detector that flags large motion spikes."""

    def __init__(self, threshold: float, min_area_ratio: float, cooldown_sec: float) -> None:
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
        self.cooldown_sec = cooldown_sec
        self.prev_gray = None
        self.last_trigger_at = 0.0

    def analyze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.prev_gray is None:
            self.prev_gray = blurred
            return None, {"mean_diff": 0.0, "motion_ratio": 0.0}

        diff = cv2.absdiff(self.prev_gray, blurred)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_ratio = float(cv2.countNonZero(thresh)) / float(frame.shape[0] * frame.shape[1])
        mean_diff = float(diff.mean())
        self.prev_gray = blurred

        event = None
        now_monotonic = time.monotonic()
        if (
            mean_diff >= self.threshold
            and motion_ratio >= self.min_area_ratio
            and (now_monotonic - self.last_trigger_at) >= self.cooldown_sec
        ):
            self.last_trigger_at = now_monotonic
            event = {
                "behavior_type": "motion",
                "type_label": "Movement Spike",
                "summary": "Large movement change detected in the monitored scene.",
                "metrics": {
                    "mean_diff": round(mean_diff, 2),
                    "motion_ratio": round(motion_ratio, 4),
                },
            }
        return event, {"mean_diff": mean_diff, "motion_ratio": motion_ratio}


@dataclass(slots=True)
class PendingEvidence:
    incident_id: str
    session_id: str
    node_id: str
    camera_label: str
    type_label: str
    behavior_type: str
    summary: str
    created_at: str
    display_time: str
    frames: list = field(default_factory=list)
    remaining_post_frames: int = 0


class EvidenceBuilder:
    """Collects frames around an event and writes them to local evidence storage."""

    def __init__(self, config) -> None:
        self.config = config
        self.pre_frames = deque(maxlen=config.pre_event_frames)

    def remember_frame(self, frame) -> None:
        self.pre_frames.append(frame.copy())

    def start_sequence(self, session_id: str, node_id: str, camera_label: str, event: dict, frame) -> PendingEvidence:
        sequence = PendingEvidence(
            incident_id=make_id(f"{node_id}-incident"),
            session_id=session_id,
            node_id=node_id,
            camera_label=camera_label,
            type_label=event["type_label"],
            behavior_type=event["behavior_type"],
            summary=event["summary"],
            created_at=utc_now_iso(),
            display_time=display_clock_now(),
            remaining_post_frames=self.config.post_event_frames,
        )
        sequence.frames.extend(list(self.pre_frames))
        sequence.frames.append(frame.copy())
        return sequence

    def advance_sequence(self, sequence: PendingEvidence, frame) -> bool:
        sequence.frames.append(frame.copy())
        sequence.remaining_post_frames -= 1
        return sequence.remaining_post_frames <= 0

    def finalize_sequence(self, sequence: PendingEvidence) -> tuple[IncidentManifest, list[dict]]:
        incident_dir = self.config.evidence_root / sequence.session_id / sequence.incident_id
        frames_dir = incident_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        asset_records = []
        frame_names = []
        for index, frame in enumerate(sequence.frames):
            name = f"frame_{index:03d}.jpg"
            path = frames_dir / name
            cv2.imwrite(str(path), frame)
            frame_names.append(name)
            asset_records.append({"asset_type": "frame", "file_path": path, "filename": name})

        poster_name = "poster.jpg"
        poster_path = incident_dir / poster_name
        poster_source = sequence.frames[min(len(sequence.frames) - 1, max(0, self.config.pre_event_frames))]
        cv2.imwrite(str(poster_path), poster_source)
        asset_records.append({"asset_type": "poster", "file_path": poster_path, "filename": poster_name})

        gif_name = ""
        if PIL_AVAILABLE and sequence.frames:
            gif_name = "evidence.gif"
            gif_path = incident_dir / gif_name
            pil_frames = []
            for frame in sequence.frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frames.append(Image.fromarray(rgb))
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=180,
                loop=0,
            )
            asset_records.append({"asset_type": "gif", "file_path": gif_path, "filename": gif_name})

        manifest = IncidentManifest(
            incident_id=sequence.incident_id,
            session_id=sequence.session_id,
            node_id=sequence.node_id,
            camera_label=sequence.camera_label,
            behavior_type=sequence.behavior_type,
            type_label=sequence.type_label,
            student_numbers=[],
            created_at=sequence.created_at,
            display_time=sequence.display_time,
            review_status="unverified",
            poster_path="",
            gif_path="",
            frame_count=len(sequence.frames),
            summary=sequence.summary,
            sync_status="queued",
            sync_attempts=0,
            asset_names=[poster_name] + ([gif_name] if gif_name else []) + frame_names,
        )
        manifest_path = incident_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2),
            encoding="utf-8",
        )
        return manifest, [
            {"asset_type": "manifest", "file_path": manifest_path, "filename": "manifest.json"},
            *asset_records,
        ]


def annotate_frame(frame, *, node_name: str, camera_label: str, session_id: str, fps: float, metrics: dict, banner_text: str = ""):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    cv2.rectangle(annotated, (0, 0), (width, 86), (18, 24, 29), -1)
    cv2.putText(annotated, node_name, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 245, 247), 2, cv2.LINE_AA)
    cv2.putText(annotated, camera_label, (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (197, 92, 46), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"FPS {fps:.1f}", (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (197, 217, 208), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Session {session_id or '--'}", (width - 260, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 245, 247), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Diff {metrics.get('mean_diff', 0.0):.1f}", (width - 260, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 245, 247), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Motion {metrics.get('motion_ratio', 0.0):.4f}", (width - 260, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 245, 247), 1, cv2.LINE_AA)
    if banner_text:
        cv2.rectangle(annotated, (0, height - 60), (width, height), (197, 92, 46), -1)
        cv2.putText(annotated, banner_text, (18, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated
