from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import sys

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.node_agent.config import NodeAgentConfig
from central_dashboard.node_agent.front_runtime import (
    _normalize_front_runtime_incident,
    _resolve_calibration_path,
    front_runtime_config,
)
from central_dashboard.node_agent.state import NodeRuntime
from central_dashboard.shared.dto import IncidentManifest


ROOT = Path(__file__).resolve().parents[1]


class FrontRuntimeBackendTests(unittest.TestCase):
    def test_front_runtime_runner_publishes_frames_and_queues_sync_items(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config = NodeAgentConfig(
                config_path=tmpdir / "node.ini",
                node_id="front",
                display_name="Front Node",
                camera_label="Front Camera",
                profile="front",
                host="front.test",
                port=8091,
                api_key="front-key",
                central_base_url="http://central.test:8090",
                registration_interval_sec=10.0,
                heartbeat_interval_sec=10.0,
                sync_interval_sec=10.0,
                http_timeout_sec=5.0,
                local_db_path=tmpdir / "queue.sqlite3",
                source_mode="webcam",
                camera_index=0,
                video_path=None,
                preview_width=640,
                preview_fps=15.0,
                jpeg_quality=75,
                detector_mode="front_runtime",
                runtime_config_path=tmpdir / "runtime.ini",
                motion_threshold=5.0,
                motion_min_area_ratio=0.001,
                motion_cooldown_sec=0.05,
                annotated_banner_ttl_sec=1.0,
                evidence_root=tmpdir / "evidence",
                pre_event_frames=2,
                post_event_frames=2,
            )

            def fake_runner(runtime: NodeRuntime, session) -> None:
                raw = np.zeros((72, 128, 3), dtype=np.uint8)
                annotated = np.full((72, 128, 3), 200, dtype=np.uint8)
                poster_path = tmpdir / "evidence" / "frames" / "poster.jpg"
                poster_path.parent.mkdir(parents=True, exist_ok=True)
                poster_path.write_bytes(b"jpeg-data")

                runtime.mark_session_running()
                runtime.publish_detector_frames(
                    raw,
                    annotated,
                    processing_fps=14.5,
                )
                runtime.record_finalized_incident(
                    IncidentManifest(
                        incident_id="incident-001",
                        session_id=session.session_id,
                        node_id=config.node_id,
                        camera_label=config.camera_label,
                        behavior_type="head",
                        type_label="Head Tilting",
                        student_numbers=[5],
                        created_at="2026-04-08T10:00:00Z",
                        display_time="10:00 AM",
                        frame_count=1,
                        summary="Student #05 head tilting detected",
                        sync_status="queued",
                        sync_attempts=0,
                        asset_names=["frames/poster.jpg"],
                    ),
                    [
                        {
                            "asset_type": "poster",
                            "file_path": poster_path,
                            "filename": "frames/poster.jpg",
                        }
                    ],
                )

            runtime = NodeRuntime(config, front_runtime_runner=fake_runner)
            ack = runtime.start_session(
                {
                    "subject_code": "CS321",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-08",
                    "start_time": "09:00",
                    "end_time": "11:00",
                }
            )
            self.assertTrue(ack.ok)
            runtime._session_thread.join(timeout=2.0)  # type: ignore[union-attr]

            heartbeat = runtime.heartbeat()
            self.assertEqual(heartbeat.extra["detector_mode"], "front_runtime")
            self.assertGreaterEqual(heartbeat.incident_count, 1)

            due_items = runtime.sync_queue.due_items(limit=10)
            self.assertEqual({item.item_type for item in due_items}, {"manifest", "asset"})
            manifest_item = next(item for item in due_items if item.item_type == "manifest")
            asset_item = next(item for item in due_items if item.item_type == "asset")
            self.assertEqual(
                manifest_item.payload["manifest_payload"]["type_label"],
                "Head Tilting",
            )
            self.assertEqual(asset_item.payload["filename"], "frames/poster.jpg")

            chunk = next(runtime.stream_generator("annotated"))
            self.assertIn(b"Content-Type: image/jpeg", chunk)
            runtime.close()

    def test_normalize_front_runtime_incident_uses_incident_relative_asset_names(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            evidence_root = tmpdir / "evidence"
            frame_a = evidence_root / "head_behavior" / "events" / "incident-001" / "frames" / "f00_pre.jpg"
            frame_b = evidence_root / "head_behavior" / "events" / "incident-001" / "frames" / "f01_event.jpg"
            gif_path = evidence_root / "head_behavior" / "events" / "incident-001" / "evidence.gif"
            for path in (frame_a, frame_b, gif_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

            incident, assets = _normalize_front_runtime_incident(
                node_config=SimpleNamespace(node_id="front", camera_label="Front Camera"),
                session=SimpleNamespace(session_id="session-001"),
                evidence_root=evidence_root,
                front_manifest={
                    "id": "incident-001",
                    "behavior_type": "head",
                    "type_label": "Head Tilting",
                    "student_numbers": [5],
                    "created_at": "2026-04-08T10:00:00Z",
                    "display_time": "10:00 AM",
                    "summary": "Student #05 head tilting detected",
                    "frame_count": 2,
                    "manifest_relpath": "head_behavior/events/incident-001/manifest.json",
                    "poster_relpath": "head_behavior/events/incident-001/frames/f01_event.jpg",
                    "gif_relpath": "head_behavior/events/incident-001/evidence.gif",
                    "frame_relpaths": [
                        "head_behavior/events/incident-001/frames/f00_pre.jpg",
                        "head_behavior/events/incident-001/frames/f01_event.jpg",
                    ],
                },
            )

            self.assertEqual(incident.incident_id, "incident-001")
            self.assertEqual(incident.student_numbers, [5])
            self.assertEqual(
                [asset["filename"] for asset in assets],
                ["frames/f01_event.jpg", "evidence.gif", "frames/f00_pre.jpg"],
            )
            self.assertEqual(
                [asset["asset_type"] for asset in assets],
                ["poster", "gif", "frame"],
            )

    def test_resolve_calibration_path_prefers_saved_default_and_auto_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            runtime_cfg = front_runtime_config.load_runtime_config(
                str(ROOT / "node_front_runtime.ini")
            )
            default_profile = tmpdir / "saved_setup.json"
            auto_profile = tmpdir / "auto_setup.json"
            default_profile.write_text("{}", encoding="utf-8")
            auto_profile.write_text("{}", encoding="utf-8")

            runtime_cfg = replace(
                runtime_cfg,
                webcam_source=replace(
                    runtime_cfg.webcam_source,
                    default_setup_profile=default_profile,
                    auto_use_saved_setup=True,
                ),
            )

            setup_io = SimpleNamespace(
                default_setup_profile_path=lambda _label: auto_profile
            )
            head_mod = SimpleNamespace(log_info=lambda _message: None)
            path = _resolve_calibration_path(
                SimpleNamespace(source_mode="webcam"),
                runtime_cfg,
                setup_io,
                "front_webcam",
                head_mod,
            )
            self.assertEqual(path, default_profile)

            runtime_cfg = replace(
                runtime_cfg,
                webcam_source=replace(
                    runtime_cfg.webcam_source,
                    default_setup_profile=tmpdir / "missing.json",
                    auto_use_saved_setup=True,
                ),
            )
            path = _resolve_calibration_path(
                SimpleNamespace(source_mode="webcam"),
                runtime_cfg,
                setup_io,
                "front_webcam",
                head_mod,
            )
            self.assertEqual(path, auto_profile)


if __name__ == "__main__":
    unittest.main()
