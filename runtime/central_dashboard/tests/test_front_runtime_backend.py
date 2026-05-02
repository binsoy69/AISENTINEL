from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import tempfile
import unittest
import sys

import cv2
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
from central_dashboard.shared.http import HttpResult
from edge_node_runtime import front_node_all_behavior_pi as combined_runtime
from edge_node_runtime import runtime_support


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT.parents[1] / "config"


class FakeHttpClient:
    def __init__(self):
        self.requests = []

    def post_json(self, url: str, payload: dict, *, headers=None, timeout=5.0):
        self.requests.append((url, payload, timeout))
        return HttpResult(200, {"ok": True}, '{"ok": true}')

    def post_file(
        self,
        url: str,
        fields: dict,
        *,
        file_field: str,
        file_path,
        filename: str,
        headers=None,
        timeout=5.0,
    ):
        self.requests.append((url, {**fields, "filename": filename}, timeout))
        return HttpResult(200, {"ok": True}, '{"ok": true}')


class FrontRuntimeBackendTests(unittest.TestCase):
    def test_webcam_backend_attempt_builder_uses_current_platform(self):
        attempts = runtime_support._build_webcam_backend_attempts()

        self.assertTrue(attempts)
        self.assertTrue(any(backend_name == "default" for backend_name, *_ in attempts))

    def test_head_tilt_and_shoulder_turn_share_public_head_tilt_label(self):
        self.assertEqual(
            combined_runtime._sequence_type_label(
                {"behavior_type": "head", "behavior": "head_tilt"}
            ),
            "Head Tilt",
        )
        self.assertEqual(
            combined_runtime._sequence_type_label(
                {"behavior_type": "head", "behavior": "shoulder_turn"}
            ),
            "Head Tilt",
        )

    def test_incident_suppressor_blocks_repeated_head_signals_until_clear_and_window(self):
        suppressor = combined_runtime.IncidentSuppressor(
            duplicate_suppression_sec=60.0,
            clear_required_sec=3.0,
        )
        key = combined_runtime._head_suppression_key(5)

        suppressor.begin_frame()
        self.assertTrue(suppressor.allow(key, 10.0))
        suppressor.end_frame(10.0)

        suppressor.begin_frame()
        self.assertFalse(suppressor.allow(key, 12.0))
        suppressor.end_frame(12.0)

        suppressor.begin_frame()
        suppressor.end_frame(70.0)
        suppressor.begin_frame()
        self.assertTrue(suppressor.allow(key, 71.0))

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
                http_timeout_sec=5.0,
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

            runner_ready = threading.Event()

            def fake_runner(runtime: NodeRuntime, session) -> None:
                raw = np.zeros((72, 128, 3), dtype=np.uint8)
                annotated = np.full((72, 128, 3), 200, dtype=np.uint8)
                debug = np.full((72, 128, 3), 80, dtype=np.uint8)
                poster_path = tmpdir / "evidence" / "frames" / "poster.jpg"
                poster_path.parent.mkdir(parents=True, exist_ok=True)
                poster_path.write_bytes(b"jpeg-data")

                runtime.mark_session_running()
                runtime.publish_detector_frames(
                    raw,
                    annotated,
                    processing_fps=14.5,
                    debug_frame=debug,
                )
                runtime.record_finalized_incident(
                    IncidentManifest(
                        incident_id="incident-001",
                        session_id=session.session_id,
                        node_id=config.node_id,
                        camera_label=config.camera_label,
                        behavior_type="head",
                        type_label="Head Tilt",
                        student_numbers=[5],
                        created_at="2026-04-08T10:00:00Z",
                        display_time="10:00 AM",
                        frame_count=1,
                        summary="Student #05 head tilt detected",
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
                runner_ready.set()
                while not runtime.should_stop_requested():
                    time.sleep(0.01)

            http_client = FakeHttpClient()
            runtime = NodeRuntime(config, http_client=http_client, front_runtime_runner=fake_runner)
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
            self.assertTrue(runner_ready.wait(timeout=2.0))
            self.assertTrue(runtime.upload_worker.wait_until_idle(timeout=2.0))
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                heartbeat = runtime.heartbeat()
                if heartbeat.extra["stream"]["has_annotated_frame"] and heartbeat.extra["stream"]["has_debug_frame"]:
                    break
                time.sleep(0.01)

            heartbeat = runtime.heartbeat()
            self.assertEqual(heartbeat.extra["detector_mode"], "front_runtime")
            self.assertTrue(heartbeat.extra["stream"]["has_annotated_frame"])
            self.assertTrue(heartbeat.extra["stream"]["has_debug_frame"])
            self.assertGreater(heartbeat.extra["stream"]["debug_seq"], 0)
            self.assertTrue(heartbeat.extra["stream"]["last_frame_at"])
            self.assertGreaterEqual(heartbeat.incident_count, 1)

            self.assertEqual(len(http_client.requests), 2)
            manifest_item = http_client.requests[0][1]
            asset_item = http_client.requests[1][1]
            self.assertEqual(
                manifest_item["type_label"],
                "Head Tilt",
            )
            self.assertEqual(asset_item["filename"], "frames/poster.jpg")

            chunk = next(runtime.stream_generator("annotated"))
            self.assertIn(b"Content-Type: image/jpeg", chunk)
            debug_chunk = next(runtime.stream_generator("debug"))
            self.assertIn(b"Content-Type: image/jpeg", debug_chunk)
            runtime.close()

    def test_node_runtime_heartbeat_exposes_sound_and_noise_incident_queues_snapshot(self):
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
                http_timeout_sec=5.0,
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

            runner_ready = threading.Event()

            def fake_runner(runtime: NodeRuntime, session) -> None:
                poster_path = tmpdir / "evidence" / session.session_id / "noise-001" / "poster.jpg"
                poster_path.parent.mkdir(parents=True, exist_ok=True)
                poster_path.write_bytes(b"noise-jpeg")
                runtime.mark_session_running()
                runtime.update_sound_telemetry(
                    {
                        "enabled": True,
                        "current_db": 61.4,
                        "threshold_db": 55.0,
                        "over_threshold": True,
                        "status": "alert",
                        "updated_at": "2026-04-22T12:00:00Z",
                        "last_error": "",
                    }
                )
                runtime.record_finalized_incident(
                    IncidentManifest(
                        incident_id="noise-001",
                        session_id=session.session_id,
                        node_id=config.node_id,
                        camera_label=config.camera_label,
                        behavior_type="noise",
                        type_label="Noise Threshold Exceeded",
                        student_numbers=[],
                        created_at="2026-04-22T12:00:00Z",
                        display_time="12:00 PM",
                        frame_count=1,
                        summary="Estimated noise 61.4 dB exceeded 55.0 dB threshold.",
                        sync_status="queued",
                        sync_attempts=0,
                        asset_names=["poster.jpg"],
                    ),
                    [
                        {
                            "asset_type": "poster",
                            "file_path": poster_path,
                            "filename": "poster.jpg",
                        }
                    ],
                )
                runner_ready.set()
                while not runtime.should_stop_requested():
                    time.sleep(0.01)

            http_client = FakeHttpClient()
            runtime = NodeRuntime(config, http_client=http_client, front_runtime_runner=fake_runner)
            ack = runtime.start_session(
                {
                    "subject_code": "CS321",
                    "professor": "Dr. Reyes",
                    "session_date": "2026-04-22",
                    "start_time": "09:00",
                    "end_time": "11:00",
                }
            )
            self.assertTrue(ack.ok)
            self.assertTrue(runner_ready.wait(timeout=2.0))
            self.assertTrue(runtime.upload_worker.wait_until_idle(timeout=2.0))

            heartbeat = runtime.heartbeat()
            self.assertTrue(heartbeat.extra["sound"]["enabled"])
            self.assertEqual(heartbeat.extra["sound"]["current_db"], 61.4)
            self.assertTrue(heartbeat.extra["sound"]["over_threshold"])

            self.assertEqual(len(http_client.requests), 2)
            manifest_item = http_client.requests[0][1]
            asset_item = http_client.requests[1][1]
            self.assertEqual(
                manifest_item["behavior_type"],
                "noise",
            )
            self.assertEqual(asset_item["asset_type"], "poster")
            self.assertEqual(asset_item["filename"], "poster.jpg")
            runtime.close()

    def test_normalize_front_runtime_incident_uses_stable_upload_asset_names(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            evidence_root = tmpdir / "evidence"
            frame_a = evidence_root / "head_behavior" / "events" / "incident-001" / "frames" / "f00_pre.jpg"
            frame_b = evidence_root / "head_behavior" / "events" / "incident-001" / "frames" / "f01_event.jpg"
            for path in (frame_a, frame_b):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

            incident, assets = _normalize_front_runtime_incident(
                node_config=SimpleNamespace(node_id="front", camera_label="Front Camera"),
                session=SimpleNamespace(session_id="session-001"),
                evidence_root=evidence_root,
                front_manifest={
                    "id": "incident-001",
                    "behavior_type": "head",
                    "type_label": "Head Tilt",
                    "student_numbers": [5],
                    "created_at": "2026-04-08T10:00:00Z",
                    "display_time": "10:00 AM",
                    "summary": "Student #05 head tilt detected",
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
                ["poster.jpg", "evidence.gif"],
            )
            self.assertEqual(
                [asset["asset_type"] for asset in assets],
                ["poster", "gif"],
            )

    def test_normalize_front_runtime_incident_uses_event_frame_as_snapshot_only(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            evidence_root = tmpdir / "evidence"
            frame_dir = evidence_root / "objects" / "events" / "incident-001" / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            pre_frame = frame_dir / "f10_pre01.jpg"
            event_frame = frame_dir / "f11_event.jpg"
            cv2.imwrite(str(pre_frame), np.zeros((32, 48, 3), dtype=np.uint8))
            cv2.imwrite(str(event_frame), np.full((32, 48, 3), 180, dtype=np.uint8))

            incident, assets = _normalize_front_runtime_incident(
                node_config=SimpleNamespace(node_id="front", camera_label="Front Camera"),
                session=SimpleNamespace(session_id="session-001"),
                evidence_root=evidence_root,
                front_manifest={
                    "id": "incident-001",
                    "behavior_type": "object",
                    "type_label": "Cheat Sheet",
                    "student_numbers": [8],
                    "created_at": "2026-04-08T10:00:00Z",
                    "display_time": "10:00 AM",
                    "summary": "Student #08 cheat sheet detected",
                    "frame_count": 1,
                    "manifest_relpath": "objects/events/incident-001/manifest.json",
                    "frame_relpaths": [
                        "objects/events/incident-001/frames/f10_pre01.jpg",
                        "objects/events/incident-001/frames/f11_event.jpg",
                    ],
                },
            )

            self.assertEqual(incident.asset_names, ["poster.jpg"])
            self.assertEqual(incident.frame_count, 1)
            self.assertEqual(len(assets), 1)
            self.assertEqual(assets[0]["asset_type"], "poster")
            self.assertEqual(assets[0]["filename"], "poster.jpg")
            self.assertEqual(assets[0]["file_path"], event_frame)

    def test_normalize_front_runtime_noise_incident_stays_snapshot_only(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            evidence_root = tmpdir / "evidence"
            poster_path = evidence_root / "noise" / "events" / "noise-001" / "poster.jpg"
            gif_path = evidence_root / "noise" / "events" / "noise-001" / "evidence.gif"
            poster_path.parent.mkdir(parents=True, exist_ok=True)
            poster_path.write_bytes(b"noise-jpeg")
            gif_path.write_bytes(b"should-not-sync")

            incident, assets = _normalize_front_runtime_incident(
                node_config=SimpleNamespace(node_id="front", camera_label="Front Camera"),
                session=SimpleNamespace(session_id="session-001"),
                evidence_root=evidence_root,
                front_manifest={
                    "id": "noise-001",
                    "behavior_type": "noise",
                    "type_label": "Noise Threshold Exceeded",
                    "created_at": "2026-04-08T10:00:00Z",
                    "display_time": "10:00 AM",
                    "summary": "Estimated noise 60.0 dB exceeded 55.0 dB threshold.",
                    "manifest_relpath": "noise/events/noise-001/manifest.json",
                    "poster_relpath": "noise/events/noise-001/poster.jpg",
                    "gif_relpath": "noise/events/noise-001/evidence.gif",
                },
            )

            self.assertEqual(incident.asset_names, ["poster.jpg"])
            self.assertEqual([asset["asset_type"] for asset in assets], ["poster"])

    def test_resolve_calibration_path_prefers_saved_default_and_auto_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            runtime_cfg = front_runtime_config.load_runtime_config(
                str(CONFIG_ROOT / "front_node.ini.example")
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
