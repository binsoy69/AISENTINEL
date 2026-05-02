from __future__ import annotations

import configparser
import builtins
import os
from pathlib import Path
import tempfile
import unittest
import sys
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPO_ROOT / "runtime"

for path in (REPO_ROOT, RUNTIME_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from central_dashboard.central_service.config import load_central_service_config
from programs import _launcher_common as launcher_common
from programs._launcher_common import (
    central_dashboard_config,
    run_script,
    save_node_video_default,
    select_video_file,
    validate_node_video_config,
)
from edge_node_runtime import front_node_all_behavior_pi as combined_runtime


class LauncherAndPreviewTests(unittest.TestCase):
    def test_video_launchers_resolve_configured_default_videos(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            video_path = tmpdir / "classroom.mp4"
            video_path.write_bytes(b"video")
            node_config = tmpdir / "front_node.ini"
            node_config.write_text(
                f"""
[capture]
source_mode = webcam
video_path =

[video_source]
default_video = {video_path}
""".strip(),
                encoding="utf-8",
            )

            configured_video = validate_node_video_config(node_config)

        self.assertEqual(configured_video, video_path.resolve(strict=False))

    def test_select_video_file_returns_existing_picker_path(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            initial_video = tmpdir / "initial.mp4"
            selected_video = tmpdir / "selected.mp4"
            initial_video.write_bytes(b"initial")
            selected_video.write_bytes(b"selected")
            picker_calls = []

            def picker(**kwargs):
                picker_calls.append(kwargs)
                return str(selected_video)

            result = select_video_file(
                title="Pick a video",
                initial_path=initial_video,
                picker=picker,
            )

        self.assertEqual(result, selected_video.resolve(strict=False))
        self.assertEqual(picker_calls[0]["title"], "Pick a video")
        self.assertEqual(Path(picker_calls[0]["initialdir"]), initial_video.parent)

    def test_select_video_file_cancel_exits_cleanly(self):
        with self.assertRaises(SystemExit) as raised:
            select_video_file(title="Pick a video", picker=lambda **_: "")

        self.assertIn("cancelled", str(raised.exception).lower())

    def test_select_video_file_missing_gui_exits_with_readable_error(self):
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tkinter":
                raise ImportError("tkinter unavailable")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(SystemExit) as raised:
                select_video_file(title="Pick a video")

        message = str(raised.exception)
        self.assertIn("Video file picker is unavailable", message)
        self.assertIn("tkinter unavailable", message)

    def test_save_node_video_default_updates_runtime_and_clears_stale_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            old_video = tmpdir / "old.mp4"
            new_video = tmpdir / "new.mp4"
            old_video.write_bytes(b"old")
            new_video.write_bytes(b"new")
            node_config = tmpdir / "node.ini"
            node_config.write_text(
                f"""
[capture]
source_mode = webcam
video_path = {old_video}

[video_source]
default_video = {old_video}
default_setup_profile = runtime/central_dashboard/data/node_front/setup_profiles/old.json
""".strip(),
                encoding="utf-8",
            )

            save_node_video_default(node_config, new_video)

            parser = configparser.ConfigParser()
            parser.read(node_config, encoding="utf-8")

        self.assertEqual(Path(parser.get("video_source", "default_video")), new_video)
        self.assertEqual(parser.get("video_source", "default_setup_profile"), "")
        self.assertEqual(parser.get("capture", "video_path"), "")

    def test_save_node_video_default_keeps_profile_for_same_video(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            video = tmpdir / "same.mp4"
            video.write_bytes(b"same")
            node_config = tmpdir / "node.ini"
            node_config.write_text(
                f"""
[capture]
source_mode = webcam
video_path =

[video_source]
default_video = {video}
default_setup_profile = runtime/central_dashboard/data/node_front/setup_profiles/same.json
""".strip(),
                encoding="utf-8",
            )

            save_node_video_default(node_config, video)

            parser = configparser.ConfigParser()
            parser.read(node_config, encoding="utf-8")

        self.assertEqual(
            parser.get("video_source", "default_setup_profile"),
            "runtime/central_dashboard/data/node_front/setup_profiles/same.json",
        )

    def test_run_node_video_calibration_passes_selected_video_to_script(self):
        selected_config = Path("config/front_node.ini")
        selected_video = Path("test-videos/selected.mp4")

        with (
            mock.patch.object(
                launcher_common,
                "select_node_video_file",
                return_value=(selected_config, selected_video),
            ),
            mock.patch.object(launcher_common, "run_script") as run_script_mock,
        ):
            launcher_common.run_node_video_calibration("front_node.ini")

        args = run_script_mock.call_args.args
        self.assertEqual(args[1:], ("--config", str(selected_config), "--video", str(selected_video)))

    def test_run_node_sound_calibration_passes_node_config_to_script(self):
        selected_config = Path("config/front_node.ini")

        with (
            mock.patch.object(
                launcher_common,
                "require_operator_config",
                return_value=selected_config,
            ),
            mock.patch.object(launcher_common, "run_script") as run_script_mock,
        ):
            launcher_common.run_node_sound_calibration("front_node.ini")

        args = run_script_mock.call_args.args
        self.assertEqual(args[1:], ("--config", str(selected_config)))

    def test_run_sound_sensor_test_passes_configured_calibration_file(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            calibration_path = tmpdir / "ky037.json"
            calibration_path.write_text("{}", encoding="utf-8")
            node_config = tmpdir / "front_node.ini"
            node_config.write_text(
                f"""
[sound_sensor]
calibration_config = {calibration_path}
""".strip(),
                encoding="utf-8",
            )

            with (
                mock.patch.object(
                    launcher_common,
                    "require_operator_config",
                    return_value=node_config,
                ),
                mock.patch.object(launcher_common, "run_script") as run_script_mock,
            ):
                launcher_common.run_sound_sensor_test("front_node.ini")

        args = run_script_mock.call_args.args
        self.assertEqual(args[1:], ("--config-file", str(calibration_path.resolve(strict=False))))

    def test_run_script_includes_launched_script_directory_for_sibling_imports(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            helper_path = tmpdir / "sibling_helper.py"
            helper_path.write_text("MESSAGE = 'sibling import worked'\n", encoding="utf-8")

            output_path = tmpdir / "output.txt"
            script_path = tmpdir / "script_with_sibling_import.py"
            script_path.write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "import sys",
                        "from sibling_helper import MESSAGE",
                        "Path(sys.argv[1]).write_text(MESSAGE, encoding='utf-8')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            run_script(script_path, str(output_path))

            self.assertEqual(output_path.read_text(encoding="utf-8"), "sibling import worked")
            self.assertNotIn(str(tmpdir.resolve()), sys.path)

    def test_external_central_config_base_resolves_relative_data_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config_path = tmpdir / "central_service.ini"
            config_path.write_text(
                """
[service]
host = 127.0.0.1
port = 8090
db_path = data/central.sqlite3
evidence_root = data/evidence

[browser_auth]
username = admin
password = admin123
secret_key = test-secret

[node:front]
display_name = Front Node
camera_label = Front Camera
api_key = front-key
""".strip(),
                encoding="utf-8",
            )

            old_value = os.environ.get("AISENTINEL_CONFIG_BASE")
            os.environ["AISENTINEL_CONFIG_BASE"] = str(tmpdir)
            try:
                config = load_central_service_config(config_path)
            finally:
                if old_value is None:
                    os.environ.pop("AISENTINEL_CONFIG_BASE", None)
                else:
                    os.environ["AISENTINEL_CONFIG_BASE"] = old_value

        self.assertEqual(config.db_path, (tmpdir / "data" / "central.sqlite3").resolve(strict=False))
        self.assertEqual(config.evidence_root, (tmpdir / "data" / "evidence").resolve(strict=False))

    def test_live_preview_draws_only_confirmed_incident_targets(self):
        raw_frame = np.zeros((120, 160, 3), dtype=np.uint8)
        snapshot = {
            "raw_frame": raw_frame,
            "student_boxes": {
                1: (10, 10, 50, 60),
                2: (70, 10, 110, 60),
            },
            "object_boxes": {
                (1, "phone"): [(20, 22, 34, 42)],
                (2, "cheat_sheet"): [(80, 22, 96, 42)],
            },
        }

        preview = combined_runtime.build_live_preview_frame(
            snapshot,
            frame_object_alerts=[
                {"student_num": 1, "class_name": "phone", "confidence": 0.9}
            ],
        )

        self.assertTrue(np.any(preview[10, 10] != raw_frame[10, 10]))
        self.assertTrue(np.any(preview[22, 20] != raw_frame[22, 20]))
        self.assertTrue(np.array_equal(preview[10, 70], raw_frame[10, 70]))
        self.assertTrue(np.array_equal(preview[22, 80], raw_frame[22, 80]))

    def test_debug_preview_uses_diagnostic_overlay_without_mutating_clean_preview(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        points = {
            combined_runtime.head_mod.KP_NOSE: (30, 18),
            combined_runtime.head_mod.KP_LEFT_EAR: (22, 20),
            combined_runtime.head_mod.KP_RIGHT_EAR: (38, 20),
            combined_runtime.head_mod.KP_LEFT_SHOULDER: (18, 42),
            combined_runtime.head_mod.KP_RIGHT_SHOULDER: (52, 42),
            combined_runtime.pass_mod.KP_LEFT_WRIST: (20, 76),
            combined_runtime.pass_mod.KP_RIGHT_WRIST: (50, 76),
        }
        for idx, (x, y) in points.items():
            keypoints[idx] = (x, y, 0.95)

        class OneFrameCapture:
            def __init__(self, source_frame):
                self.source_frame = source_frame
                self.read_count = 0

            def read(self):
                if self.read_count:
                    return False, None
                self.read_count += 1
                return True, self.source_frame.copy()

            def get(self, prop):
                if prop == combined_runtime.cv2.CAP_PROP_FRAME_WIDTH:
                    return self.source_frame.shape[1]
                if prop == combined_runtime.cv2.CAP_PROP_FRAME_HEIGHT:
                    return self.source_frame.shape[0]
                if prop == combined_runtime.cv2.CAP_PROP_FPS:
                    return 30.0
                if prop == combined_runtime.cv2.CAP_PROP_FRAME_COUNT:
                    return 1
                return 0

            def set(self, _prop, _value):
                return None

        class PoseEstimator:
            def detect_pose(self, _frame):
                return [{"bbox": (12, 12, 58, 86), "confidence": 0.9, "keypoints": keypoints.copy()}]

        class EmptyDetector:
            def detect(self, _frame):
                return []

        class Tracker:
            def update(self, detections):
                return [1 for _ in detections]

        published = {}

        def publish_callback(raw_frame, annotated_frame, metrics, *, debug_frame=None):
            published["raw"] = raw_frame.copy()
            published["annotated"] = annotated_frame.copy()
            published["debug"] = debug_frame.copy()
            published["metrics"] = dict(metrics)

        combined_runtime.run_detection(
            OneFrameCapture(frame),
            PoseEstimator(),
            EmptyDetector(),
            EmptyDetector(),
            Tracker(),
            {1: 5},
            {1: 0.0},
            [{"track_id": 1, "student_num": 5, "bbox": (12, 12, 58, 86)}],
            [None],
            "unit-test.mp4",
            0,
            source_mode="video",
            source_fps=30.0,
            frame_publish_callback=publish_callback,
            publish_local_preview=False,
        )

        self.assertTrue(np.array_equal(published["annotated"], frame))
        self.assertTrue(np.any(published["debug"] != frame))
        self.assertIn("processing_fps", published["metrics"])

    def test_evidence_sequence_reports_recording_incident_immediately(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            old_head_dir = combined_runtime.HEAD_EVIDENCE_DIR
            combined_runtime.HEAD_EVIDENCE_DIR = Path(tmpdir_str) / "head_behavior"
            try:
                detected = []
                sequence_queue = []

                sequence = combined_runtime.queue_evidence_sequence(
                    None,
                    sequence_queue,
                    [],
                    "head",
                    1.25,
                    incident_detected_callback=lambda incident: detected.append(dict(incident)),
                    student_num=5,
                    behavior="head_tilt",
                )

                self.assertEqual(len(detected), 1)
                self.assertEqual(detected[0]["id"], sequence["incident_id"])
                self.assertEqual(detected[0]["status"], "recording")
                self.assertEqual(detected[0]["student_numbers"], [5])
                self.assertEqual(detected[0]["poster_relpath"], "")
                self.assertEqual(detected[0]["gif_relpath"], "")
            finally:
                combined_runtime.HEAD_EVIDENCE_DIR = old_head_dir

    def test_evidence_sequence_uses_configured_pre_post_and_gif_cap(self):
        def snapshot(name, value):
            return {
                "name": name,
                "raw_frame": np.full((8, 8, 3), value, dtype=np.uint8),
            }

        class TaskSink:
            def __init__(self):
                self.tasks = []

            def put(self, task):
                self.tasks.append(task)

        old_pre = combined_runtime.EVIDENCE_PRE_EVENT_FRAMES
        old_post = combined_runtime.EVIDENCE_POST_EVENT_FRAMES
        old_gif = combined_runtime.EVIDENCE_GIF_FRAME_COUNT
        old_head_dir = combined_runtime.HEAD_EVIDENCE_DIR
        try:
            combined_runtime.EVIDENCE_PRE_EVENT_FRAMES = 3
            combined_runtime.EVIDENCE_POST_EVENT_FRAMES = 1
            combined_runtime.EVIDENCE_GIF_FRAME_COUNT = 4
            with tempfile.TemporaryDirectory() as tmpdir_str:
                combined_runtime.HEAD_EVIDENCE_DIR = Path(tmpdir_str) / "head_behavior"
                recent_frames = [
                    snapshot(f"recent-{idx}", idx)
                    for idx in range(5)
                ]
                sequence_queue = []
                task_sink = TaskSink()

                sequence = combined_runtime.queue_evidence_sequence(
                    task_sink,
                    sequence_queue,
                    recent_frames,
                    "head",
                    1.25,
                    student_num=5,
                    behavior="head_tilt",
                )

                self.assertEqual(
                    [item["name"] for item in sequence["pre_event_snapshots"]],
                    ["recent-2", "recent-3", "recent-4"],
                )
                remaining = combined_runtime.flush_evidence_sequences(
                    task_sink,
                    sequence_queue,
                    snapshot("event", 200),
                )
                self.assertEqual(remaining, [sequence])
                self.assertEqual(task_sink.tasks, [])

                remaining = combined_runtime.flush_evidence_sequences(
                    task_sink,
                    sequence_queue,
                    snapshot("post", 240),
                )

                self.assertEqual(remaining, [])
                self.assertEqual(task_sink.tasks[0]["type"], "finalize")
                snapshots, event_index = combined_runtime._sequence_gif_snapshots(sequence)
                self.assertEqual(
                    [item["name"] for item in snapshots],
                    ["recent-3", "recent-4", "event", "post"],
                )
                self.assertEqual(event_index, 2)
        finally:
            combined_runtime.EVIDENCE_PRE_EVENT_FRAMES = old_pre
            combined_runtime.EVIDENCE_POST_EVENT_FRAMES = old_post
            combined_runtime.EVIDENCE_GIF_FRAME_COUNT = old_gif
            combined_runtime.HEAD_EVIDENCE_DIR = old_head_dir

    def test_single_frame_gif_count_keeps_event_snapshot_as_poster(self):
        old_pre = combined_runtime.EVIDENCE_PRE_EVENT_FRAMES
        old_post = combined_runtime.EVIDENCE_POST_EVENT_FRAMES
        old_gif = combined_runtime.EVIDENCE_GIF_FRAME_COUNT
        old_evidence_dir = combined_runtime.EVIDENCE_DIR
        try:
            combined_runtime.EVIDENCE_PRE_EVENT_FRAMES = 3
            combined_runtime.EVIDENCE_POST_EVENT_FRAMES = 2
            combined_runtime.EVIDENCE_GIF_FRAME_COUNT = 1
            with tempfile.TemporaryDirectory() as tmpdir_str:
                evidence_dir = Path(tmpdir_str)
                combined_runtime.EVIDENCE_DIR = evidence_dir
                sequence = {
                    "behavior_type": "head",
                    "student_num": 5,
                    "event_dir": evidence_dir,
                    "frame_paths": [],
                    "frame_count": 0,
                    "poster_relpath": "",
                    "gif_relpath": "",
                    "pre_event_snapshots": [
                        {"raw_frame": np.full((8, 8, 3), 10, dtype=np.uint8)},
                        {"raw_frame": np.full((8, 8, 3), 20, dtype=np.uint8)},
                        {"raw_frame": np.full((8, 8, 3), 30, dtype=np.uint8)},
                    ],
                    "event_snapshot": {
                        "raw_frame": np.full((8, 8, 3), 200, dtype=np.uint8)
                    },
                    "post_event_snapshots": [
                        {"raw_frame": np.full((8, 8, 3), 40, dtype=np.uint8)},
                        {"raw_frame": np.full((8, 8, 3), 50, dtype=np.uint8)},
                    ],
                }
                poster_frames = []

                def capture_imwrite(_path, frame):
                    poster_frames.append(frame.copy())
                    return True

                with mock.patch.object(combined_runtime.cv2, "imwrite", side_effect=capture_imwrite), \
                     mock.patch.object(combined_runtime, "_save_evidence_gif", return_value=True):
                    combined_runtime._save_grouped_evidence_media(sequence)

                self.assertEqual(sequence["frame_count"], 1)
                self.assertEqual(int(poster_frames[0][0, 0, 0]), 200)
        finally:
            combined_runtime.EVIDENCE_PRE_EVENT_FRAMES = old_pre
            combined_runtime.EVIDENCE_POST_EVENT_FRAMES = old_post
            combined_runtime.EVIDENCE_GIF_FRAME_COUNT = old_gif
            combined_runtime.EVIDENCE_DIR = old_evidence_dir


if __name__ == "__main__":
    unittest.main()
