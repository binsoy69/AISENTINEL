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
FRONT_RUNTIME_ROOT = RUNTIME_ROOT / "front_node_pi"

for path in (REPO_ROOT, RUNTIME_ROOT, FRONT_RUNTIME_ROOT):
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
import front_node_all_behavior_pi as combined_runtime


class LauncherAndPreviewTests(unittest.TestCase):
    def _configured_default_video_name(self, runtime_config_name: str) -> str:
        parser = configparser.ConfigParser()
        parser.read(central_dashboard_config(runtime_config_name), encoding="utf-8")
        return Path(parser.get("video_source", "default_video")).name

    def test_video_launchers_resolve_configured_default_videos(self):
        front_video = validate_node_video_config(
            central_dashboard_config("node_front_video.ini")
        )
        mid_video = validate_node_video_config(
            central_dashboard_config("node_mid_video.ini")
        )

        self.assertEqual(front_video.name, self._configured_default_video_name("node_front_runtime.ini"))
        self.assertEqual(mid_video.name, self._configured_default_video_name("node_mid_runtime.ini"))
        self.assertTrue(front_video.exists())
        self.assertTrue(mid_video.exists())

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
            runtime_config = tmpdir / "runtime.ini"
            node_config = tmpdir / "node.ini"
            runtime_config.write_text(
                f"""
[video_source]
default_video = {old_video}
default_setup_profile = runtime/central_dashboard/data/node_front/setup_profiles/old.json
""".strip(),
                encoding="utf-8",
            )
            node_config.write_text(
                f"""
[capture]
source_mode = video
video_path = {old_video}

[detector]
runtime_config_path = {runtime_config}
""".strip(),
                encoding="utf-8",
            )

            save_node_video_default(node_config, new_video)

            parser = configparser.ConfigParser()
            parser.read(runtime_config, encoding="utf-8")
            node_parser = configparser.ConfigParser()
            node_parser.read(node_config, encoding="utf-8")

        self.assertEqual(Path(parser.get("video_source", "default_video")), new_video)
        self.assertEqual(parser.get("video_source", "default_setup_profile"), "")
        self.assertEqual(node_parser.get("capture", "video_path"), "")

    def test_save_node_video_default_keeps_profile_for_same_video(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            video = tmpdir / "same.mp4"
            video.write_bytes(b"same")
            runtime_config = tmpdir / "runtime.ini"
            node_config = tmpdir / "node.ini"
            runtime_config.write_text(
                f"""
[video_source]
default_video = {video}
default_setup_profile = runtime/central_dashboard/data/node_front/setup_profiles/same.json
""".strip(),
                encoding="utf-8",
            )
            node_config.write_text(
                f"""
[capture]
source_mode = video
video_path =

[detector]
runtime_config_path = {runtime_config}
""".strip(),
                encoding="utf-8",
            )

            save_node_video_default(node_config, video)

            parser = configparser.ConfigParser()
            parser.read(runtime_config, encoding="utf-8")

        self.assertEqual(
            parser.get("video_source", "default_setup_profile"),
            "runtime/central_dashboard/data/node_front/setup_profiles/same.json",
        )

    def test_run_node_video_calibration_passes_selected_video_to_script(self):
        selected_config = Path("runtime/central_dashboard/node_front_video.ini")
        selected_video = Path("test-videos/selected.mp4")

        with (
            mock.patch.object(
                launcher_common,
                "select_node_video_file",
                return_value=(selected_config, selected_video),
            ),
            mock.patch.object(launcher_common, "run_script") as run_script_mock,
        ):
            launcher_common.run_node_video_calibration("node_front_video.ini")

        args = run_script_mock.call_args.args
        self.assertEqual(args[1:], ("--config", str(selected_config), "--video", str(selected_video)))

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


if __name__ == "__main__":
    unittest.main()
