from __future__ import annotations

import unittest
from pathlib import Path
import sys
import tempfile

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.central_service.config import load_central_service_config
from central_dashboard.node_agent.config import load_node_agent_config
from central_dashboard.node_agent.front_runtime import front_runtime_config


ROOT = Path(__file__).resolve().parents[1]
EDGE_NODE_ROOT = ROOT.parent / "edge_node_runtime"
CONFIG_ROOT = ROOT.parents[1] / "config"


class ConfigTests(unittest.TestCase):
    def test_load_central_service_config(self):
        config = load_central_service_config(CONFIG_ROOT / "central.ini.example")
        self.assertEqual(config.port, 8090)
        self.assertIn("front", config.known_nodes)
        self.assertIn("mid", config.known_nodes)
        self.assertEqual(config.known_nodes["front"].camera_label, "Front Camera")

    def test_load_node_agent_config(self):
        config = load_node_agent_config(CONFIG_ROOT / "front_node.ini.example")
        self.assertEqual(config.node_id, "front")
        self.assertEqual(config.port, 8091)
        self.assertEqual(config.source_mode, "webcam")
        self.assertEqual(config.detector_mode, "front_runtime")
        self.assertEqual(
            config.runtime_config_path,
            (CONFIG_ROOT / "front_node.ini.example").resolve(strict=False),
        )
        self.assertEqual(config.startup_detection_delay_sec, 5.0)
        self.assertIn("runtime", str(config.evidence_root))
        self.assertIn("central_dashboard", str(config.evidence_root))

    def test_load_front_runtime_sound_config(self):
        config = front_runtime_config.load_runtime_config(
            str(CONFIG_ROOT / "front_node.ini.example")
        )
        self.assertFalse(config.sound_sensor.enabled)
        self.assertEqual(config.sound_sensor.i2c_address, 0x48)
        self.assertEqual(config.sound_sensor.alert_threshold_db, 55.0)
        self.assertEqual(
            config.sound_sensor.calibration_config,
            (
                ROOT.parents[1]
                / "runtime/central_dashboard/data/node_front/sound/ky037_ads1015_config.json"
            ).resolve(strict=False),
        )

    def test_dashboard_js_renders_evidence_processing_snapshot_state(self):
        script = (ROOT / "central_service" / "static" / "dashboard.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("function evidenceCellMarkup", script)
        self.assertIn("Evidence processing", script)
        self.assertIn("View Snapshot", script)
        self.assertIn("View GIF", script)
        self.assertLess(
            script.index("if (incident?.gif_url)"),
            script.index("Evidence processing"),
        )

    def test_front_and_mid_runtime_configs_load_spam_suppression(self):
        for filename in ("front_node.ini.example", "mid_node.ini.example"):
            config = front_runtime_config.load_runtime_config(str(CONFIG_ROOT / filename))
            self.assertEqual(config.spam_suppression.duplicate_suppression_sec, 60.0)
            self.assertEqual(config.spam_suppression.clear_required_sec, 3.0)
            self.assertEqual(config.evidence.gif_frame_count, 5)
            self.assertEqual(config.evidence.gif_max_width, 640)
            self.assertEqual(config.evidence.gif_fps, 4.0)

    def test_runtime_config_prefers_agent_camera_and_evidence_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            canonical_evidence = tmpdir / "canonical-evidence"
            legacy_evidence = tmpdir / "legacy-evidence"
            config_path = tmpdir / "front_node.ini"
            config_path.write_text(
                f"""
[capture]
camera_index = 7

[webcam_source]
camera_index = 2

[evidence]
root = {canonical_evidence.as_posix()}

[outputs]
evidence_root = {legacy_evidence.as_posix()}
""".strip(),
                encoding="utf-8",
            )

            config = front_runtime_config.load_runtime_config(str(config_path))

        self.assertEqual(config.webcam_source.camera_index, 7)
        self.assertEqual(config.evidence_root, canonical_evidence.resolve(strict=False))

    def test_runtime_config_keeps_legacy_camera_and_evidence_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            legacy_evidence = tmpdir / "legacy-evidence"
            config_path = tmpdir / "front_node.ini"
            config_path.write_text(
                f"""
[webcam_source]
camera_index = 4

[outputs]
evidence_root = {legacy_evidence.as_posix()}
""".strip(),
                encoding="utf-8",
            )

            config = front_runtime_config.load_runtime_config(str(config_path))

        self.assertEqual(config.webcam_source.camera_index, 4)
        self.assertEqual(config.evidence_root, legacy_evidence.resolve(strict=False))

    def test_node_agent_config_keeps_legacy_camera_and_evidence_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            legacy_evidence = tmpdir / "legacy-evidence"
            config_path = tmpdir / "front_node.ini"
            config_path.write_text(
                f"""
[agent]
node_id = front

[capture]
source_mode = webcam

[webcam_source]
camera_index = 6

[outputs]
evidence_root = {legacy_evidence.as_posix()}
""".strip(),
                encoding="utf-8",
            )

            config = load_node_agent_config(config_path)

        self.assertEqual(config.camera_index, 6)
        self.assertEqual(config.evidence_root, legacy_evidence.resolve(strict=False))

    def test_central_dashboard_records_export_controls_are_removed(self):
        script = (ROOT / "central_service" / "static" / "dashboard.js").read_text(
            encoding="utf-8"
        )
        template = (ROOT / "central_service" / "templates" / "dashboard.html").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("records-export", template)
        self.assertNotIn("function exportRecords", script)

    def test_central_dashboard_records_pagination_controls_are_present(self):
        script = (ROOT / "central_service" / "static" / "dashboard.js").read_text(
            encoding="utf-8"
        )
        template = (ROOT / "central_service" / "templates" / "dashboard.html").read_text(
            encoding="utf-8"
        )
        stylesheet = (ROOT / "central_service" / "static" / "app.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("records-pagination-summary", template)
        self.assertIn("records-prev-page", template)
        self.assertIn("records-next-page", template)
        self.assertIn("const RECORDS_PAGE_SIZE = 10", script)
        self.assertIn("recordsPage", script)
        self.assertIn("function resetRecordsPagination", script)
        self.assertIn(".records-pagination", stylesheet)

    def test_front_node_records_export_controls_are_removed(self):
        script = (EDGE_NODE_ROOT / "web" / "static" / "dashboard.js").read_text(
            encoding="utf-8"
        )
        template = (EDGE_NODE_ROOT / "web" / "templates" / "dashboard.html").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("records-export", template)
        self.assertNotIn("function exportRecords", script)


if __name__ == "__main__":
    unittest.main()
