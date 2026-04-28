from __future__ import annotations

import unittest
from pathlib import Path
import sys

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.central_service.config import load_central_service_config
from central_dashboard.node_agent.config import load_node_agent_config
from central_dashboard.node_agent.front_runtime import front_runtime_config


ROOT = Path(__file__).resolve().parents[1]
FRONT_NODE_ROOT = ROOT.parent / "front_node_pi"


class ConfigTests(unittest.TestCase):
    def test_load_central_service_config(self):
        config = load_central_service_config(ROOT / "central_service.ini")
        self.assertEqual(config.port, 8090)
        self.assertIn("front", config.known_nodes)
        self.assertIn("mid", config.known_nodes)
        self.assertEqual(config.known_nodes["front"].camera_label, "Front Camera")

    def test_load_node_agent_config(self):
        config = load_node_agent_config(ROOT / "node_front.ini")
        self.assertEqual(config.node_id, "front")
        self.assertEqual(config.port, 8091)
        self.assertEqual(config.source_mode, "webcam")
        self.assertEqual(config.detector_mode, "front_runtime")
        self.assertEqual(
            config.runtime_config_path,
            (ROOT / "node_front_runtime.ini").resolve(strict=False),
        )
        self.assertIn("runtime", str(config.evidence_root))
        self.assertIn("central_dashboard", str(config.evidence_root))

    def test_load_front_runtime_sound_config(self):
        config = front_runtime_config.load_runtime_config(
            str(ROOT / "node_front_runtime.ini")
        )
        self.assertTrue(config.sound_sensor.enabled)
        self.assertEqual(config.sound_sensor.i2c_address, 0x48)
        self.assertEqual(config.sound_sensor.alert_threshold_db, 55.0)
        self.assertEqual(
            config.sound_sensor.calibration_config,
            (ROOT.parents[1] / "tests" / "tests_on_pi" / "ky037_ads1015_config.json").resolve(strict=False),
        )

    def test_dashboard_js_renders_evidence_processing_snapshot_state(self):
        script = (ROOT / "central_service" / "static" / "dashboard.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("function evidenceCellMarkup", script)
        self.assertIn("Evidence processing", script)
        self.assertIn("View Snapshot", script)
        self.assertNotIn("View GIF", script)
        self.assertLess(
            script.index("if (incident?.poster_url)"),
            script.index("Evidence processing"),
        )

    def test_front_and_mid_runtime_configs_load_spam_suppression(self):
        for filename in ("node_front_runtime.ini", "node_mid_runtime.ini"):
            config = front_runtime_config.load_runtime_config(str(ROOT / filename))
            self.assertEqual(config.spam_suppression.duplicate_suppression_sec, 60.0)
            self.assertEqual(config.spam_suppression.clear_required_sec, 3.0)

    def test_central_dashboard_records_export_controls_are_removed(self):
        script = (ROOT / "central_service" / "static" / "dashboard.js").read_text(
            encoding="utf-8"
        )
        template = (ROOT / "central_service" / "templates" / "dashboard.html").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("records-export", template)
        self.assertNotIn("function exportRecords", script)

    def test_front_node_records_export_controls_are_removed(self):
        script = (FRONT_NODE_ROOT / "web" / "static" / "dashboard.js").read_text(
            encoding="utf-8"
        )
        template = (FRONT_NODE_ROOT / "web" / "templates" / "dashboard.html").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("records-export", template)
        self.assertNotIn("function exportRecords", script)


if __name__ == "__main__":
    unittest.main()
