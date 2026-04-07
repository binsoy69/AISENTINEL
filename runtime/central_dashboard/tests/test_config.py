from __future__ import annotations

import unittest
from pathlib import Path
import sys

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.central_service.config import load_central_service_config
from central_dashboard.node_agent.config import load_node_agent_config


ROOT = Path(__file__).resolve().parents[1]


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
        self.assertIn("runtime", str(config.evidence_root))
        self.assertIn("central_dashboard", str(config.evidence_root))


if __name__ == "__main__":
    unittest.main()
