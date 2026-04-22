from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import time
import unittest
import sys

TEST_ROOT = Path(__file__).resolve().parents[2]
FRONT_RUNTIME_DIR = TEST_ROOT / "front_node_pi"
if str(FRONT_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(FRONT_RUNTIME_DIR))

from sound_monitor import SoundMonitorService, ThresholdCrossingGate, build_settings_from_sound_config


class SoundMonitorTests(unittest.TestCase):
    def test_build_settings_from_sound_config_reads_calibration_file(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            calibration_path = tmpdir / "ky037.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "quiet_db": 45.0,
                        "loud_db": 55.0,
                        "ref_quiet_rms_mv": 14.12,
                        "ref_loud_rms_mv": 29.84,
                    }
                ),
                encoding="utf-8",
            )

            settings = build_settings_from_sound_config(
                SimpleNamespace(
                    enabled=True,
                    calibration_config=calibration_path,
                    alert_threshold_db=58.0,
                    incident_cooldown_sec=12.0,
                    i2c_bus=1,
                    i2c_address=0x48,
                    adc_channel=0,
                    full_scale=4.096,
                    data_rate=1600,
                    sample_interval=0.002,
                    window_seconds=1.0,
                )
            )

            self.assertEqual(settings.address, 0x48)
            self.assertEqual(settings.channel, 0)
            self.assertEqual(settings.alert_threshold_db, 58.0)
            self.assertEqual(settings.ref_quiet_rms_mv, 14.12)
            self.assertEqual(settings.ref_loud_rms_mv, 29.84)

    def test_threshold_crossing_gate_requires_reset_and_cooldown(self):
        gate = ThresholdCrossingGate()

        self.assertFalse(
            gate.should_emit(
                54.0,
                55.0,
                now_monotonic=0.0,
                cooldown_sec=10.0,
            )
        )
        self.assertTrue(
            gate.should_emit(
                56.0,
                55.0,
                now_monotonic=1.0,
                cooldown_sec=10.0,
            )
        )
        self.assertFalse(
            gate.should_emit(
                57.0,
                55.0,
                now_monotonic=2.0,
                cooldown_sec=10.0,
            )
        )
        self.assertFalse(
            gate.should_emit(
                54.5,
                55.0,
                now_monotonic=3.0,
                cooldown_sec=10.0,
            )
        )
        self.assertFalse(
            gate.should_emit(
                56.5,
                55.0,
                now_monotonic=5.0,
                cooldown_sec=10.0,
            )
        )
        self.assertFalse(
            gate.should_emit(
                54.0,
                55.0,
                now_monotonic=11.5,
                cooldown_sec=10.0,
            )
        )
        self.assertTrue(
            gate.should_emit(
                56.2,
                55.0,
                now_monotonic=12.0,
                cooldown_sec=10.0,
            )
        )

    def test_sound_monitor_service_updates_snapshot_and_emits_threshold_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            calibration_path = tmpdir / "ky037.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "quiet_db": 45.0,
                        "loud_db": 55.0,
                        "ref_quiet_rms_mv": 14.12,
                        "ref_loud_rms_mv": 29.84,
                    }
                ),
                encoding="utf-8",
            )

            telemetry_events = []
            threshold_events = []
            sample_results = iter(
                [
                    {"estimated_db": 56.0, "rms_mv": 24.0, "p2p_mv": 88.0},
                    {"estimated_db": 57.2, "rms_mv": 25.2, "p2p_mv": 90.0},
                    {"estimated_db": 54.0, "rms_mv": 18.4, "p2p_mv": 61.0},
                ]
            )

            def fake_sample_window(_bus, _settings):
                try:
                    return next(sample_results)
                except StopIteration as exc:
                    raise RuntimeError("done") from exc

            service = SoundMonitorService(
                SimpleNamespace(
                    enabled=True,
                    calibration_config=calibration_path,
                    alert_threshold_db=55.0,
                    incident_cooldown_sec=10.0,
                    i2c_bus=1,
                    i2c_address=0x48,
                    adc_channel=0,
                    full_scale=4.096,
                    data_rate=1600,
                    sample_interval=0.002,
                    window_seconds=0.01,
                ),
                on_telemetry=lambda payload: telemetry_events.append(dict(payload)),
                on_threshold_cross=lambda payload: threshold_events.append(dict(payload)),
                log_fn=lambda _message: None,
                open_bus_fn=lambda _settings: object(),
                sample_window_fn=fake_sample_window,
                close_bus_fn=lambda _bus: None,
            )

            service.start()
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if service.snapshot().get("status") == "error":
                    break
                time.sleep(0.01)

            snapshot = service.snapshot()
            service.stop()

            self.assertGreaterEqual(len(telemetry_events), 3)
            self.assertEqual(len(threshold_events), 1)
            self.assertEqual(threshold_events[0]["threshold_db"], 55.0)
            self.assertEqual(snapshot["status"], "error")
            self.assertEqual(snapshot["last_error"], "done")


if __name__ == "__main__":
    unittest.main()
