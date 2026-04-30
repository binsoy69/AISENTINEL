from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
import configparser
import importlib.util
import io
import json
import tempfile
import time
import unittest
import sys

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from edge_node_runtime import sound_monitor
from edge_node_runtime.sound_monitor import (
    SoundMonitorService,
    ThresholdCrossingGate,
    build_settings_from_sound_config,
)

CALIBRATION_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "calibrate_sound_sensor.py"
)
CALIBRATION_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "calibrate_sound_sensor",
    CALIBRATION_SCRIPT_PATH,
)
calibrate_sound_sensor = importlib.util.module_from_spec(CALIBRATION_SCRIPT_SPEC)
assert CALIBRATION_SCRIPT_SPEC.loader is not None
sys.modules[CALIBRATION_SCRIPT_SPEC.name] = calibrate_sound_sensor
CALIBRATION_SCRIPT_SPEC.loader.exec_module(calibrate_sound_sensor)


class SoundMonitorTests(unittest.TestCase):
    def test_sample_window_allows_uncalibrated_reference_capture(self):
        class FakeBus:
            def __init__(self):
                self.code = 100

            def read_i2c_block_data(self, _address, _register, _length):
                self.code += 8
                raw = self.code << 4
                return [(raw >> 8) & 0xFF, raw & 0xFF]

        result = sound_monitor.sample_window(
            FakeBus(),
            SimpleNamespace(
                address=0x48,
                full_scale=4.096,
                window_seconds=0.003,
                sample_interval=0.001,
                quiet_db=45.0,
                loud_db=55.0,
                ref_quiet_rms_mv=None,
                ref_loud_rms_mv=None,
            ),
        )

        self.assertIn("rms_mv", result)
        self.assertNotIn("estimated_db", result)

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

    def test_open_ads1015_closes_bus_when_initial_config_write_fails(self):
        class FakeBus:
            def __init__(self):
                self.closed = False

            def write_i2c_block_data(self, _address, _register, _data):
                raise OSError(121, "Remote I/O error")

            def close(self):
                self.closed = True

        fake_bus = FakeBus()
        original_load_smbus = sound_monitor.load_smbus
        sound_monitor.load_smbus = lambda: lambda _bus_number: fake_bus
        try:
            with self.assertRaises(OSError):
                sound_monitor.open_ads1015(
                    SimpleNamespace(
                        bus=1,
                        address=0x48,
                        channel=0,
                        full_scale=4.096,
                        data_rate=1600,
                    )
                )
        finally:
            sound_monitor.load_smbus = original_load_smbus

        self.assertTrue(fake_bus.closed)

    def test_print_i2c_error_uses_configured_bus_and_lists_common_addresses(self):
        output = io.StringIO()

        with redirect_stdout(output):
            sound_monitor.print_i2c_error(
                SimpleNamespace(bus=3, address=0x49),
                OSError(121, "Remote I/O error"),
            )

        text = output.getvalue()
        self.assertIn("0x49", text)
        self.assertIn("i2cdetect -y 3", text)
        self.assertIn("0x48, 0x49, 0x4A, or 0x4B", text)

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


class SoundCalibrationScriptTests(unittest.TestCase):
    def test_default_calibration_path_replaces_placeholder_with_node_data_path(self):
        parser = configparser.ConfigParser()
        parser.read_string(
            """
[agent]
node_id = front

[sound_sensor]
calibration_config = CHANGE_ME_SOUND_CALIBRATION.json
""".strip()
        )

        path = calibrate_sound_sensor.default_calibration_path(
            parser,
            Path("config/front_node.ini"),
        )

        self.assertEqual(
            path,
            (
                calibrate_sound_sensor.REPO_ROOT
                / "runtime/central_dashboard/data/node_front/sound/ky037_ads1015_config.json"
            ).resolve(strict=False),
        )

    def test_update_node_ini_writes_sound_path_and_enables_complete_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config_path = tmpdir / "front_node.ini"
            config_path.write_text(
                """
[agent]
node_id = front

[sound_sensor]
enabled = false
calibration_config =
""".strip(),
                encoding="utf-8",
            )
            parser = calibrate_sound_sensor.read_node_ini(config_path)
            calibration_path = (
                calibrate_sound_sensor.REPO_ROOT
                / "runtime/central_dashboard/data/node_front/sound/ky037_ads1015_config.json"
            )

            calibrate_sound_sensor.update_node_ini(
                config_path,
                parser,
                calibrate_sound_sensor.CalibrationSettings(
                    config_file=calibration_path,
                    bus=1,
                    address=0x49,
                    channel=2,
                    full_scale=4.096,
                    data_rate=1600,
                    sample_interval=0.002,
                    window_seconds=1.0,
                    quiet_db=45.0,
                    loud_db=55.0,
                    ref_quiet_rms_mv=12.3,
                    ref_loud_rms_mv=30.5,
                    alert_threshold_db=55.0,
                    incident_cooldown_sec=10.0,
                ),
                enable_complete_calibration=True,
            )

            updated = configparser.ConfigParser()
            updated.read(config_path, encoding="utf-8")

        self.assertTrue(updated.getboolean("sound_sensor", "enabled"))
        self.assertEqual(
            updated.get("sound_sensor", "calibration_config"),
            "runtime/central_dashboard/data/node_front/sound/ky037_ads1015_config.json",
        )
        self.assertEqual(updated.get("sound_sensor", "i2c_address"), "0x49")
        self.assertEqual(updated.getint("sound_sensor", "adc_channel"), 2)


if __name__ == "__main__":
    unittest.main()
