#!/usr/bin/env python3
"""
KY-037 + ADS1015 calibration script for Raspberry Pi.

This script captures the quiet and loud classroom reference levels and saves
them to a JSON file. By default it runs an interactive two-step calibration:

1. capture quiet reference near the lower threshold, usually 45 dB
2. capture loud reference near the upper threshold, usually 55 dB

Typical usage:
    python3 tests/tests_on_pi/diagnostics/ky037_ads1015_calibrate.py
    python3 tests/tests_on_pi/diagnostics/ky037_ads1015_calibrate.py --capture-quiet
    python3 tests/tests_on_pi/diagnostics/ky037_ads1015_calibrate.py --capture-loud
    python3 tests/tests_on_pi/diagnostics/ky037_ads1015_calibrate.py --show-config
"""

from __future__ import annotations

import argparse
import sys

from ky037_ads1015_common import (
    DATA_RATE_TO_BITS,
    DEFAULT_CONFIG_PATH,
    FULL_SCALE_TO_PGA_BITS,
    close_bus,
    open_ads1015,
    print_config,
    print_i2c_error,
    resolve_config,
    sample_window,
    save_config_file,
    validate_settings,
)


def print_capture_result(label: str, result: dict[str, float], settings: argparse.Namespace) -> None:
    """Print a compact line after each saved calibration capture."""
    print(
        f"{label}_reference rms={result['rms_mv']:.2f}mV "
        f"p2p={result['p2p_mv']:.2f}mV"
    )
    if settings.debug:
        print(
            f"debug address=0x{settings.address:02X} channel=A{settings.channel} "
            f"mean={result['mean_v']:.4f}V min={result['min_v']:.4f}V "
            f"max={result['max_v']:.4f}V rate={settings.data_rate}SPS "
            f"samples={int(result['sample_count'])}"
        )


def capture_reference(settings: argparse.Namespace, label: str) -> None:
    """Capture one calibration window and save it as quiet or loud reference."""
    bus = None
    try:
        bus = open_ads1015(settings)
        result = sample_window(bus, settings)
    except OSError as exc:
        print_i2c_error(settings, exc)
        sys.exit(1)
    finally:
        if bus is not None:
            close_bus(bus)

    if label == "quiet":
        settings.ref_quiet_rms_mv = result["rms_mv"]
    else:
        settings.ref_loud_rms_mv = result["rms_mv"]

    save_config_file(settings.config_file, settings)
    print_capture_result(label, result, settings)
    print(
        f"[INFO] Saved {label} reference to {settings.config_file}: "
        f"{result['rms_mv']:.2f}mV RMS"
    )


def wait_for_enter(prompt: str) -> None:
    """Pause for the operator when running interactively."""
    try:
        input(prompt)
    except EOFError:
        print("[INFO] No interactive input available. Capturing immediately.")


def run_interactive_calibration(settings: argparse.Namespace) -> None:
    """Guide the operator through quiet and loud reference capture."""
    print("KY-037 ADS1015 Calibration")
    print(f"Config file : {settings.config_file}")
    print(f"Quiet target: {settings.quiet_db:.1f} dB")
    print(f"Loud target : {settings.loud_db:.1f} dB")
    print()
    print("Use a phone SPL app or sound level meter beside the sensor.")
    print("The script will save the measured RMS values to the JSON file.")
    print()

    wait_for_enter(
        f"Set the room to about {settings.quiet_db:.1f} dB, then press Enter to capture quiet reference..."
    )
    capture_reference(settings, "quiet")
    print()
    wait_for_enter(
        f"Set the room to about {settings.loud_db:.1f} dB, then press Enter to capture loud reference..."
    )
    capture_reference(settings, "loud")
    print()
    print("[INFO] Calibration complete.")
    print(
        "Run `python3 tests/tests_on_pi/diagnostics/ky037_sound_threshold_test.py` to "
        "monitor estimated dB using the saved values."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KY-037 ADS1015 calibration script for Raspberry Pi",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="JSON file used to load and save calibration/settings",
    )
    parser.add_argument(
        "--bus",
        type=int,
        default=None,
        help="I2C bus number for the ADS1015",
    )
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 0),
        default=None,
        help="ADS1015 I2C address in decimal or hex, for example 0x48",
    )
    parser.add_argument(
        "--channel",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help="ADS1015 single-ended input channel connected to KY-037 AO",
    )
    parser.add_argument(
        "--full-scale",
        type=float,
        choices=tuple(FULL_SCALE_TO_PGA_BITS.keys()),
        default=None,
        help="ADS1015 full-scale voltage range in volts",
    )
    parser.add_argument(
        "--data-rate",
        type=int,
        choices=tuple(DATA_RATE_TO_BITS.keys()),
        default=None,
        help="ADS1015 conversion rate in samples per second",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=None,
        help="Seconds between conversion register reads inside each window",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Measurement window size in seconds",
    )
    parser.add_argument(
        "--quiet-db",
        type=float,
        default=None,
        help="Lower classroom threshold in dB after calibration",
    )
    parser.add_argument(
        "--loud-db",
        type=float,
        default=None,
        help="Upper classroom threshold in dB after calibration",
    )
    parser.add_argument(
        "--ref-quiet-rms-mv",
        type=float,
        default=None,
        help="Optional manual quiet reference override before saving config",
    )
    parser.add_argument(
        "--ref-loud-rms-mv",
        type=float,
        default=None,
        help="Optional manual loud reference override before saving config",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also print mean/min/max voltage and sample count after capture",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save the resolved settings to the config file and exit",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved config and exit",
    )
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--capture-quiet",
        action="store_true",
        help="Capture one window and save its RMS value as the quiet reference",
    )
    capture_group.add_argument(
        "--capture-loud",
        action="store_true",
        help="Capture one window and save its RMS value as the loud reference",
    )

    settings = resolve_config(
        parser.parse_args(),
        passthrough_keys=(
            "debug",
            "save_config",
            "show_config",
            "capture_quiet",
            "capture_loud",
        ),
    )
    validate_settings(parser, settings)
    return settings


def main() -> int:
    settings = parse_args()

    if settings.save_config:
        save_config_file(settings.config_file, settings)
        print(f"[INFO] Saved config to {settings.config_file}")
        return 0

    if settings.show_config:
        print_config(settings)
        return 0

    if settings.capture_quiet:
        capture_reference(settings, "quiet")
        return 0

    if settings.capture_loud:
        capture_reference(settings, "loud")
        return 0

    run_interactive_calibration(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
