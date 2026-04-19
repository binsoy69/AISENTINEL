#!/usr/bin/env python3
"""
KY-037 continuous digital sound detector for Raspberry Pi.

This script reads the KY-037 DO/D0 pin only. It runs continuously until Ctrl+C
and prints only one of these two normal monitoring messages:

    no sound detected
    sound detected

Prerequisites on Raspberry Pi OS:
    sudo apt update
    sudo apt install python3-gpiozero python3-lgpio

Usage examples:
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py --pin 17
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py --active-high
"""

from __future__ import annotations

import argparse
import sys
import time


GPIO_INSTALL_COMMAND = "sudo apt install python3-gpiozero python3-lgpio"
DEFAULT_POLL_INTERVAL_SECONDS = 0.05
NO_SOUND_MESSAGE = "no sound detected"
SOUND_MESSAGE = "sound detected"


def interpret_sound_detected(raw_high: bool, active_high: bool) -> bool:
    """Map a raw GPIO level to the sound-detected state."""
    return raw_high if active_high else not raw_high


def load_digital_input_device():
    """Import gpiozero lazily so static checks can run off-Pi."""
    try:
        from gpiozero import DigitalInputDevice
    except ImportError:
        print("[ERROR] gpiozero is not installed.")
        print(f"Install GPIO dependencies with: {GPIO_INSTALL_COMMAND}")
        sys.exit(1)

    return DigitalInputDevice


def create_input_device(pin: int):
    """Create a digital input for the externally driven KY-037 DO pin."""
    DigitalInputDevice = load_digital_input_device()
    try:
        return DigitalInputDevice(pin=pin, pull_up=None, active_state=True)
    except Exception as exc:
        print(f"[ERROR] Could not open GPIO{pin}: {exc}")
        print("Confirm this is running on a Raspberry Pi with GPIO access.")
        print(f"Install GPIO dependencies with: {GPIO_INSTALL_COMMAND}")
        sys.exit(1)


def read_raw_high(device) -> bool:
    """Read the raw logic level. active_state=True makes value match HIGH."""
    return bool(device.value)


def status_message(sound_detected: bool) -> str:
    return SOUND_MESSAGE if sound_detected else NO_SOUND_MESSAGE


def monitor_gpio(args: argparse.Namespace) -> None:
    device = create_input_device(args.pin)

    try:
        raw_high = read_raw_high(device)
        stable_raw_high = raw_high
        candidate_raw_high = raw_high
        candidate_since = time.monotonic()
        current_sound_detected = interpret_sound_detected(raw_high, args.active_high)
        print(status_message(current_sound_detected), flush=True)

        debounce_seconds = args.debounce_ms / 1000.0

        while True:
            now = time.monotonic()
            raw_high = read_raw_high(device)

            if raw_high != candidate_raw_high:
                candidate_raw_high = raw_high
                candidate_since = now
                time.sleep(args.poll_interval)
                continue

            if raw_high != stable_raw_high and now - candidate_since >= debounce_seconds:
                stable_raw_high = raw_high
                new_sound_detected = interpret_sound_detected(
                    stable_raw_high,
                    args.active_high,
                )
                if new_sound_detected != current_sound_detected:
                    current_sound_detected = new_sound_detected
                    print(status_message(current_sound_detected), flush=True)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        return
    finally:
        device.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KY-037 continuous sound detector for Raspberry Pi",
    )
    parser.add_argument(
        "--pin",
        type=int,
        default=17,
        help="BCM GPIO pin connected to KY-037 DO/D0 (default: 17)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--active-low",
        dest="active_high",
        action="store_false",
        help="Treat raw GPIO LOW as sound detected (default)",
    )
    mode_group.add_argument(
        "--active-high",
        dest="active_high",
        action="store_true",
        help="Treat raw GPIO HIGH as sound detected",
    )
    parser.set_defaults(active_high=False)
    parser.add_argument(
        "--debounce-ms",
        type=int,
        default=50,
        help="Stable time required before accepting a GPIO state change (default: 50)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="Seconds between GPIO reads (default: 0.05)",
    )

    args = parser.parse_args()
    if args.pin < 0:
        parser.error("--pin must be a BCM GPIO number >= 0")
    if args.debounce_ms < 0:
        parser.error("--debounce-ms must be >= 0")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be > 0")

    return args


def main() -> int:
    monitor_gpio(parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
