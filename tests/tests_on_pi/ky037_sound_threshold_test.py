#!/usr/bin/env python3
"""
KY-037 continuous digital sound detector for Raspberry Pi.

This version follows the RPi.GPIO pattern from the KY-037 Raspberry Pi
tutorial: read the digital D0 pin in an infinite loop and print the current
state continuously.

Normal monitoring output is limited to:

    no sound detected
    sound detected

Prerequisite on Raspberry Pi OS:
    sudo apt update
    sudo apt install python3-rpi.gpio

Usage:
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py --debug
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py --active-high
"""

from __future__ import annotations

import argparse
import sys
import time


GPIO_INSTALL_COMMAND = "sudo apt install python3-rpi.gpio"
DEFAULT_SOUND_SENSOR_PIN = 4
DEFAULT_POLL_INTERVAL_SECONDS = 0.2
NO_SOUND_MESSAGE = "no sound detected"
SOUND_MESSAGE = "sound detected"


def load_gpio():
    """Import RPi.GPIO lazily so --help still works off the Raspberry Pi."""
    try:
        import RPi.GPIO as GPIO
    except ImportError:
        print("[ERROR] RPi.GPIO is not installed.")
        print(f"Install GPIO support with: {GPIO_INSTALL_COMMAND}")
        sys.exit(1)

    return GPIO


def is_sound_detected(GPIO, sound_sensor_pin: int, active_low: bool) -> tuple[bool, int]:
    """Return interpreted sound state plus the raw GPIO value."""
    raw_value = GPIO.input(sound_sensor_pin)
    if active_low:
        return raw_value == GPIO.LOW, raw_value
    return raw_value == GPIO.HIGH, raw_value


def detect_sound(GPIO, args: argparse.Namespace) -> None:
    """Read the KY-037 digital output and print the current detection state."""
    sound_detected, raw_value = is_sound_detected(GPIO, args.pin, args.active_low)
    if sound_detected:
        print(SOUND_MESSAGE, flush=True)
    else:
        print(NO_SOUND_MESSAGE, flush=True)
    if args.debug:
        print(f"raw GPIO{args.pin}={raw_value}", flush=True)


def monitor_gpio(args: argparse.Namespace) -> None:
    GPIO = load_gpio()
    GPIO.setmode(GPIO.BCM)
    pull_mode = GPIO.PUD_UP if args.active_low else GPIO.PUD_DOWN
    GPIO.setup(args.pin, GPIO.IN, pull_up_down=pull_mode)

    try:
        while True:
            detect_sound(GPIO, args)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        return
    finally:
        GPIO.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KY-037 continuous sound detector using RPi.GPIO",
    )
    parser.add_argument(
        "--pin",
        type=int,
        default=DEFAULT_SOUND_SENSOR_PIN,
        help="BCM GPIO pin connected to KY-037 D0 (default: 4)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--active-low",
        dest="active_low",
        action="store_true",
        help="Treat raw GPIO LOW as sound detected (default)",
    )
    mode_group.add_argument(
        "--active-high",
        dest="active_low",
        action="store_false",
        help="Treat raw GPIO HIGH as sound detected",
    )
    parser.set_defaults(active_low=True)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="Seconds between GPIO reads (default: 0.2)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also print the raw GPIO value after each detection message",
    )

    args = parser.parse_args()
    if args.pin < 0:
        parser.error("--pin must be a BCM GPIO number >= 0")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be > 0")

    return args


def main() -> int:
    monitor_gpio(parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
