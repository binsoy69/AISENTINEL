#!/usr/bin/env python3
"""
INMP441 microphone terminal test for Raspberry Pi.

This script intentionally uses ALSA's built-in terminal VU meter instead of
parsing raw PCM in Python. It is more reliable on Raspberry Pi for quick
hardware validation.

Prerequisites on Raspberry Pi OS:
    sudo apt update
    sudo apt install alsa-utils

Usage examples:
    python3 tests/tests_on_pi/inmp441_mic_test.py --list
    python3 tests/tests_on_pi/inmp441_mic_test.py --device plughw:1,0
    python3 tests/tests_on_pi/inmp441_mic_test.py --device plughw:1,0 --duration 10

Notes:
    - The INMP441 is a digital I2S microphone, not an analog mic.
    - The script assumes your Raspberry Pi already exposes the mic as an ALSA
      capture device. Verify with: arecord -l
    - The monitor does not save any audio files.
    - Stop the monitor with Ctrl+C.
"""

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


CAPTURE_DEVICE_PATTERN = re.compile(
    r"card\s+(?P<card>\d+):\s+(?P<card_id>[^\[]+)\[(?P<card_name>[^\]]+)\],\s+"
    r"device\s+(?P<device>\d+):\s+(?P<device_id>[^\[]+)\[(?P<device_name>[^\]]+)\]"
)


@dataclass
class CaptureDevice:
    card_index: int
    card_id: str
    card_name: str
    device_index: int
    device_id: str
    device_name: str

    @property
    def alsa_name(self) -> str:
        return f"plughw:{self.card_index},{self.device_index}"


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command and capture stdout/stderr."""
    return subprocess.run(command, capture_output=True, text=True, check=False)


def ensure_arecord_available() -> None:
    """Exit early if arecord is not installed."""
    if shutil.which("arecord"):
        return

    print("[ERROR] `arecord` was not found.")
    print("Install it with: sudo apt install alsa-utils")
    sys.exit(1)


def list_capture_devices() -> list[CaptureDevice]:
    """Return parsed ALSA capture devices from `arecord -l`."""
    result = run_command(["arecord", "-l"])
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")

    devices: list[CaptureDevice] = []
    for match in CAPTURE_DEVICE_PATTERN.finditer(output):
        devices.append(
            CaptureDevice(
                card_index=int(match.group("card")),
                card_id=match.group("card_id").strip(),
                card_name=match.group("card_name").strip(),
                device_index=int(match.group("device")),
                device_id=match.group("device_id").strip(),
                device_name=match.group("device_name").strip(),
            )
        )

    return devices


def print_capture_devices(devices: list[CaptureDevice]) -> None:
    """Pretty-print capture devices."""
    print("\n" + "=" * 60)
    print("ALSA Capture Devices")
    print("=" * 60)

    if not devices:
        print("No capture devices detected by ALSA.")
        print("Check wiring and confirm the I2S mic is exposed by the OS.")
        return

    for device in devices:
        print(
            f"- {device.alsa_name:<12} "
            f"card={device.card_name} ({device.card_id}) | "
            f"device={device.device_name} ({device.device_id})"
        )


def choose_device(user_device: str | None, devices: list[CaptureDevice]) -> str:
    """Pick a device string to pass to arecord."""
    if user_device:
        return user_device

    if len(devices) == 1:
        return devices[0].alsa_name

    if not devices:
        print("[ERROR] No ALSA capture device found.")
        print("Run `arecord -l` and fix the I2S driver or wiring first.")
        sys.exit(1)

    print("[ERROR] Multiple capture devices were found. Select one explicitly.")
    print("Example:")
    print("  python3 tests/tests_on_pi/inmp441_mic_test.py --device plughw:1,0")
    sys.exit(1)


def monitor_audio(
    device: str,
    sample_rate: int,
    channels: int,
    fmt: str,
    duration: int | None,
) -> None:
    """Run ALSA's live terminal meter without saving a file."""
    command = [
        "arecord",
        "-D",
        device,
        "-c",
        str(channels),
        "-r",
        str(sample_rate),
        "-f",
        fmt,
        "-t",
        "raw",
        "-V",
        "mono",
        "-v",
        "/dev/null",
    ]

    if duration is not None:
        command.extend([
            "-d",
            str(duration),
        ])

    print("\nALSA live meter starting. Press Ctrl+C to stop.\n")

    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        return

    print(f"\n[ERROR] arecord exited with code {result.returncode}.")
    print("Try the same command directly in the terminal to see ALSA's full error output:")
    print(
        "  "
        + " ".join(
            [
                "arecord",
                "-D",
                device,
                "-c",
                str(channels),
                "-r",
                str(sample_rate),
                "-f",
                fmt,
                "-t",
                "raw",
                "-V",
                "mono",
                "-v",
                "/dev/null",
            ]
        )
    )
    sys.exit(result.returncode or 1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="INMP441 I2S microphone test for Raspberry Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/tests_on_pi/inmp441_mic_test.py --list
  python3 tests/tests_on_pi/inmp441_mic_test.py --device plughw:1,0
  python3 tests/tests_on_pi/inmp441_mic_test.py --device plughw:1,0 --duration 10
""",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List ALSA capture devices and exit",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="ALSA capture device, for example plughw:1,0",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Optional monitor duration in seconds. Use 0 to run until quit (default: 0)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=48000,
        help="Sample rate in Hz (default: 48000)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of channels to record (default: 1)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="S32_LE",
        help="ALSA sample format (default: S32_LE)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("INMP441 Microphone Test")
    print("AISENTINEL Project - Raspberry Pi")
    print("=" * 60)

    args = parse_args()
    ensure_arecord_available()

    devices = list_capture_devices()

    if args.list:
        print_capture_devices(devices)
        return 0

    device = choose_device(args.device, devices)
    duration = args.duration if args.duration > 0 else None

    print(f"\nUsing device     : {device}")
    print(f"Sample rate      : {args.rate} Hz")
    print(f"Format           : {args.format}")
    print(f"Channels         : {args.channels}")
    if duration is None:
        print("Run mode         : Continuous until Ctrl+C")
    else:
        print(f"Run mode         : {duration} s")
    print("\nSpeak or clap near the microphone.")

    monitor_audio(
        device=device,
        sample_rate=args.rate,
        channels=args.channels,
        fmt=args.format,
        duration=duration,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
