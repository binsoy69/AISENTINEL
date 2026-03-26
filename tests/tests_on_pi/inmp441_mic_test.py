#!/usr/bin/env python3
"""
INMP441 microphone test script for Raspberry Pi.

This script uses ALSA's `arecord` utility so it works on a Raspberry Pi
without extra Python audio packages.

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
    - By default the test runs continuously until you quit with `q` or Ctrl+C.
"""

from __future__ import annotations

import argparse
import array
import math
import re
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
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


def dbfs(level: float) -> float:
    """Convert a linear full-scale value to dBFS."""
    if level <= 0:
        return float("-inf")
    return 20.0 * math.log10(level)


def meter(level: float, width: int = 40) -> str:
    """Return a simple ASCII bar meter."""
    level = max(0.0, min(level, 1.0))
    filled = int(round(level * width))
    return "#" * filled + "-" * (width - filled)


def analyze_pcm_bytes(raw_bytes: bytes) -> dict[str, float]:
    """Analyze a raw 32-bit little-endian PCM chunk."""
    samples = array.array("i")
    samples.frombytes(raw_bytes)
    if sys.byteorder != "little":
        samples.byteswap()

    if not samples:
        return {
            "peak": 0.0,
            "peak_dbfs": float("-inf"),
            "rms": 0.0,
            "rms_dbfs": float("-inf"),
        }

    max_int = float((1 << 31) - 1)
    peak = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))

    peak_full_scale = peak / max_int
    rms_full_scale = rms / max_int

    return {
        "peak": peak_full_scale,
        "peak_dbfs": dbfs(peak_full_scale),
        "rms": rms_full_scale,
        "rms_dbfs": dbfs(rms_full_scale),
    }


class TerminalKeyReader:
    """Read single keys from a terminal without blocking."""

    def __init__(self) -> None:
        self.enabled = False
        self.fd: int | None = None
        self.old_settings: list[int] | None = None

    def __enter__(self) -> "TerminalKeyReader":
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def quit_requested(self) -> bool:
        if not self.enabled:
            return False

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return False

        key = sys.stdin.read(1)
        return key.lower() == "q"


def print_live_level(elapsed_seconds: float, stats: dict[str, float]) -> None:
    """Print a live level meter on a single terminal line."""
    rms_dbfs = stats["rms_dbfs"]
    peak_dbfs = stats["peak_dbfs"]
    level = stats["rms"]
    normalized = 0.0 if rms_dbfs == float("-inf") else min(max((rms_dbfs + 60.0) / 60.0, 0.0), 1.0)

    sys.stdout.write(
        "\r"
        f"[{elapsed_seconds:5.1f}s] "
        f"current RMS: {rms_dbfs:6.1f} dBFS | "
        f"peak: {peak_dbfs:6.1f} dBFS | "
        f"[{meter(normalized)}]"
    )
    sys.stdout.flush()


def print_session_summary(
    elapsed_seconds: float,
    session_peak_dbfs: float,
    session_rms_dbfs: float,
) -> None:
    """Print a short summary when monitoring stops."""
    print("\n" + "=" * 60)
    print("Live Monitor Summary")
    print("=" * 60)
    print(f"Duration         : {elapsed_seconds:.1f} s")
    print(f"Highest peak     : {session_peak_dbfs:.1f} dBFS")
    print(f"Highest RMS      : {session_rms_dbfs:.1f} dBFS")
    print("=" * 60)


def monitor_audio(
    device: str,
    sample_rate: int,
    channels: int,
    fmt: str,
    duration: int | None,
) -> None:
    """Monitor live audio level in the terminal without saving files."""
    if fmt != "S32_LE":
        print("[ERROR] Live dB monitoring currently expects --format S32_LE.")
        print("Use the default format or extend the script for other PCM widths.")
        sys.exit(1)

    bytes_per_sample = 4
    frames_per_chunk = max(sample_rate // 10, 1024)
    chunk_bytes = frames_per_chunk * channels * bytes_per_sample

    command = [
        "arecord",
        "-q",
        "-D",
        device,
        "-f",
        fmt,
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
        "-t",
        "raw",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    start_time = time.time()
    session_peak_dbfs = float("-inf")
    session_rms_dbfs = float("-inf")

    try:
        with TerminalKeyReader() as key_reader:
            while True:
                if duration is not None and (time.time() - start_time) >= duration:
                    break

                chunk = process.stdout.read(chunk_bytes) if process.stdout else b""
                if not chunk:
                    break

                live_stats = analyze_pcm_bytes(chunk)
                session_peak_dbfs = max(session_peak_dbfs, live_stats["peak_dbfs"])
                session_rms_dbfs = max(session_rms_dbfs, live_stats["rms_dbfs"])
                print_live_level(time.time() - start_time, live_stats)

                if key_reader.quit_requested():
                    print("\n[INFO] Quit requested.")
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring interrupted by user.")
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    sys.stdout.write("\n")
    sys.stdout.flush()

    stderr_output = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if session_peak_dbfs == float("-inf"):
        print("[ERROR] Monitoring failed or returned no audio.")
        if stderr_output.strip():
            print(stderr_output.strip())
        print("\nTry listing devices first:")
        print("  python3 tests/tests_on_pi/inmp441_mic_test.py --list")
        sys.exit(1)

    print_session_summary(
        elapsed_seconds=time.time() - start_time,
        session_peak_dbfs=session_peak_dbfs,
        session_rms_dbfs=session_rms_dbfs,
    )


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
        print("Run mode         : Continuous until `q` or Ctrl+C")
    else:
        print(f"Run mode         : {duration} s")
    print("\nSpeak or clap near the microphone. Press `q` or Ctrl+C to stop.")

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
