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
    python3 tests/tests_on_pi/inmp441_mic_test.py --device plughw:1,0 --duration 8

Notes:
    - The INMP441 is a digital I2S microphone, not an analog mic.
    - The script assumes your Raspberry Pi already exposes the mic as an ALSA
      capture device. Verify with: arecord -l
"""

from __future__ import annotations

import argparse
import array
import math
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


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


def analyze_wav(path: Path) -> dict[str, float | int]:
    """Analyze a recorded WAV file and return basic audio statistics."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if sample_width != 4:
        raise ValueError(
            f"Expected 32-bit PCM data, got sample width {sample_width} bytes."
        )

    samples = array.array("i")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()

    if not samples:
        raise ValueError("No samples were recorded.")

    max_int = float((1 << 31) - 1)
    peak = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))

    peak_full_scale = peak / max_int
    rms_full_scale = rms / max_int

    return {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "duration_seconds": frame_count / sample_rate if sample_rate else 0.0,
        "peak": peak_full_scale,
        "peak_dbfs": dbfs(peak_full_scale),
        "rms": rms_full_scale,
        "rms_dbfs": dbfs(rms_full_scale),
    }


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


def print_analysis(stats: dict[str, float | int], output_path: Path) -> None:
    """Print recording summary and a simple pass/fail interpretation."""
    peak = float(stats["peak"])
    rms = float(stats["rms"])
    peak_db = float(stats["peak_dbfs"])
    rms_db = float(stats["rms_dbfs"])

    print("\n" + "=" * 60)
    print("Recording Summary")
    print("=" * 60)
    print(f"Saved file       : {output_path}")
    print(f"Sample rate      : {int(stats['sample_rate'])} Hz")
    print(f"Channels         : {int(stats['channels'])}")
    print(f"Duration         : {float(stats['duration_seconds']):.2f} s")
    print(f"Peak level       : {peak:.4f} ({peak_db:.1f} dBFS)")
    print(f"RMS level        : {rms:.4f} ({rms_db:.1f} dBFS)")
    print(f"Peak meter       : [{meter(peak)}]")
    print(f"RMS meter        : [{meter(min(rms * 4.0, 1.0))}]")

    print("\nAssessment:")
    if peak >= 0.95:
        print("- Signal is clipping or very close to clipping. Reduce gain or sound level.")
    elif peak >= 0.05:
        print("- Signal level looks healthy. Speak near the mic to confirm waveform quality.")
    elif peak >= 0.01:
        print("- Mic is working, but the level is low. Move closer or recheck the module orientation.")
    else:
        print("- Signal is extremely low. Wiring, I2S setup, or device selection is likely wrong.")

    if rms < 0.001:
        print("- RMS is close to silence. If you spoke during the test, recheck SCK, WS, and SD wiring.")


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


def write_wav(
    output_path: Path,
    raw_audio: bytes,
    channels: int,
    sample_rate: int,
    sample_width: int,
) -> None:
    """Write raw PCM bytes to a WAV file."""
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_audio)


def record_audio(
    device: str,
    duration: int,
    sample_rate: int,
    channels: int,
    fmt: str,
    output_path: Path,
) -> None:
    """Record audio with a live dB meter, then save it as a WAV file."""
    if fmt != "S32_LE":
        print("[ERROR] Live dB monitoring currently expects --format S32_LE.")
        print("Use the default format or extend the script for other PCM widths.")
        sys.exit(1)

    bytes_per_sample = 4
    frames_per_chunk = max(sample_rate // 10, 1024)
    chunk_bytes = frames_per_chunk * channels * bytes_per_sample
    target_bytes = duration * sample_rate * channels * bytes_per_sample

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

    captured = bytearray()
    start_time = time.time()

    try:
        while len(captured) < target_bytes:
            bytes_left = target_bytes - len(captured)
            current_chunk_size = min(chunk_bytes, bytes_left)
            chunk = process.stdout.read(current_chunk_size) if process.stdout else b""

            if not chunk:
                break

            captured.extend(chunk)
            live_stats = analyze_pcm_bytes(chunk)
            print_live_level(time.time() - start_time, live_stats)
    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted by user.")
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

    if not captured:
        stderr_output = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        print("[ERROR] Recording failed or returned no audio.")
        if stderr_output.strip():
            print(stderr_output.strip())
        print("\nTry listing devices first:")
        print("  python3 tests/tests_on_pi/inmp441_mic_test.py --list")
        sys.exit(1)

    write_wav(
        output_path=output_path,
        raw_audio=bytes(captured),
        channels=channels,
        sample_rate=sample_rate,
        sample_width=bytes_per_sample,
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
  python3 tests/tests_on_pi/inmp441_mic_test.py --device plughw:1,0 --duration 8
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
        default=5,
        help="Recording duration in seconds (default: 5)",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output WAV path",
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or Path("test_captures") / f"inmp441_test_{timestamp}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nUsing device     : {device}")
    print(f"Recording length : {args.duration} s")
    print(f"Sample rate      : {args.rate} Hz")
    print(f"Format           : {args.format}")
    print(f"Output file      : {output_path}")
    print("\nSpeak or clap near the microphone while the capture is running.")

    record_audio(
        device=device,
        duration=args.duration,
        sample_rate=args.rate,
        channels=args.channels,
        fmt=args.format,
        output_path=output_path,
    )

    try:
        stats = analyze_wav(output_path)
    except ValueError as error:
        print(f"\n[ERROR] Could not analyze recording: {error}")
        print("If your device uses another sample format, rerun with a matching `--format`.")
        return 1

    print_analysis(stats, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
