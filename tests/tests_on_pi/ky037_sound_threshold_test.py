#!/usr/bin/env python3
"""
KY-037 digital sound-threshold monitor for Raspberry Pi.

This test reads the KY-037 DO/D0 pin only. It does not estimate real dB,
because Raspberry Pi GPIO has no analog input and this setup does not use an
ADC for the KY-037 AO/A0 output.

Prerequisites on Raspberry Pi OS:
    sudo apt update
    sudo apt install python3-gpiozero python3-lgpio

Usage examples:
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py --duration 30
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py --pin 17 --active-high
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py --csv ky037_noise_log.csv --duration 300

Notes:
    - Default wiring uses KY-037 DO/D0 -> Raspberry Pi GPIO17, physical pin 11.
    - Default interpretation is active-low: raw GPIO LOW means noisy.
    - If the quiet/noisy state is reversed, rerun with --active-high.
    - Stop a continuous run with Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO


GPIO_INSTALL_COMMAND = "sudo apt install python3-gpiozero python3-lgpio"
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.05


@dataclass
class MonitorState:
    start_time: float
    last_accounted_time: float
    stable_raw_high: bool
    current_raw_high: bool
    current_noisy: bool
    candidate_raw_high: bool
    candidate_since: float
    noisy_seconds: float = 0.0
    noisy_events: int = 0
    transitions: int = 0
    samples: int = 0

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.start_time)

    def noisy_percent(self, now: float) -> float:
        elapsed = self.elapsed(now)
        if elapsed <= 0:
            return 0.0
        return min(100.0, max(0.0, (self.noisy_seconds / elapsed) * 100.0))


def local_timestamp() -> str:
    """Return a local ISO timestamp suitable for terminal and CSV logs."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def raw_label(raw_high: bool) -> str:
    return "HIGH" if raw_high else "LOW"


def state_label(noisy: bool) -> str:
    return "NOISY" if noisy else "quiet"


def interpret_noisy(raw_high: bool, active_high: bool) -> bool:
    """Map a raw GPIO level to the classroom noise state."""
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
    """Create a floating digital input for an externally driven KY-037 DO pin."""
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


def open_csv_writer(csv_path: str | None) -> tuple[TextIO | None, csv.DictWriter | None]:
    """Open the optional CSV log and write its header."""
    if not csv_path:
        return None, None

    path = Path(csv_path).expanduser()
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)

    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "timestamp",
            "elapsed_s",
            "pin_bcm",
            "raw_gpio_state",
            "interpreted_state",
            "noisy_events",
            "transitions",
            "noisy_time_percent",
            "samples",
        ],
    )
    writer.writeheader()
    handle.flush()
    return handle, writer


def write_csv_row(
    writer: csv.DictWriter | None,
    args: argparse.Namespace,
    state: MonitorState,
    now: float,
) -> None:
    """Append one timestamped reading when CSV logging is enabled."""
    if writer is None:
        return

    writer.writerow(
        {
            "timestamp": local_timestamp(),
            "elapsed_s": f"{state.elapsed(now):.3f}",
            "pin_bcm": args.pin,
            "raw_gpio_state": raw_label(state.stable_raw_high),
            "interpreted_state": state_label(state.current_noisy),
            "noisy_events": state.noisy_events,
            "transitions": state.transitions,
            "noisy_time_percent": f"{state.noisy_percent(now):.2f}",
            "samples": state.samples,
        }
    )


def account_elapsed_time(state: MonitorState, now: float) -> None:
    """Accumulate noisy time since the previous sample."""
    delta = max(0.0, now - state.last_accounted_time)
    if state.current_noisy:
        state.noisy_seconds += delta
    state.last_accounted_time = now


def process_sample(
    state: MonitorState,
    raw_high: bool,
    active_high: bool,
    debounce_seconds: float,
    now: float,
) -> bool:
    """
    Update debounced GPIO state.

    Returns True when the interpreted quiet/noisy state changed.
    """
    state.samples += 1
    state.current_raw_high = raw_high

    if raw_high != state.candidate_raw_high:
        state.candidate_raw_high = raw_high
        state.candidate_since = now
        return False

    if raw_high == state.stable_raw_high:
        return False

    if now - state.candidate_since < debounce_seconds:
        return False

    previous_noisy = state.current_noisy
    state.stable_raw_high = raw_high
    state.current_noisy = interpret_noisy(raw_high, active_high)
    state.transitions += 1

    if state.current_noisy and not previous_noisy:
        state.noisy_events += 1

    return state.current_noisy != previous_noisy


def print_intro(args: argparse.Namespace) -> None:
    active_mode = (
        "active-high: raw HIGH means noisy"
        if args.active_high
        else "active-low: raw LOW means noisy"
    )
    run_mode = "Continuous until Ctrl+C" if args.duration == 0 else f"{args.duration} s"

    print("=" * 68)
    print("KY-037 Sound Threshold Test")
    print("AISENTINEL Project - Raspberry Pi")
    print("=" * 68)
    print("Measurement      : Digital threshold only, not calibrated dB")
    print(f"GPIO input       : BCM GPIO{args.pin} (default physical pin 11)")
    print(f"Logic mode       : {active_mode}")
    print(f"Debounce         : {args.debounce_ms} ms")
    print(f"Summary interval : {args.summary_interval} s")
    print(f"Run mode         : {run_mode}")
    if args.csv:
        print(f"CSV log          : {args.csv}")
    print()
    print("Speak, clap, or adjust the KY-037 potentiometer and watch state changes.")
    print("If quiet/noisy is reversed, rerun with --active-high.")
    print()


def print_summary(state: MonitorState, now: float) -> None:
    print(
        "[SUMMARY] "
        f"elapsed={state.elapsed(now):6.1f}s "
        f"state={state_label(state.current_noisy):5} "
        f"raw={raw_label(state.stable_raw_high):4} "
        f"events={state.noisy_events:3d} "
        f"transitions={state.transitions:3d} "
        f"noisy_time={state.noisy_percent(now):5.1f}% "
        f"samples={state.samples}"
    )


def print_transition(state: MonitorState, now: float) -> None:
    print(
        "[EVENT]   "
        f"elapsed={state.elapsed(now):6.2f}s "
        f"state={state_label(state.current_noisy):5} "
        f"raw={raw_label(state.stable_raw_high):4} "
        f"events={state.noisy_events}"
    )


def monitor_gpio(args: argparse.Namespace) -> None:
    device = create_input_device(args.pin)
    csv_handle = None
    csv_writer = None

    try:
        csv_handle, csv_writer = open_csv_writer(args.csv)
        start = time.monotonic()
        initial_raw_high = read_raw_high(device)
        initial_noisy = interpret_noisy(initial_raw_high, args.active_high)
        state = MonitorState(
            start_time=start,
            last_accounted_time=start,
            stable_raw_high=initial_raw_high,
            current_raw_high=initial_raw_high,
            current_noisy=initial_noisy,
            candidate_raw_high=initial_raw_high,
            candidate_since=start,
        )

        print(
            "[START]   "
            f"state={state_label(state.current_noisy):5} "
            f"raw={raw_label(state.stable_raw_high):4} "
            f"timestamp={local_timestamp()}"
        )

        debounce_seconds = args.debounce_ms / 1000.0
        next_summary = start + args.summary_interval
        end_time = None if args.duration == 0 else start + args.duration

        while True:
            now = time.monotonic()
            if end_time is not None and now >= end_time:
                break

            raw_high = read_raw_high(device)
            account_elapsed_time(state, now)
            changed = process_sample(
                state=state,
                raw_high=raw_high,
                active_high=args.active_high,
                debounce_seconds=debounce_seconds,
                now=now,
            )

            if changed:
                print_transition(state, now)

            write_csv_row(csv_writer, args, state, now)

            if now >= next_summary:
                print_summary(state, now)
                if csv_handle is not None:
                    csv_handle.flush()
                next_summary = now + args.summary_interval

            time.sleep(DEFAULT_SAMPLE_INTERVAL_SECONDS)

        final_now = time.monotonic()
        account_elapsed_time(state, final_now)
        print_summary(state, final_now)
        print("\n[INFO] Monitor finished.")
    except KeyboardInterrupt:
        final_now = time.monotonic()
        print(f"\n[INFO] Monitor stopped at {local_timestamp()}.")
        try:
            print_summary(state, final_now)
        except UnboundLocalError:
            pass
    finally:
        if csv_handle is not None:
            csv_handle.flush()
            csv_handle.close()
        device.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KY-037 digital sound threshold test for Raspberry Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/tests_on_pi/ky037_sound_threshold_test.py --duration 30
  python3 tests/tests_on_pi/ky037_sound_threshold_test.py --pin 17 --active-high
  python3 tests/tests_on_pi/ky037_sound_threshold_test.py --csv ky037_noise_log.csv --duration 300
""",
    )

    parser.add_argument(
        "--pin",
        type=int,
        default=17,
        help="BCM GPIO pin connected to KY-037 DO/D0 (default: 17)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Monitor duration in seconds. Use 0 to run until Ctrl+C (default: 60)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--active-low",
        dest="active_high",
        action="store_false",
        help="Treat raw GPIO LOW as noisy (default)",
    )
    mode_group.add_argument(
        "--active-high",
        dest="active_high",
        action="store_true",
        help="Treat raw GPIO HIGH as noisy",
    )
    parser.set_defaults(active_high=False)
    parser.add_argument(
        "--debounce-ms",
        type=int,
        default=50,
        help="Stable time required before accepting a GPIO state change (default: 50)",
    )
    parser.add_argument(
        "--summary-interval",
        type=int,
        default=5,
        help="Seconds between terminal summary lines (default: 5)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output path for timestamped readings",
    )

    args = parser.parse_args()
    if args.pin < 0:
        parser.error("--pin must be a BCM GPIO number >= 0")
    if args.duration < 0:
        parser.error("--duration must be >= 0")
    if args.debounce_ms < 0:
        parser.error("--debounce-ms must be >= 0")
    if args.summary_interval <= 0:
        parser.error("--summary-interval must be > 0")

    return args


def main() -> int:
    args = parse_args()
    print_intro(args)
    monitor_gpio(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
