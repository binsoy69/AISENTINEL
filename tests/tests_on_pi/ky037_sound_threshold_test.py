#!/usr/bin/env python3
"""
KY-037 analog noise monitor using an ADS1015 ADC on Raspberry Pi.

This script reads KY-037 AO/A0 through ADS1015 AIN0 over I2C. It continuously
prints a per-window analog signal summary and can optionally map the measured
signal level to paper-style classroom categories after calibration.

By default, the script is uncalibrated and reports:
    status=uncalibrated rms=...mV p2p=...mV

After calibration, it reports:
    estimated_db=... status=normal|warning|loud rms=...mV p2p=...mV

Prerequisites on Raspberry Pi OS:
    sudo apt update
    sudo apt install python3-smbus i2c-tools

Typical usage:
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py --debug
    python3 tests/tests_on_pi/ky037_sound_threshold_test.py --ref-quiet-rms-mv 14 --ref-loud-rms-mv 30
"""

from __future__ import annotations

import argparse
import math
import sys
import time


I2C_INSTALL_COMMAND = "sudo apt install python3-smbus i2c-tools"
DEFAULT_I2C_BUS = 1
DEFAULT_I2C_ADDRESS = 0x48
DEFAULT_CHANNEL = 0
DEFAULT_FULL_SCALE = 4.096
DEFAULT_DATA_RATE = 1600
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.002
DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_QUIET_DB = 45.0
DEFAULT_LOUD_DB = 55.0

REG_CONVERSION = 0x00
REG_CONFIG = 0x01

CHANNEL_TO_MUX_BITS = {
    0: 0b100,
    1: 0b101,
    2: 0b110,
    3: 0b111,
}

FULL_SCALE_TO_PGA_BITS = {
    6.144: 0b000,
    4.096: 0b001,
    2.048: 0b010,
    1.024: 0b011,
    0.512: 0b100,
    0.256: 0b101,
}

DATA_RATE_TO_BITS = {
    128: 0b000,
    250: 0b001,
    490: 0b010,
    920: 0b011,
    1600: 0b100,
    2400: 0b101,
    3300: 0b110,
}


def load_smbus():
    """Import SMBus lazily so --help and py_compile still work off-Pi."""
    try:
        from smbus import SMBus
    except ImportError:
        try:
            from smbus2 import SMBus
        except ImportError:
            print("[ERROR] Neither `smbus` nor `smbus2` is installed.")
            print(f"Install I2C dependencies with: {I2C_INSTALL_COMMAND}")
            sys.exit(1)

    return SMBus


def build_config_word(channel: int, full_scale: float, data_rate: int) -> int:
    """Build an ADS1015 continuous-conversion config word for single-ended reads."""
    return (
        (1 << 15)
        | (CHANNEL_TO_MUX_BITS[channel] << 12)
        | (FULL_SCALE_TO_PGA_BITS[full_scale] << 9)
        | (0 << 8)
        | (DATA_RATE_TO_BITS[data_rate] << 5)
        | 0x0003
    )


def write_config(bus, address: int, config_word: int) -> None:
    """Write the ADS1015 config register in big-endian byte order."""
    bus.write_i2c_block_data(
        address,
        REG_CONFIG,
        [(config_word >> 8) & 0xFF, config_word & 0xFF],
    )


def read_conversion_code(bus, address: int) -> int:
    """Read the ADS1015 conversion register and decode the signed 12-bit value."""
    data = bus.read_i2c_block_data(address, REG_CONVERSION, 2)
    raw = (data[0] << 8) | data[1]
    code = raw >> 4
    if code & 0x800:
        code -= 1 << 12
    return code


def code_to_voltage(code: int, full_scale: float) -> float:
    """Convert a signed ADS1015 code to volts."""
    return (code / 2048.0) * full_scale


def estimate_db(
    rms_mv: float,
    ref_quiet_rms_mv: float,
    ref_loud_rms_mv: float,
    quiet_db: float,
    loud_db: float,
) -> float:
    """Linearly map calibrated RMS millivolts into an estimated dB value."""
    slope = (loud_db - quiet_db) / (ref_loud_rms_mv - ref_quiet_rms_mv)
    return quiet_db + ((rms_mv - ref_quiet_rms_mv) * slope)


def classify_db(estimated_db: float, quiet_db: float, loud_db: float) -> str:
    """Map estimated dB to paper-style classroom categories."""
    if estimated_db < quiet_db:
        return "normal"
    if estimated_db < loud_db:
        return "warning"
    return "loud"


def sample_window(bus, args: argparse.Namespace) -> dict[str, float]:
    """Collect one measurement window and compute signal statistics."""
    samples: list[float] = []
    deadline = time.monotonic() + args.window_seconds

    while time.monotonic() < deadline:
        code = read_conversion_code(bus, args.address)
        samples.append(code_to_voltage(code, args.full_scale))
        time.sleep(args.sample_interval)

    if not samples:
        raise RuntimeError("No ADC samples were collected in the measurement window.")

    mean_v = sum(samples) / len(samples)
    rms_v = math.sqrt(sum((sample - mean_v) ** 2 for sample in samples) / len(samples))
    min_v = min(samples)
    max_v = max(samples)
    p2p_v = max_v - min_v

    result = {
        "mean_v": mean_v,
        "min_v": min_v,
        "max_v": max_v,
        "rms_mv": rms_v * 1000.0,
        "p2p_mv": p2p_v * 1000.0,
        "sample_count": float(len(samples)),
    }

    if args.ref_quiet_rms_mv is not None and args.ref_loud_rms_mv is not None:
        estimated = estimate_db(
            rms_mv=result["rms_mv"],
            ref_quiet_rms_mv=args.ref_quiet_rms_mv,
            ref_loud_rms_mv=args.ref_loud_rms_mv,
            quiet_db=args.quiet_db,
            loud_db=args.loud_db,
        )
        result["estimated_db"] = estimated
    return result


def print_result(result: dict[str, float], args: argparse.Namespace) -> None:
    """Print one compact monitoring line per window."""
    estimated_db = result.get("estimated_db")
    if estimated_db is None:
        print(
            f"status=uncalibrated rms={result['rms_mv']:.2f}mV "
            f"p2p={result['p2p_mv']:.2f}mV"
        )
    else:
        status = classify_db(estimated_db, args.quiet_db, args.loud_db)
        print(
            f"estimated_db={estimated_db:.1f} status={status} "
            f"rms={result['rms_mv']:.2f}mV p2p={result['p2p_mv']:.2f}mV"
        )

    if args.debug:
        print(
            f"debug address=0x{args.address:02X} channel=A{args.channel} "
            f"mean={result['mean_v']:.4f}V min={result['min_v']:.4f}V "
            f"max={result['max_v']:.4f}V rate={args.data_rate}SPS "
            f"samples={int(result['sample_count'])}"
        )


def monitor_ads1015(args: argparse.Namespace) -> None:
    """Configure the ADS1015 and continuously report noise windows."""
    SMBus = load_smbus()
    bus = SMBus(args.bus)

    try:
        config_word = build_config_word(args.channel, args.full_scale, args.data_rate)
        write_config(bus, args.address, config_word)
        time.sleep(max(0.01, 2.0 / args.data_rate))

        while True:
            result = sample_window(bus, args)
            print_result(result, args)
    except KeyboardInterrupt:
        return
    except OSError as exc:
        print(f"[ERROR] Could not communicate with ADS1015 at 0x{args.address:02X}: {exc}")
        print("Check I2C wiring, enable I2C on the Raspberry Pi, and verify the address with `i2cdetect -y 1`.")
        sys.exit(1)
    finally:
        close = getattr(bus, "close", None)
        if callable(close):
            close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KY-037 analog noise monitor using ADS1015 on Raspberry Pi",
    )
    parser.add_argument(
        "--bus",
        type=int,
        default=DEFAULT_I2C_BUS,
        help="I2C bus number for the ADS1015 (default: 1)",
    )
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 0),
        default=DEFAULT_I2C_ADDRESS,
        help="ADS1015 I2C address in decimal or hex, for example 0x48 (default: 0x48)",
    )
    parser.add_argument(
        "--channel",
        type=int,
        choices=(0, 1, 2, 3),
        default=DEFAULT_CHANNEL,
        help="ADS1015 single-ended input channel connected to KY-037 AO (default: 0)",
    )
    parser.add_argument(
        "--full-scale",
        type=float,
        choices=tuple(FULL_SCALE_TO_PGA_BITS.keys()),
        default=DEFAULT_FULL_SCALE,
        help="ADS1015 full-scale voltage range in volts (default: 4.096)",
    )
    parser.add_argument(
        "--data-rate",
        type=int,
        choices=tuple(DATA_RATE_TO_BITS.keys()),
        default=DEFAULT_DATA_RATE,
        help="ADS1015 conversion rate in samples per second (default: 1600)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        help="Seconds between conversion register reads inside each window (default: 0.002)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Measurement window size in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--quiet-db",
        type=float,
        default=DEFAULT_QUIET_DB,
        help="Lower classroom threshold in dB after calibration (default: 45)",
    )
    parser.add_argument(
        "--loud-db",
        type=float,
        default=DEFAULT_LOUD_DB,
        help="Upper classroom threshold in dB after calibration (default: 55)",
    )
    parser.add_argument(
        "--ref-quiet-rms-mv",
        type=float,
        default=None,
        help="Calibrated RMS millivolts that correspond to the quiet-db threshold",
    )
    parser.add_argument(
        "--ref-loud-rms-mv",
        type=float,
        default=None,
        help="Calibrated RMS millivolts that correspond to the loud-db threshold",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also print mean/min/max voltage, ADS1015 address, and sample count",
    )

    args = parser.parse_args()

    if args.bus < 0:
        parser.error("--bus must be >= 0")
    if args.address < 0 or args.address > 0x7F:
        parser.error("--address must be a valid 7-bit I2C address")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be > 0")
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be > 0")
    if args.loud_db <= args.quiet_db:
        parser.error("--loud-db must be greater than --quiet-db")
    if (args.ref_quiet_rms_mv is None) != (args.ref_loud_rms_mv is None):
        parser.error(
            "--ref-quiet-rms-mv and --ref-loud-rms-mv must be provided together"
        )
    if (
        args.ref_quiet_rms_mv is not None
        and args.ref_loud_rms_mv is not None
        and args.ref_loud_rms_mv <= args.ref_quiet_rms_mv
    ):
        parser.error("--ref-loud-rms-mv must be greater than --ref-quiet-rms-mv")

    return args


def main() -> int:
    monitor_ads1015(parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
