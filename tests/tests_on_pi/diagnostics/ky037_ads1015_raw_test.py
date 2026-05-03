#!/usr/bin/env python3
"""Continuously print raw KY-037 ADS1015 conversion values."""

from __future__ import annotations

import argparse
import sys
import time

from ky037_ads1015_common import (
    DATA_RATE_TO_BITS,
    DEFAULT_CONFIG_PATH,
    FULL_SCALE_TO_PGA_BITS,
    close_bus,
    code_to_voltage,
    open_ads1015,
    print_config,
    print_i2c_error,
    read_conversion_code,
    resolve_config,
    validate_settings,
)


DEFAULT_READ_INTERVAL_SECONDS = 0.1


def format_raw_sample(sample_number: int, code: int, settings: argparse.Namespace) -> str:
    """Format one raw ADS1015 reading for terminal monitoring."""
    voltage = code_to_voltage(code, settings.full_scale)
    return (
        f"sample={sample_number} raw_code={code} "
        f"voltage={voltage:.4f}V voltage_mv={voltage * 1000.0:.2f}"
    )


def validate_raw_settings(
    parser: argparse.ArgumentParser,
    settings: argparse.Namespace,
) -> None:
    """Validate ADS1015 settings used by the raw monitor."""
    validate_settings(parser, settings)
    if settings.channel not in (0, 1, 2, 3):
        parser.error("--channel must be 0, 1, 2, or 3")
    if settings.full_scale not in FULL_SCALE_TO_PGA_BITS:
        parser.error("--full-scale must be one of the ADS1015 supported ranges")
    if settings.data_rate not in DATA_RATE_TO_BITS:
        parser.error("--data-rate must be one of the ADS1015 supported rates")
    if settings.read_interval <= 0:
        parser.error("--read-interval must be > 0")
    if settings.count < 0:
        parser.error("--count must be >= 0")


def monitor_ads1015_raw(settings: argparse.Namespace) -> None:
    """Configure the ADS1015 and continuously report raw conversion values."""
    bus = None
    sample_number = 0

    try:
        bus = open_ads1015(settings)
        print("KY-037 ADS1015 Raw Monitor")
        print(
            f"address=0x{settings.address:02X} channel=A{settings.channel} "
            f"full_scale=+/-{settings.full_scale:g}V rate={settings.data_rate}SPS "
            f"read_interval={settings.read_interval:g}s"
        )
        print("Press Ctrl+C to stop.")

        while settings.count == 0 or sample_number < settings.count:
            sample_number += 1
            code = read_conversion_code(bus, settings.address)
            print(format_raw_sample(sample_number, code, settings), flush=True)
            time.sleep(settings.read_interval)
    except KeyboardInterrupt:
        return
    except OSError as exc:
        print_i2c_error(settings, exc)
        sys.exit(1)
    finally:
        if bus is not None:
            close_bus(bus)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "KY-037 ADS1015 raw monitor. This does not require sound calibration; "
            "it continuously prints raw ADC codes and converted voltage."
        ),
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Optional JSON file used to load ADS1015 settings if it exists",
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
        "--read-interval",
        type=float,
        default=DEFAULT_READ_INTERVAL_SECONDS,
        help="Seconds between printed raw reads",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of reads before exit. The default 0 runs until Ctrl+C.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved ADS1015 config and exit",
    )

    settings = resolve_config(
        parser.parse_args(),
        passthrough_keys=("read_interval", "count", "show_config"),
    )
    validate_raw_settings(parser, settings)
    return settings


def main() -> int:
    settings = parse_args()

    if settings.show_config:
        print_config(settings)
        print(f"read_interval = {settings.read_interval:g}")
        print(f"count = {settings.count}")
        return 0

    monitor_ads1015_raw(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
