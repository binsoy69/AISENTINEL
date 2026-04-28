#!/usr/bin/env python3
"""KY-037 ADS1015 test monitor that outputs estimated dB using saved calibration."""

from __future__ import annotations

import argparse
import sys

from ky037_ads1015_common import (
    DATA_RATE_TO_BITS,
    DEFAULT_CONFIG_PATH,
    FULL_SCALE_TO_PGA_BITS,
    classify_db,
    close_bus,
    open_ads1015,
    print_config,
    print_i2c_error,
    require_calibration,
    resolve_config,
    sample_window,
    validate_settings,
)


def print_result(result: dict[str, float], settings: argparse.Namespace) -> None:
    """Print one compact monitoring line per window."""
    estimated_db = result["estimated_db"]
    status = classify_db(estimated_db, settings.quiet_db, settings.loud_db)

    if settings.debug:
        print(
            f"estimated_db={estimated_db:.1f} status={status} "
            f"rms={result['rms_mv']:.2f}mV p2p={result['p2p_mv']:.2f}mV"
        )
        print(
            f"debug address=0x{settings.address:02X} channel=A{settings.channel} "
            f"mean={result['mean_v']:.4f}V min={result['min_v']:.4f}V "
            f"max={result['max_v']:.4f}V rate={settings.data_rate}SPS "
            f"samples={int(result['sample_count'])}"
        )
        return

    print(f"estimated_db={estimated_db:.1f} status={status}")


def monitor_ads1015(settings: argparse.Namespace) -> None:
    """Configure the ADS1015 and continuously report estimated dB windows."""
    bus = None

    try:
        bus = open_ads1015(settings)
        while True:
            result = sample_window(bus, settings)
            print_result(result, settings)
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
        description="KY-037 ADS1015 test monitor that outputs estimated dB",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="JSON file used to load calibration/settings",
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
        help="Optional quiet reference override",
    )
    parser.add_argument(
        "--ref-loud-rms-mv",
        type=float,
        default=None,
        help="Optional loud reference override",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also print RMS, peak-to-peak, mean/min/max voltage, and sample count",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved config and exit",
    )

    settings = resolve_config(
        parser.parse_args(),
        passthrough_keys=("debug", "show_config"),
    )
    validate_settings(parser, settings)
    return settings


def main() -> int:
    settings = parse_args()

    if settings.show_config:
        print_config(settings)
        return 0

    require_calibration(settings)
    monitor_ads1015(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
