#!/usr/bin/env python3
"""Shared ADS1015 helpers for KY-037 Raspberry Pi scripts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
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
DEFAULT_CONFIG_PATH = Path(__file__).with_name("ky037_ads1015_config.json")

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

CONFIG_KEYS = (
    "bus",
    "address",
    "channel",
    "full_scale",
    "data_rate",
    "sample_interval",
    "window_seconds",
    "quiet_db",
    "loud_db",
    "ref_quiet_rms_mv",
    "ref_loud_rms_mv",
)

DEFAULT_CONFIG = {
    "bus": DEFAULT_I2C_BUS,
    "address": DEFAULT_I2C_ADDRESS,
    "channel": DEFAULT_CHANNEL,
    "full_scale": DEFAULT_FULL_SCALE,
    "data_rate": DEFAULT_DATA_RATE,
    "sample_interval": DEFAULT_SAMPLE_INTERVAL_SECONDS,
    "window_seconds": DEFAULT_WINDOW_SECONDS,
    "quiet_db": DEFAULT_QUIET_DB,
    "loud_db": DEFAULT_LOUD_DB,
    "ref_quiet_rms_mv": None,
    "ref_loud_rms_mv": None,
}


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for saved calibration files."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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


def load_config_file(config_path: Path) -> dict:
    """Load a saved JSON calibration/settings file if it exists."""
    if not config_path.exists():
        return {}

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Could not read config file {config_path}: {exc}")
        sys.exit(1)

    config: dict = {}
    for key in CONFIG_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if key == "address" and value is not None:
            config[key] = int(str(value), 0)
        else:
            config[key] = value
    return config


def make_config_payload(settings: argparse.Namespace) -> dict:
    """Convert resolved settings into a JSON payload for persistence."""
    return {
        "bus": settings.bus,
        "address": f"0x{settings.address:02X}",
        "channel": settings.channel,
        "full_scale": settings.full_scale,
        "data_rate": settings.data_rate,
        "sample_interval": settings.sample_interval,
        "window_seconds": settings.window_seconds,
        "quiet_db": settings.quiet_db,
        "loud_db": settings.loud_db,
        "ref_quiet_rms_mv": settings.ref_quiet_rms_mv,
        "ref_loud_rms_mv": settings.ref_loud_rms_mv,
        "updated_at": utc_now_iso(),
    }


def save_config_file(config_path: Path, settings: argparse.Namespace) -> None:
    """Write the current settings and calibration values to a JSON file."""
    payload = make_config_payload(settings)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_config(settings: argparse.Namespace) -> None:
    """Print the resolved config in JSON form for inspection."""
    print(json.dumps(make_config_payload(settings), indent=2))


def resolve_config(
    raw_args: argparse.Namespace,
    passthrough_keys: tuple[str, ...] = (),
) -> argparse.Namespace:
    """Merge CLI overrides with a saved config file and repo defaults."""
    config_path = Path(raw_args.config_file).expanduser()
    saved = load_config_file(config_path)

    resolved = argparse.Namespace(config_file=config_path)
    for key in passthrough_keys:
        setattr(resolved, key, getattr(raw_args, key))

    for key in CONFIG_KEYS:
        cli_value = getattr(raw_args, key, None)
        if cli_value is not None:
            value = cli_value
        elif key in saved:
            value = saved[key]
        else:
            value = DEFAULT_CONFIG[key]
        setattr(resolved, key, value)

    return resolved


def validate_settings(
    parser: argparse.ArgumentParser,
    settings: argparse.Namespace,
) -> None:
    """Validate common ADS1015 runtime settings."""
    if settings.bus < 0:
        parser.error("--bus must be >= 0")
    if settings.address < 0 or settings.address > 0x7F:
        parser.error("--address must be a valid 7-bit I2C address")
    if settings.sample_interval <= 0:
        parser.error("--sample-interval must be > 0")
    if settings.window_seconds <= 0:
        parser.error("--window-seconds must be > 0")
    if settings.loud_db <= settings.quiet_db:
        parser.error("--loud-db must be greater than --quiet-db")
    if (
        settings.ref_quiet_rms_mv is not None
        and settings.ref_loud_rms_mv is not None
        and settings.ref_loud_rms_mv <= settings.ref_quiet_rms_mv
    ):
        parser.error("--ref-loud-rms-mv must be greater than --ref-quiet-rms-mv")


def require_calibration(settings: argparse.Namespace) -> None:
    """Exit if the saved or provided config does not contain both references."""
    if (
        settings.ref_quiet_rms_mv is not None
        and settings.ref_loud_rms_mv is not None
    ):
        return

    print("[ERROR] Calibration is incomplete.")
    print(
        "Run `python3 tests/tests_on_pi/ky037_ads1015_calibrate.py` first to "
        "capture the quiet and loud reference values."
    )
    print(f"Expected config file: {settings.config_file}")
    sys.exit(1)


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


def open_ads1015(settings: argparse.Namespace):
    """Open the I2C bus and configure the ADS1015 for continuous reads."""
    SMBus = load_smbus()
    bus = SMBus(settings.bus)
    config_word = build_config_word(
        settings.channel,
        settings.full_scale,
        settings.data_rate,
    )
    write_config(bus, settings.address, config_word)
    time.sleep(max(0.01, 2.0 / settings.data_rate))
    return bus


def close_bus(bus) -> None:
    """Close an SMBus handle if the implementation exposes close()."""
    close = getattr(bus, "close", None)
    if callable(close):
        close()


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


def sample_window(bus, settings: argparse.Namespace) -> dict[str, float]:
    """Collect one measurement window and compute signal statistics."""
    samples: list[float] = []
    deadline = time.monotonic() + settings.window_seconds

    while time.monotonic() < deadline:
        code = read_conversion_code(bus, settings.address)
        samples.append(code_to_voltage(code, settings.full_scale))
        time.sleep(settings.sample_interval)

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

    if (
        settings.ref_quiet_rms_mv is not None
        and settings.ref_loud_rms_mv is not None
    ):
        result["estimated_db"] = estimate_db(
            rms_mv=result["rms_mv"],
            ref_quiet_rms_mv=settings.ref_quiet_rms_mv,
            ref_loud_rms_mv=settings.ref_loud_rms_mv,
            quiet_db=settings.quiet_db,
            loud_db=settings.loud_db,
        )
    return result


def print_i2c_error(settings: argparse.Namespace, exc: OSError) -> None:
    """Print a consistent ADS1015 communication error."""
    print(f"[ERROR] Could not communicate with ADS1015 at 0x{settings.address:02X}: {exc}")
    print(
        "Check I2C wiring, enable I2C on the Raspberry Pi, and verify the "
        "address with `i2cdetect -y 1`."
    )
