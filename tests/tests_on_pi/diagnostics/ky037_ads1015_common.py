#!/usr/bin/env python3
"""Shared CLI/config helpers for the KY-037 Raspberry Pi scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPO_ROOT / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from edge_node_runtime.sound_monitor import (  # noqa: E402
    CHANNEL_TO_MUX_BITS,
    DATA_RATE_TO_BITS,
    DEFAULT_CHANNEL,
    DEFAULT_DATA_RATE,
    DEFAULT_FULL_SCALE,
    DEFAULT_I2C_ADDRESS,
    DEFAULT_I2C_BUS,
    DEFAULT_LOUD_DB,
    DEFAULT_QUIET_DB,
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    FULL_SCALE_TO_PGA_BITS,
    I2C_INSTALL_COMMAND,
    build_config_word,
    classify_db,
    close_bus,
    code_to_voltage,
    estimate_db,
    load_smbus,
    open_ads1015,
    print_i2c_error,
    read_conversion_code,
    sample_window,
    utc_now_iso,
    write_config,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("ky037_ads1015_config.json")

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
        "Run `python3 tests/tests_on_pi/diagnostics/ky037_ads1015_calibrate.py` first to "
        "capture the quiet and loud reference values."
    )
    print(f"Expected config file: {settings.config_file}")
    sys.exit(1)
