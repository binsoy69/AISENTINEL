#!/usr/bin/env python3
"""Calibrate a node KY-037 + ADS1015 sound sensor and update its INI config."""

from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPO_ROOT / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from edge_node_runtime.sound_monitor import (  # noqa: E402
    DATA_RATE_TO_BITS,
    DEFAULT_ALERT_THRESHOLD_DB,
    DEFAULT_CHANNEL,
    DEFAULT_DATA_RATE,
    DEFAULT_FULL_SCALE,
    DEFAULT_I2C_ADDRESS,
    DEFAULT_I2C_BUS,
    DEFAULT_INCIDENT_COOLDOWN_SEC,
    DEFAULT_LOUD_DB,
    DEFAULT_QUIET_DB,
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    FULL_SCALE_TO_PGA_BITS,
    close_bus,
    open_ads1015,
    print_i2c_error,
    sample_window,
    utc_now_iso,
)


PLACEHOLDER_TOKENS = ("CHANGE_ME", "SOUND_CALIBRATION")


@dataclass(slots=True)
class CalibrationSettings:
    config_file: Path
    bus: int
    address: int
    channel: int
    full_scale: float
    data_rate: int
    sample_interval: float
    window_seconds: float
    quiet_db: float
    loud_db: float
    ref_quiet_rms_mv: float | None
    ref_loud_rms_mv: float | None
    alert_threshold_db: float
    incident_cooldown_sec: float
    debug: bool = False


def resolve_cli_path(raw_value: str | None, *, default: Path | None = None) -> Path:
    if raw_value is None or not str(raw_value).strip():
        if default is None:
            raise SystemExit("A path value is required.")
        return default.resolve(strict=False)
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def resolve_repo_path(raw_value: str | Path) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def repo_relative(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_node_ini(config_path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    loaded = parser.read(config_path, encoding="utf-8")
    if not loaded:
        raise SystemExit(f"Node config not found: {config_path}")
    return parser


def write_node_ini(config_path: Path, parser: configparser.ConfigParser) -> None:
    with config_path.open("w", encoding="utf-8") as stream:
        parser.write(stream)


def clean_node_id(raw_value: str) -> str:
    value = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in raw_value.strip().lower()
    ).strip("_")
    return value or "front"


def node_id_from_config(parser: configparser.ConfigParser, config_path: Path) -> str:
    node_id = parser.get("agent", "node_id", fallback="").strip()
    if node_id:
        return clean_node_id(node_id)
    stem = config_path.stem.lower()
    if stem.endswith("_node"):
        stem = stem[: -len("_node")]
    return clean_node_id(stem)


def is_placeholder_path(raw_value: str | None) -> bool:
    value = str(raw_value or "").strip()
    if not value:
        return True
    return any(token in value for token in PLACEHOLDER_TOKENS)


def default_calibration_path(
    parser: configparser.ConfigParser,
    config_path: Path,
) -> Path:
    raw_value = parser.get("sound_sensor", "calibration_config", fallback="").strip()
    if not is_placeholder_path(raw_value):
        return resolve_repo_path(raw_value)

    node_id = node_id_from_config(parser, config_path)
    return (
        REPO_ROOT
        / "runtime"
        / "central_dashboard"
        / "data"
        / f"node_{node_id}"
        / "sound"
        / "ky037_ads1015_config.json"
    ).resolve(strict=False)


def load_calibration_payload(config_file: Path) -> dict:
    if not config_file.exists():
        return {}
    try:
        raw = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read calibration JSON {config_file}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SystemExit(f"Calibration JSON must contain an object: {config_file}")
    return raw


def get_ini_value(
    parser: configparser.ConfigParser,
    option: str,
    fallback,
    *,
    getter_name: str = "get",
):
    if not parser.has_option("sound_sensor", option):
        return fallback
    getter = getattr(parser, getter_name)
    return getter("sound_sensor", option)


def parse_optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def payload_value(payload: dict, key: str, fallback=None):
    value = payload.get(key, fallback)
    if key == "address" and value is not None:
        return int(str(value), 0)
    return value


def resolve_settings(
    args: argparse.Namespace,
    parser: configparser.ConfigParser,
    config_file: Path,
) -> CalibrationSettings:
    saved = load_calibration_payload(config_file)

    quiet_db = float(
        args.quiet_db
        if args.quiet_db is not None
        else payload_value(saved, "quiet_db", DEFAULT_QUIET_DB)
    )
    loud_db = float(
        args.loud_db
        if args.loud_db is not None
        else payload_value(saved, "loud_db", DEFAULT_LOUD_DB)
    )
    alert_threshold_db = float(
        args.alert_threshold_db
        if args.alert_threshold_db is not None
        else get_ini_value(
            parser,
            "alert_threshold_db",
            payload_value(saved, "alert_threshold_db", DEFAULT_ALERT_THRESHOLD_DB),
            getter_name="getfloat",
        )
    )

    return CalibrationSettings(
        config_file=config_file,
        bus=int(
            args.bus
            if args.bus is not None
            else get_ini_value(
                parser,
                "i2c_bus",
                payload_value(saved, "bus", DEFAULT_I2C_BUS),
                getter_name="getint",
            )
        ),
        address=int(
            args.address
            if args.address is not None
            else int(
                str(
                    get_ini_value(
                        parser,
                        "i2c_address",
                        payload_value(saved, "address", DEFAULT_I2C_ADDRESS),
                    )
                ),
                0,
            )
        ),
        channel=int(
            args.channel
            if args.channel is not None
            else get_ini_value(
                parser,
                "adc_channel",
                payload_value(saved, "channel", DEFAULT_CHANNEL),
                getter_name="getint",
            )
        ),
        full_scale=float(
            args.full_scale
            if args.full_scale is not None
            else get_ini_value(
                parser,
                "full_scale",
                payload_value(saved, "full_scale", DEFAULT_FULL_SCALE),
                getter_name="getfloat",
            )
        ),
        data_rate=int(
            args.data_rate
            if args.data_rate is not None
            else get_ini_value(
                parser,
                "data_rate",
                payload_value(saved, "data_rate", DEFAULT_DATA_RATE),
                getter_name="getint",
            )
        ),
        sample_interval=float(
            args.sample_interval
            if args.sample_interval is not None
            else get_ini_value(
                parser,
                "sample_interval",
                payload_value(
                    saved,
                    "sample_interval",
                    DEFAULT_SAMPLE_INTERVAL_SECONDS,
                ),
                getter_name="getfloat",
            )
        ),
        window_seconds=float(
            args.window_seconds
            if args.window_seconds is not None
            else get_ini_value(
                parser,
                "window_seconds",
                payload_value(saved, "window_seconds", DEFAULT_WINDOW_SECONDS),
                getter_name="getfloat",
            )
        ),
        quiet_db=quiet_db,
        loud_db=loud_db,
        ref_quiet_rms_mv=parse_optional_float(
            args.ref_quiet_rms_mv
            if args.ref_quiet_rms_mv is not None
            else payload_value(saved, "ref_quiet_rms_mv")
        ),
        ref_loud_rms_mv=parse_optional_float(
            args.ref_loud_rms_mv
            if args.ref_loud_rms_mv is not None
            else payload_value(saved, "ref_loud_rms_mv")
        ),
        alert_threshold_db=alert_threshold_db,
        incident_cooldown_sec=float(
            args.incident_cooldown_sec
            if args.incident_cooldown_sec is not None
            else get_ini_value(
                parser,
                "incident_cooldown_sec",
                payload_value(
                    saved,
                    "incident_cooldown_sec",
                    DEFAULT_INCIDENT_COOLDOWN_SEC,
                ),
                getter_name="getfloat",
            )
        ),
        debug=bool(args.debug),
    )


def validate_settings(parser: argparse.ArgumentParser, settings: CalibrationSettings) -> None:
    if settings.bus < 0:
        parser.error("--bus must be >= 0")
    if settings.address < 0 or settings.address > 0x7F:
        parser.error("--address must be a valid 7-bit I2C address")
    if settings.channel not in (0, 1, 2, 3):
        parser.error("--channel must be 0, 1, 2, or 3")
    if settings.full_scale not in FULL_SCALE_TO_PGA_BITS:
        parser.error("--full-scale must be one of the ADS1015 supported ranges")
    if settings.data_rate not in DATA_RATE_TO_BITS:
        parser.error("--data-rate must be one of the ADS1015 supported rates")
    if settings.sample_interval <= 0:
        parser.error("--sample-interval must be > 0")
    if settings.window_seconds <= 0:
        parser.error("--window-seconds must be > 0")
    if settings.loud_db <= settings.quiet_db:
        parser.error("--loud-db must be greater than --quiet-db")
    if settings.alert_threshold_db <= 0:
        parser.error("--alert-threshold-db must be > 0")
    if settings.incident_cooldown_sec < 0:
        parser.error("--incident-cooldown-sec must be >= 0")
    if (
        settings.ref_quiet_rms_mv is not None
        and settings.ref_loud_rms_mv is not None
        and settings.ref_loud_rms_mv <= settings.ref_quiet_rms_mv
    ):
        parser.error("--ref-loud-rms-mv must be greater than --ref-quiet-rms-mv")


def make_config_payload(settings: CalibrationSettings) -> dict:
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
        "alert_threshold_db": settings.alert_threshold_db,
        "incident_cooldown_sec": settings.incident_cooldown_sec,
        "updated_at": utc_now_iso(),
    }


def save_config_file(config_file: Path, settings: CalibrationSettings) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(make_config_payload(settings), indent=2) + "\n",
        encoding="utf-8",
    )


def calibration_is_complete(settings: CalibrationSettings) -> bool:
    return settings.ref_quiet_rms_mv is not None and settings.ref_loud_rms_mv is not None


def update_node_ini(
    config_path: Path,
    parser: configparser.ConfigParser,
    settings: CalibrationSettings,
    *,
    enable_complete_calibration: bool,
) -> None:
    if not parser.has_section("sound_sensor"):
        parser.add_section("sound_sensor")

    parser.set("sound_sensor", "calibration_config", repo_relative(settings.config_file))
    if enable_complete_calibration:
        parser.set(
            "sound_sensor",
            "enabled",
            "true" if calibration_is_complete(settings) else "false",
        )

    parser.set("sound_sensor", "alert_threshold_db", f"{settings.alert_threshold_db:.1f}")
    parser.set("sound_sensor", "incident_cooldown_sec", f"{settings.incident_cooldown_sec:.1f}")
    parser.set("sound_sensor", "i2c_bus", str(settings.bus))
    parser.set("sound_sensor", "i2c_address", f"0x{settings.address:02X}")
    parser.set("sound_sensor", "adc_channel", str(settings.channel))
    parser.set("sound_sensor", "full_scale", f"{settings.full_scale:g}")
    parser.set("sound_sensor", "data_rate", str(settings.data_rate))
    parser.set("sound_sensor", "sample_interval", f"{settings.sample_interval:g}")
    parser.set("sound_sensor", "window_seconds", f"{settings.window_seconds:g}")
    write_node_ini(config_path, parser)


def print_capture_result(
    label: str,
    result: dict[str, float],
    settings: CalibrationSettings,
) -> None:
    print(
        f"{label}_reference rms={result['rms_mv']:.2f}mV "
        f"p2p={result['p2p_mv']:.2f}mV"
    )
    if settings.debug:
        print(
            f"debug address=0x{settings.address:02X} channel=A{settings.channel} "
            f"mean={result['mean_v']:.4f}V min={result['min_v']:.4f}V "
            f"max={result['max_v']:.4f}V rate={settings.data_rate}SPS "
            f"samples={int(result['sample_count'])}"
        )


def capture_reference(settings: CalibrationSettings, label: str) -> None:
    bus = None
    try:
        bus = open_ads1015(settings)
        result = sample_window(bus, settings)
    except OSError as exc:
        print_i2c_error(settings, exc)
        raise SystemExit(1) from exc
    finally:
        if bus is not None:
            close_bus(bus)

    if label == "quiet":
        settings.ref_quiet_rms_mv = result["rms_mv"]
    else:
        settings.ref_loud_rms_mv = result["rms_mv"]

    save_config_file(settings.config_file, settings)
    print_capture_result(label, result, settings)
    print(
        f"[INFO] Saved {label} reference to {settings.config_file}: "
        f"{result['rms_mv']:.2f}mV RMS"
    )


def wait_for_enter(prompt: str) -> None:
    try:
        input(prompt)
    except EOFError:
        print("[INFO] No interactive input available. Capturing immediately.")


def run_interactive_calibration(settings: CalibrationSettings) -> None:
    print("KY-037 ADS1015 Calibration")
    print(f"Config file : {settings.config_file}")
    print(f"Quiet target: {settings.quiet_db:.1f} dB")
    print(f"Loud target : {settings.loud_db:.1f} dB")
    print()
    print("Use a phone SPL app or sound level meter beside the sensor.")
    print("The script saves the measured RMS values to JSON and updates the node INI.")
    print()

    wait_for_enter(
        f"Set the room to about {settings.quiet_db:.1f} dB, then press Enter to capture quiet reference..."
    )
    capture_reference(settings, "quiet")
    print()
    wait_for_enter(
        f"Set the room to about {settings.loud_db:.1f} dB, then press Enter to capture loud reference..."
    )
    capture_reference(settings, "loud")


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Calibrate a KY-037 ADS1015 sound sensor and update a node INI.",
    )
    parser.add_argument(
        "--config",
        default="config/front_node.ini",
        help="Node INI to update, for example config/front_node.ini",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help=(
            "Calibration JSON path. Defaults to the INI value, or "
            "runtime/central_dashboard/data/node_<id>/sound/ky037_ads1015_config.json."
        ),
    )
    parser.add_argument("--bus", type=int, default=None, help="I2C bus number")
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 0),
        default=None,
        help="ADS1015 address, for example 0x48",
    )
    parser.add_argument("--channel", type=int, choices=(0, 1, 2, 3), default=None)
    parser.add_argument(
        "--full-scale",
        type=float,
        choices=tuple(FULL_SCALE_TO_PGA_BITS.keys()),
        default=None,
    )
    parser.add_argument(
        "--data-rate",
        type=int,
        choices=tuple(DATA_RATE_TO_BITS.keys()),
        default=None,
    )
    parser.add_argument("--sample-interval", type=float, default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--quiet-db", type=float, default=None)
    parser.add_argument("--loud-db", type=float, default=None)
    parser.add_argument("--alert-threshold-db", type=float, default=None)
    parser.add_argument("--incident-cooldown-sec", type=float, default=None)
    parser.add_argument("--ref-quiet-rms-mv", type=float, default=None)
    parser.add_argument("--ref-loud-rms-mv", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save resolved JSON/settings and update the INI without sampling hardware.",
    )
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument(
        "--no-ini-update",
        action="store_true",
        help="Do not update [sound_sensor] in the node INI.",
    )
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="Do not set [sound_sensor] enabled=true after a complete calibration.",
    )
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument("--capture-quiet", action="store_true")
    capture_group.add_argument("--capture-loud", action="store_true")
    return parser, parser.parse_args()


def main() -> int:
    arg_parser, args = parse_args()
    config_path = resolve_cli_path(args.config)
    node_ini = read_node_ini(config_path)
    config_file = (
        resolve_repo_path(args.config_file)
        if args.config_file
        else default_calibration_path(node_ini, config_path)
    )
    settings = resolve_settings(args, node_ini, config_file)
    validate_settings(arg_parser, settings)

    if args.show_config:
        print(json.dumps(make_config_payload(settings), indent=2))
        print(f"node_ini = {config_path}")
        return 0

    if args.save_config:
        save_config_file(settings.config_file, settings)
    elif args.capture_quiet:
        capture_reference(settings, "quiet")
    elif args.capture_loud:
        capture_reference(settings, "loud")
    else:
        run_interactive_calibration(settings)
        print()
        print("[INFO] Calibration complete.")

    if not args.no_ini_update:
        update_node_ini(
            config_path,
            node_ini,
            settings,
            enable_complete_calibration=not args.no_enable,
        )
        print(f"[INFO] Updated [sound_sensor] calibration_config in {config_path}")
        if calibration_is_complete(settings) and not args.no_enable:
            print("[INFO] Sound monitoring is enabled in the node INI.")
        elif not calibration_is_complete(settings):
            print("[INFO] Calibration is not complete yet; capture both quiet and loud references before enabling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
