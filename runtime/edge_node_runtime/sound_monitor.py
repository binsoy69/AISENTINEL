#!/usr/bin/env python3
"""KY-037 + ADS1015 runtime helpers for AISENTINEL front-node sound monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Callable


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
DEFAULT_ALERT_THRESHOLD_DB = DEFAULT_LOUD_DB
DEFAULT_INCIDENT_COOLDOWN_SEC = 10.0

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

DEFAULT_SOUND_SNAPSHOT = {
    "enabled": False,
    "current_db": None,
    "threshold_db": None,
    "over_threshold": False,
    "status": "disabled",
    "updated_at": "",
    "last_error": "",
}


def utc_now_iso() -> str:
    """Return a stable UTC timestamp."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def display_clock_now() -> str:
    """Return a human-readable local clock string for dashboards/incidents."""
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def load_smbus():
    """Import SMBus lazily so the module remains importable off-device."""
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


@dataclass(frozen=True, slots=True)
class CalibrationReference:
    quiet_db: float
    loud_db: float
    ref_quiet_rms_mv: float
    ref_loud_rms_mv: float
    source_path: Path


@dataclass(frozen=True, slots=True)
class ADS1015Settings:
    bus: int
    address: int
    channel: int
    full_scale: float
    data_rate: int
    sample_interval: float
    window_seconds: float
    quiet_db: float
    loud_db: float
    ref_quiet_rms_mv: float
    ref_loud_rms_mv: float
    alert_threshold_db: float
    incident_cooldown_sec: float


@dataclass(frozen=True, slots=True)
class SoundTelemetry:
    enabled: bool
    current_db: float | None
    threshold_db: float | None
    over_threshold: bool
    status: str
    updated_at: str
    last_error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ThresholdCrossingGate:
    """Emit one incident per below->above threshold transition with cooldown."""

    is_above_threshold: bool = False
    last_incident_monotonic: float = float("-inf")

    def should_emit(
        self,
        estimated_db: float,
        threshold_db: float,
        *,
        now_monotonic: float,
        cooldown_sec: float,
    ) -> bool:
        over_threshold = estimated_db >= threshold_db
        if not over_threshold:
            self.is_above_threshold = False
            return False

        crossing = not self.is_above_threshold
        self.is_above_threshold = True
        if not crossing:
            return False
        if (now_monotonic - self.last_incident_monotonic) < cooldown_sec:
            return False

        self.last_incident_monotonic = now_monotonic
        return True


def load_calibration_reference(config_path: Path | str) -> CalibrationReference:
    """Load quiet/loud calibration references from the saved JSON file."""
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(f"Sound calibration file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read sound calibration file {path}: {exc}") from exc

    required = ("quiet_db", "loud_db", "ref_quiet_rms_mv", "ref_loud_rms_mv")
    missing = [key for key in required if raw.get(key) is None]
    if missing:
        raise ValueError(
            f"Sound calibration file {path} is missing required values: {', '.join(missing)}"
        )

    quiet_db = float(raw["quiet_db"])
    loud_db = float(raw["loud_db"])
    ref_quiet = float(raw["ref_quiet_rms_mv"])
    ref_loud = float(raw["ref_loud_rms_mv"])
    if loud_db <= quiet_db:
        raise ValueError("Sound calibration loud_db must be greater than quiet_db.")
    if ref_loud <= ref_quiet:
        raise ValueError(
            "Sound calibration ref_loud_rms_mv must be greater than ref_quiet_rms_mv."
        )

    return CalibrationReference(
        quiet_db=quiet_db,
        loud_db=loud_db,
        ref_quiet_rms_mv=ref_quiet,
        ref_loud_rms_mv=ref_loud,
        source_path=path,
    )


def build_settings_from_sound_config(sound_config) -> ADS1015Settings:
    """Convert runtime sound config plus calibration JSON into ADS1015 settings."""
    if sound_config.calibration_config is None:
        raise ValueError(
            "sound_sensor.calibration_config is required when sound monitoring is enabled."
        )

    calibration = load_calibration_reference(sound_config.calibration_config)
    alert_threshold_db = float(sound_config.alert_threshold_db)
    if alert_threshold_db <= 0:
        raise ValueError("sound_sensor.alert_threshold_db must be greater than 0.")
    if sound_config.incident_cooldown_sec < 0:
        raise ValueError(
            "sound_sensor.incident_cooldown_sec must be greater than or equal to 0."
        )

    return ADS1015Settings(
        bus=int(sound_config.i2c_bus),
        address=int(sound_config.i2c_address),
        channel=int(sound_config.adc_channel),
        full_scale=float(sound_config.full_scale),
        data_rate=int(sound_config.data_rate),
        sample_interval=float(sound_config.sample_interval),
        window_seconds=float(sound_config.window_seconds),
        quiet_db=calibration.quiet_db,
        loud_db=calibration.loud_db,
        ref_quiet_rms_mv=calibration.ref_quiet_rms_mv,
        ref_loud_rms_mv=calibration.ref_loud_rms_mv,
        alert_threshold_db=alert_threshold_db,
        incident_cooldown_sec=float(sound_config.incident_cooldown_sec),
    )


def build_noise_summary(estimated_db: float, threshold_db: float) -> str:
    """Return the canonical summary text for saved noise incidents."""
    return (
        f"Estimated noise {estimated_db:.1f} dB exceeded "
        f"{threshold_db:.1f} dB threshold."
    )


def build_config_word(channel: int, full_scale: float, data_rate: int) -> int:
    """Build the ADS1015 config word for single-ended reads."""
    return (
        (1 << 15)
        | (CHANNEL_TO_MUX_BITS[channel] << 12)
        | (FULL_SCALE_TO_PGA_BITS[full_scale] << 9)
        | (0 << 8)
        | (DATA_RATE_TO_BITS[data_rate] << 5)
        | 0x0003
    )


def write_config(bus, address: int, config_word: int) -> None:
    """Write the ADS1015 config register."""
    bus.write_i2c_block_data(
        address,
        REG_CONFIG,
        [(config_word >> 8) & 0xFF, config_word & 0xFF],
    )


def open_ads1015(settings):
    """Open the I2C bus and configure the ADS1015 for continuous reads."""
    SMBus = load_smbus()
    bus = SMBus(settings.bus)
    try:
        config_word = build_config_word(
            settings.channel,
            settings.full_scale,
            settings.data_rate,
        )
        write_config(bus, settings.address, config_word)
        time.sleep(max(0.01, 2.0 / settings.data_rate))
        return bus
    except Exception:
        close_bus(bus)
        raise


def close_bus(bus) -> None:
    """Close an SMBus handle if supported."""
    close = getattr(bus, "close", None)
    if callable(close):
        close()


def read_conversion_code(bus, address: int) -> int:
    """Read and decode the signed 12-bit ADS1015 conversion code."""
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
    """Map estimated dB to classroom categories."""
    if estimated_db < quiet_db:
        return "normal"
    if estimated_db < loud_db:
        return "warning"
    return "loud"


def sample_window(bus, settings) -> dict[str, float]:
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
    rms_v = math.sqrt(
        sum((sample - mean_v) ** 2 for sample in samples) / len(samples)
    )
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

    ref_quiet = getattr(settings, "ref_quiet_rms_mv", None)
    ref_loud = getattr(settings, "ref_loud_rms_mv", None)
    if (
        ref_quiet is not None
        and ref_loud is not None
        and float(ref_loud) > float(ref_quiet)
    ):
        result["estimated_db"] = estimate_db(
            rms_mv=result["rms_mv"],
            ref_quiet_rms_mv=float(ref_quiet),
            ref_loud_rms_mv=float(ref_loud),
            quiet_db=settings.quiet_db,
            loud_db=settings.loud_db,
        )
    return result


def print_i2c_error(settings, exc: OSError) -> None:
    """Print a consistent ADS1015 communication error."""
    print(f"[ERROR] Could not communicate with ADS1015 at 0x{settings.address:02X}: {exc}")
    print(
        "Check I2C wiring, enable I2C on the Raspberry Pi, and verify the "
        f"address with `i2cdetect -y {settings.bus}`."
    )
    print(
        "ADS1015 addresses are usually 0x48, 0x49, 0x4A, or 0x4B depending "
        "on the ADDR pin."
    )


class SoundMonitorService:
    """Session-scoped background sampler for the KY-037 + ADS1015."""

    def __init__(
        self,
        sound_config,
        *,
        on_telemetry: Callable[[dict], None] | None = None,
        on_threshold_cross: Callable[[dict], None] | None = None,
        log_fn: Callable[[str], None] | None = None,
        open_bus_fn: Callable[[ADS1015Settings], object] = open_ads1015,
        sample_window_fn: Callable[[object, ADS1015Settings], dict] = sample_window,
        close_bus_fn: Callable[[object], None] = close_bus,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sound_config = sound_config
        self.on_telemetry = on_telemetry
        self.on_threshold_cross = on_threshold_cross
        self.log_fn = log_fn or (lambda _message: None)
        self.open_bus_fn = open_bus_fn
        self.sample_window_fn = sample_window_fn
        self.close_bus_fn = close_bus_fn
        self.monotonic_fn = monotonic_fn

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gate = ThresholdCrossingGate()
        self._snapshot = dict(DEFAULT_SOUND_SNAPSHOT)
        self._snapshot["enabled"] = bool(getattr(sound_config, "enabled", False))
        if self._snapshot["enabled"]:
            self._snapshot["status"] = "idle"
            self._snapshot["threshold_db"] = float(
                getattr(sound_config, "alert_threshold_db", DEFAULT_ALERT_THRESHOLD_DB)
            )

    def start(self) -> bool:
        """Start the background sampler if sound monitoring is enabled."""
        if not getattr(self.sound_config, "enabled", False):
            self._update_snapshot(
                status="disabled",
                threshold_db=float(
                    getattr(
                        self.sound_config,
                        "alert_threshold_db",
                        DEFAULT_ALERT_THRESHOLD_DB,
                    )
                ),
            )
            return False

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            self._update_snapshot_locked(
                status="starting",
                last_error="",
                threshold_db=float(self.sound_config.alert_threshold_db),
            )
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="sound-monitor",
            )
            self._thread.start()
        self._emit_telemetry()
        return True

    def stop(self) -> None:
        """Stop the background sampler and publish an idle telemetry state."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            self._thread = None
            if getattr(self.sound_config, "enabled", False):
                self._update_snapshot_locked(
                    status="idle",
                    over_threshold=False,
                    last_error=self._snapshot.get("last_error", ""),
                )
            else:
                self._update_snapshot_locked(status="disabled", over_threshold=False)
        self._emit_telemetry()

    def snapshot(self) -> dict:
        """Return the latest sound telemetry snapshot."""
        with self._lock:
            return dict(self._snapshot)

    def _run(self) -> None:
        bus = None
        try:
            settings = build_settings_from_sound_config(self.sound_config)
            bus = self.open_bus_fn(settings)
            self._update_snapshot(
                status="monitoring",
                threshold_db=settings.alert_threshold_db,
                last_error="",
            )
            while not self._stop.is_set():
                result = self.sample_window_fn(bus, settings)
                estimated_db = float(result["estimated_db"])
                over_threshold = estimated_db >= settings.alert_threshold_db
                self._update_snapshot(
                    current_db=round(estimated_db, 2),
                    threshold_db=settings.alert_threshold_db,
                    over_threshold=over_threshold,
                    status="alert" if over_threshold else "monitoring",
                    last_error="",
                )
                payload = {
                    "estimated_db": round(estimated_db, 2),
                    "threshold_db": settings.alert_threshold_db,
                    "over_threshold": over_threshold,
                    "updated_at": self.snapshot().get("updated_at", utc_now_iso()),
                    "rms_mv": round(float(result["rms_mv"]), 2),
                    "p2p_mv": round(float(result["p2p_mv"]), 2),
                }
                if self._gate.should_emit(
                    estimated_db,
                    settings.alert_threshold_db,
                    now_monotonic=self.monotonic_fn(),
                    cooldown_sec=settings.incident_cooldown_sec,
                ):
                    self._emit_threshold_cross(payload)
        except Exception as exc:  # pragma: no cover - runtime safety
            self.log_fn(f"Sound monitor error: {exc}")
            self._update_snapshot(
                status="error",
                over_threshold=False,
                last_error=str(exc),
            )
        finally:
            if bus is not None:
                self.close_bus_fn(bus)

    def _emit_threshold_cross(self, payload: dict) -> None:
        if self.on_threshold_cross is None:
            return
        try:
            self.on_threshold_cross(dict(payload))
        except Exception as exc:  # pragma: no cover - runtime safety
            self.log_fn(f"Sound threshold callback error: {exc}")

    def _emit_telemetry(self) -> None:
        if self.on_telemetry is None:
            return
        try:
            self.on_telemetry(self.snapshot())
        except Exception as exc:  # pragma: no cover - runtime safety
            self.log_fn(f"Sound telemetry callback error: {exc}")

    def _update_snapshot(self, **updates) -> None:
        with self._lock:
            self._update_snapshot_locked(**updates)
        self._emit_telemetry()

    def _update_snapshot_locked(self, **updates) -> None:
        self._snapshot.update(updates)
        self._snapshot["updated_at"] = utc_now_iso()
