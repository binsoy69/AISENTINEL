"""Configuration loader for the central dashboard service."""

from __future__ import annotations

from dataclasses import dataclass, field
import configparser
import logging
import os
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent.parent
CONFIG_BASE_ENV_VAR = "AISENTINEL_CONFIG_BASE"
PLACEHOLDER_TOKENS = ("CHANGE_ME", "dev-key", "admin123", "central-dashboard-secret")


def _config_base_dir() -> Path:
    override = str(os.environ.get(CONFIG_BASE_ENV_VAR, "")).strip()
    if override:
        return Path(override).expanduser().resolve(strict=False)
    return REPO_ROOT


def _resolve_path(raw_value: str | None, *, base_dir: Path | None = None) -> Path:
    value = str(raw_value or "").strip()
    path = Path(value).expanduser() if value else Path()
    if value and not path.is_absolute():
        path = (base_dir or _config_base_dir()) / path
    return path.resolve(strict=False)


@dataclass(frozen=True, slots=True)
class BrowserAuthConfig:
    username: str
    password: str
    secret_key: str
    session_ttl_minutes: int


@dataclass(frozen=True, slots=True)
class KnownNodeConfig:
    node_id: str
    display_name: str
    camera_label: str
    api_key: str


@dataclass(frozen=True, slots=True)
class CentralServiceConfig:
    config_path: Path
    host: str
    port: int
    db_path: Path
    evidence_root: Path
    node_offline_after_sec: int
    proxy_timeout_sec: float
    stream_timeout_sec: float
    browser_auth: BrowserAuthConfig
    known_nodes: dict[str, KnownNodeConfig] = field(default_factory=dict)


def load_central_service_config(config_path: str | os.PathLike[str]) -> CentralServiceConfig:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve(strict=False)

    parser = configparser.ConfigParser()
    loaded = parser.read(path, encoding="utf-8")
    if not loaded:
        raise FileNotFoundError(f"Central service config not found: {path}")
    path_base_dir = _config_base_dir()
    _warn_on_placeholder_values(path, parser)

    known_nodes: dict[str, KnownNodeConfig] = {}
    for section in parser.sections():
        if not section.startswith("node:"):
            continue
        node_id = section.split(":", 1)[1].strip()
        known_nodes[node_id] = KnownNodeConfig(
            node_id=node_id,
            display_name=parser.get(section, "display_name", fallback=node_id.title()),
            camera_label=parser.get(section, "camera_label", fallback=node_id.title()),
            api_key=parser.get(section, "api_key", fallback="").strip(),
        )

    browser_auth = BrowserAuthConfig(
        username=parser.get("browser_auth", "username", fallback="admin").strip(),
        password=parser.get("browser_auth", "password", fallback="admin123"),
        secret_key=parser.get(
            "browser_auth",
            "secret_key",
            fallback="central-dashboard-secret",
        ).strip()
        or "central-dashboard-secret",
        session_ttl_minutes=max(
            1,
            parser.getint("browser_auth", "session_ttl_minutes", fallback=480),
        ),
    )

    return CentralServiceConfig(
        config_path=path,
        host=parser.get("service", "host", fallback="127.0.0.1").strip() or "127.0.0.1",
        port=parser.getint("service", "port", fallback=8090),
        db_path=_resolve_path(
            parser.get(
                "service",
                "db_path",
                fallback="runtime/central_dashboard/data/central_service/central.sqlite3",
            ),
            base_dir=path_base_dir,
        ),
        evidence_root=_resolve_path(
            parser.get(
                "service",
                "evidence_root",
                fallback="runtime/central_dashboard/data/central_service/evidence",
            ),
            base_dir=path_base_dir,
        ),
        node_offline_after_sec=max(
            2,
            parser.getint("service", "node_offline_after_sec", fallback=12),
        ),
        proxy_timeout_sec=max(
            1.0,
            parser.getfloat("service", "proxy_timeout_sec", fallback=10.0),
        ),
        stream_timeout_sec=max(
            1.0,
            parser.getfloat("service", "stream_timeout_sec", fallback=12.0),
        ),
        browser_auth=browser_auth,
        known_nodes=known_nodes,
    )


def _warn_on_placeholder_values(config_path: Path, parser: configparser.ConfigParser) -> None:
    logger = logging.getLogger(__name__)
    watched = [
        ("browser_auth", "password"),
        ("browser_auth", "secret_key"),
    ]
    watched.extend((section, "api_key") for section in parser.sections() if section.startswith("node:"))
    for section, option in watched:
        if not parser.has_option(section, option):
            continue
        value = parser.get(section, option, fallback="")
        if any(token in value for token in PLACEHOLDER_TOKENS):
            logger.warning(
                "Config %s still contains placeholder/default value for [%s] %s.",
                config_path,
                section,
                option,
            )
