"""Configuration loader for the central dashboard service."""

from __future__ import annotations

from dataclasses import dataclass, field
import configparser
import os
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent.parent


def _resolve_path(raw_value: str | None) -> Path:
    value = str(raw_value or "").strip()
    path = Path(value).expanduser() if value else Path()
    if value and not path.is_absolute():
        path = REPO_ROOT / path
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
            )
        ),
        evidence_root=_resolve_path(
            parser.get(
                "service",
                "evidence_root",
                fallback="runtime/central_dashboard/data/central_service/evidence",
            )
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
