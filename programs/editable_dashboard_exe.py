#!/usr/bin/env python3
"""PyInstaller entrypoint for the Windows editable demo dashboard EXE."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _app_dir() -> Path:
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return _repo_root()


def _configure_paths() -> None:
    if _is_frozen():
        return
    runtime_root = _repo_root() / "runtime"
    path_text = str(runtime_root)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _default_config_path(app_dir: Path) -> Path:
    if _is_frozen():
        return app_dir / "editable_dashboard.ini"
    return app_dir / "config" / "editable_dashboard.ini.example"


def main() -> None:
    app_dir = _app_dir()
    _configure_paths()

    config_path = Path(
        os.environ.get("AISENTINEL_EDITABLE_CONFIG", str(_default_config_path(app_dir)))
    ).expanduser()
    if not config_path.is_absolute():
        config_path = app_dir / config_path
    config_path = config_path.resolve(strict=False)

    if _is_frozen():
        os.environ.setdefault("AISENTINEL_CONFIG_BASE", str(app_dir))

    from central_dashboard.central_service.config import load_central_service_config
    from editable_dashboard.editable_service.app import create_app

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    config = load_central_service_config(config_path)
    app = create_app(config)

    print()
    print("=" * 78)
    print("  AISENTINEL - Editable Dashboard EXE")
    print(f"  Config    : {config.config_path}")
    print(f"  Data root : {config.db_path.parent}")
    print(f"  URL       : http://127.0.0.1:{config.port}")
    print("=" * 78)
    print()

    app.run(host=config.host, port=config.port, threaded=True)


if __name__ == "__main__":
    main()
