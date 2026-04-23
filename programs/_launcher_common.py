#!/usr/bin/env python3
"""Shared helpers for no-argument AISENTINEL launchers."""

from __future__ import annotations

import configparser
import os
from pathlib import Path
import runpy
import sys
import unittest


PROGRAMS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROGRAMS_DIR.parent
RUNTIME_ROOT = REPO_ROOT / "runtime"
CENTRAL_DASHBOARD_ROOT = RUNTIME_ROOT / "central_dashboard"
FRONT_NODE_RUNTIME_ROOT = RUNTIME_ROOT / "front_node_pi"


def configure_repo_environment() -> None:
    """Make launchers work from IDEs, double-clicks, and arbitrary cwd values."""
    os.chdir(REPO_ROOT)
    for path in (RUNTIME_ROOT, FRONT_NODE_RUNTIME_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def repo_path(relative_path: str | os.PathLike[str]) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path.resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def central_dashboard_config(name: str) -> Path:
    return repo_path(CENTRAL_DASHBOARD_ROOT / name)


def _read_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    loaded = parser.read(path, encoding="utf-8")
    if not loaded:
        raise SystemExit(f"Config file was not found: {path}")
    return parser


def _resolve_config_path(raw_value: str, *, label: str) -> Path:
    value = str(raw_value or "").strip()
    if not value:
        raise SystemExit(f"{label} is not configured.")
    return repo_path(value)


def validate_node_video_config(node_config_path: Path) -> Path:
    """Return the configured video path or exit with a beginner-readable message."""
    parser = _read_ini(node_config_path)
    source_mode = parser.get("capture", "source_mode", fallback="").strip().lower()
    if source_mode != "video":
        raise SystemExit(
            f"{node_config_path} must set [capture] source_mode = video for this launcher."
        )

    node_video = parser.get("capture", "video_path", fallback="").strip()
    if node_video:
        video_path = repo_path(node_video)
    else:
        runtime_config_path = _resolve_config_path(
            parser.get("detector", "runtime_config_path", fallback=""),
            label=f"{node_config_path} [detector] runtime_config_path",
        )
        runtime_parser = _read_ini(runtime_config_path)
        video_value = runtime_parser.get("video_source", "default_video", fallback="").strip()
        if not video_value:
            raise SystemExit(
                "Video mode needs a default video path. Set "
                f"[video_source] default_video in {runtime_config_path}."
            )
        video_path = repo_path(video_value)

    if not video_path.exists():
        raise SystemExit(
            "Configured video file was not found: "
            f"{video_path}\nUpdate [video_source] default_video in the node runtime INI."
        )
    return video_path


def run_script(script_path: Path, *args: str) -> None:
    """Run an existing repository script as if it was launched from the CLI."""
    configure_repo_environment()
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def run_central_dashboard(config_name: str = "central_service.ini") -> None:
    configure_repo_environment()
    from central_dashboard.central_service.app import create_app
    from central_dashboard.central_service.config import load_central_service_config

    config_path = central_dashboard_config(config_name)
    config = load_central_service_config(config_path)
    app = create_app(config)

    print()
    print("=" * 78)
    print("  AISENTINEL - Central Dashboard")
    print(f"  Config    : {config.config_path}")
    print(f"  URL       : http://127.0.0.1:{config.port}")
    print("  Browser   : open the URL above from the dashboard machine")
    print("=" * 78)
    print()

    app.run(host=config.host, port=config.port, threaded=True)


def run_node_agent(config_name: str, *, require_video: bool = False) -> None:
    configure_repo_environment()
    from central_dashboard.node_agent.app import create_app
    from central_dashboard.node_agent.config import load_node_agent_config

    config_path = central_dashboard_config(config_name)
    if require_video:
        video_path = validate_node_video_config(config_path)
        print(f"[INFO] Video source configured: {video_path}")

    config = load_node_agent_config(config_path)
    app = create_app(config)

    print()
    print("=" * 78)
    print(f"  AISENTINEL - {config.display_name} Agent")
    print(f"  Config      : {config.config_path}")
    print(f"  Source mode : {config.source_mode}")
    print(f"  Agent URL   : http://127.0.0.1:{config.port}")
    print(f"  Central URL : {config.central_base_url}")
    print("=" * 78)
    print()

    app.run(host=config.host, port=config.port, threaded=True)


def run_node_webcam_calibration(config_name: str) -> None:
    run_script(
        CENTRAL_DASHBOARD_ROOT / "scripts" / "calibrate_node_webcam.py",
        "--config",
        str(central_dashboard_config(config_name)),
    )


def run_node_video_calibration(config_name: str) -> None:
    config_path = central_dashboard_config(config_name)
    video_path = validate_node_video_config(config_path)
    run_script(
        CENTRAL_DASHBOARD_ROOT / "scripts" / "calibrate_node_video.py",
        "--config",
        str(config_path),
        "--video",
        str(video_path),
    )


def run_central_dashboard_tests() -> int:
    configure_repo_environment()
    suite = unittest.defaultTestLoader.discover(str(CENTRAL_DASHBOARD_ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1

