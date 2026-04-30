#!/usr/bin/env python3
"""Shared helpers for no-argument AISENTINEL launchers."""

from __future__ import annotations

import configparser
from dataclasses import replace
import logging
import os
from pathlib import Path
import runpy
import sys
import unittest


PROGRAMS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROGRAMS_DIR.parent
RUNTIME_ROOT = REPO_ROOT / "runtime"
CENTRAL_DASHBOARD_ROOT = RUNTIME_ROOT / "central_dashboard"
CONFIG_ROOT = REPO_ROOT / "config"
VIDEO_FILE_TYPES = (
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"),
    ("MP4 files", "*.mp4"),
    ("AVI files", "*.avi"),
    ("MOV files", "*.mov"),
    ("MKV files", "*.mkv"),
    ("All files", "*.*"),
)
PLACEHOLDER_TOKENS = ("CHANGE_ME", "SOUND_CALIBRATION")


def configure_runtime_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )


def configure_repo_environment() -> None:
    """Make launchers work from IDEs, double-clicks, and arbitrary cwd values."""
    os.chdir(REPO_ROOT)
    for path in (RUNTIME_ROOT,):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def repo_path(relative_path: str | os.PathLike[str]) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path.resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def central_dashboard_config(name: str) -> Path:
    return repo_path(CONFIG_ROOT / name)


def _missing_config_message(path: Path) -> str:
    example = path.with_suffix(path.suffix + ".example")
    if example.exists():
        return (
            f"Required local config was not found: {path}\n"
            f"Copy {example} to {path}, then edit placeholders for this machine."
        )
    return f"Required local config was not found: {path}"


def require_operator_config(config_name: str) -> Path:
    path = central_dashboard_config(config_name)
    if not path.exists():
        raise SystemExit(_missing_config_message(path))
    return path


def _read_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    loaded = parser.read(path, encoding="utf-8")
    if not loaded:
        raise SystemExit(_missing_config_message(path))
    return parser


def _resolve_config_path(raw_value: str, *, label: str) -> Path:
    value = str(raw_value or "").strip()
    if not value:
        raise SystemExit(f"{label} is not configured.")
    return repo_path(value)


def _is_placeholder_value(raw_value: str) -> bool:
    return any(token in raw_value for token in PLACEHOLDER_TOKENS)


def _repo_relative(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _write_ini(path: Path, parser: configparser.ConfigParser) -> None:
    with path.open("w", encoding="utf-8") as stream:
        parser.write(stream)


def _configured_node_video_path(node_config_path: Path) -> Path | None:
    parser = _read_ini(node_config_path)
    node_video = parser.get("capture", "video_path", fallback="").strip()
    if node_video:
        return repo_path(node_video)

    video_value = parser.get("video_source", "default_video", fallback="").strip()
    if not video_value:
        return None
    return repo_path(video_value)


def validate_node_video_config(node_config_path: Path) -> Path:
    """Return the configured video path or exit with a beginner-readable message."""
    parser = _read_ini(node_config_path)

    node_video = parser.get("capture", "video_path", fallback="").strip()
    if node_video:
        video_path = repo_path(node_video)
    else:
        video_value = parser.get("video_source", "default_video", fallback="").strip()
        if not video_value:
            raise SystemExit(
                "Video mode needs a default video path. Set "
                f"[video_source] default_video in {node_config_path}."
            )
        video_path = repo_path(video_value)

    if not video_path.exists():
        raise SystemExit(
            "Configured video file was not found: "
            f"{video_path}\nUpdate [video_source] default_video in {node_config_path}."
        )
    return video_path


def select_video_file(
    *,
    title: str,
    initial_path: Path | None = None,
    picker=None,
) -> Path:
    """Open a GUI video picker and return the selected existing file."""
    initialdir = ""
    if initial_path is not None and initial_path.exists():
        initialdir = str(initial_path.parent)

    if picker is None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.update()
        except Exception as exc:
            raise SystemExit(
                "Video file picker is unavailable. Run this launcher from a graphical "
                f"desktop session with tkinter installed. Details: {exc}"
            ) from exc

        try:
            selected = filedialog.askopenfilename(
                title=title,
                initialdir=initialdir,
                filetypes=VIDEO_FILE_TYPES,
            )
        finally:
            root.destroy()
    else:
        selected = picker(
            title=title,
            initialdir=initialdir,
            filetypes=VIDEO_FILE_TYPES,
        )

    if not selected:
        raise SystemExit("Video selection cancelled.")

    video_path = repo_path(str(selected))
    if not video_path.exists():
        raise SystemExit(f"Selected video file was not found: {video_path}")
    if not video_path.is_file():
        raise SystemExit(f"Selected video path is not a file: {video_path}")
    return video_path


def select_node_video_file(config_name: str, *, title: str, picker=None) -> tuple[Path, Path]:
    config_path = central_dashboard_config(config_name)
    initial_path = _configured_node_video_path(config_path)
    selected_path = select_video_file(
        title=title,
        initial_path=initial_path if initial_path and initial_path.exists() else None,
        picker=picker,
    )
    return config_path, selected_path


def save_node_video_default(node_config_path: Path, video_path: Path) -> None:
    """Persist the selected video in the unified node INI for replay launchers."""
    parser = _read_ini(node_config_path)
    if not parser.has_section("video_source"):
        parser.add_section("video_source")

    previous_video = parser.get("video_source", "default_video", fallback="").strip()
    previous_path = repo_path(previous_video) if previous_video else None
    same_video = (
        previous_path is not None
        and previous_path.resolve(strict=False) == video_path.resolve(strict=False)
    )

    parser.set("video_source", "default_video", _repo_relative(video_path))
    if not same_video:
        parser.set("video_source", "default_setup_profile", "")
    if parser.has_section("capture") and parser.get("capture", "video_path", fallback="").strip():
        parser.set("capture", "video_path", "")
    _write_ini(node_config_path, parser)



def run_script(script_path: Path, *args: str) -> None:
    """Run an existing repository script as if it was launched from the CLI."""
    configure_repo_environment()
    old_argv = sys.argv[:]
    old_sys_path = sys.path[:]
    script_dir = str(script_path.resolve().parent)
    try:
        sys.path.insert(0, script_dir)
        sys.argv = [str(script_path), *args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path[:] = old_sys_path


def run_central_dashboard(config_name: str = "central.ini") -> None:
    configure_repo_environment()
    configure_runtime_logging()
    from central_dashboard.central_service.app import create_app
    from central_dashboard.central_service.config import load_central_service_config

    config_path = require_operator_config(config_name)
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


def run_node_agent(
    config_name: str,
    *,
    require_video: bool = False,
    choose_video: bool = False,
    source_mode: str | None = None,
) -> None:
    configure_repo_environment()
    configure_runtime_logging()
    from central_dashboard.node_agent.app import create_app
    from central_dashboard.node_agent.config import load_node_agent_config

    config_path = require_operator_config(config_name)
    if choose_video:
        config_path, video_path = select_node_video_file(
            config_name,
            title="Select video for AISENTINEL node replay",
        )
        save_node_video_default(config_path, video_path)
        print(f"[INFO] Video source selected: {video_path}")
    elif require_video:
        video_path = validate_node_video_config(config_path)
        print(f"[INFO] Video source configured: {video_path}")

    config = load_node_agent_config(config_path)
    if source_mode:
        config = replace(config, source_mode=source_mode)
    elif require_video or choose_video:
        config = replace(config, source_mode="video")
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
    config_path = require_operator_config(config_name)
    run_script(
        CENTRAL_DASHBOARD_ROOT / "scripts" / "calibrate_node_webcam.py",
        "--config",
        str(config_path),
    )


def run_node_video_calibration(config_name: str) -> None:
    config_path, video_path = select_node_video_file(
        config_name,
        title="Select video for AISENTINEL node calibration",
    )
    run_script(
        CENTRAL_DASHBOARD_ROOT / "scripts" / "calibrate_node_video.py",
        "--config",
        str(config_path),
        "--video",
        str(video_path),
    )


def run_node_sound_calibration(config_name: str) -> None:
    config_path = require_operator_config(config_name)
    run_script(
        CENTRAL_DASHBOARD_ROOT / "scripts" / "calibrate_sound_sensor.py",
        "--config",
        str(config_path),
    )


def run_sound_sensor_test(config_name: str = "front_node.ini") -> None:
    config_path = require_operator_config(config_name)
    parser = _read_ini(config_path)
    calibration_value = parser.get("sound_sensor", "calibration_config", fallback="").strip()
    calibration_launcher = (
        "calibrate_mid_sound_sensor.py"
        if "mid" in config_name.lower()
        else "calibrate_front_sound_sensor.py"
    )
    if not calibration_value or _is_placeholder_value(calibration_value):
        raise SystemExit(
            "Sound calibration is not configured. Run "
            f"`python programs/{calibration_launcher}` first, or update "
            f"[sound_sensor] calibration_config in {config_path}."
        )
    calibration_path = repo_path(calibration_value)
    if not calibration_path.exists():
        raise SystemExit(
            f"Sound calibration JSON was not found: {calibration_path}\n"
            f"Run `python programs/{calibration_launcher}` to create it."
        )

    run_script(
        REPO_ROOT / "tests" / "tests_on_pi" / "diagnostics" / "ky037_sound_threshold_test.py",
        "--config-file",
        str(calibration_path),
    )


def run_central_dashboard_tests() -> int:
    configure_repo_environment()
    suite = unittest.defaultTestLoader.discover(str(CENTRAL_DASHBOARD_ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1
