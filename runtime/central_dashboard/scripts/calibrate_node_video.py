#!/usr/bin/env python3
"""Calibrate a central-dashboard node against a test-video runtime source."""

from __future__ import annotations

import argparse
import configparser
from dataclasses import replace
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from central_dashboard.node_agent.config import load_node_agent_config
from central_dashboard.node_agent.front_runtime import (
    front_runtime_support,
    load_front_runtime_context,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a central-dashboard node video setup profile.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "node_front.ini"),
        help="Path to the node agent INI file.",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Video path to use for calibration.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional output JSON path for the saved setup profile.",
    )
    return parser


def _resolve_path(raw_value: str | None) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve(strict=False))


def _write_runtime_values(config_path: Path, updates: dict[str, dict[str, str]]) -> None:
    parser = configparser.ConfigParser()
    loaded = parser.read(config_path, encoding="utf-8")
    if not loaded:
        raise FileNotFoundError(f"Runtime config not found: {config_path}")
    for section, values in updates.items():
        if not parser.has_section(section):
            parser.add_section(section)
        for option, value in values.items():
            parser.set(section, option, value)
    with config_path.open("w", encoding="utf-8") as stream:
        parser.write(stream)


def main() -> None:
    args = build_arg_parser().parse_args()

    video_path = _resolve_path(args.video)
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    node_config = load_node_agent_config(args.config)
    video_node_config = replace(
        node_config,
        source_mode="video",
        video_path=video_path,
    )
    runtime_cfg, modules = load_front_runtime_context(video_node_config)
    front_runtime_support.require_setup_environment(modules)

    combined_mod = modules.combined_mod
    setup_io = modules.setup_io
    head_mod = modules.head_mod

    cap = None
    pose_estimator = None

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_path}")
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            raise RuntimeError(f"Cannot read the first frame from: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or first_frame.shape[1])
        disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
        head_mod.log_info(
            f"Video resolution: {first_frame.shape[1]}x{first_frame.shape[0]}"
        )

        pose_estimator = combined_mod.SharedHailoPoseEstimator(
            str(runtime_cfg.pose_model),
            conf_threshold=runtime_cfg.pose_confidence,
        )
        tracker = combined_mod.ReacquiringLockedIoUTracker(
            iou_threshold=runtime_cfg.tracking.iou_threshold,
            max_lost=runtime_cfg.tracking.max_lost,
        )
        setup_bundle = combined_mod.run_manual_setup(
            first_frame,
            pose_estimator,
            tracker,
            disp_scale,
        )
        if setup_bundle is None:
            return

        profile_path = _resolve_path(args.profile)
        if profile_path is None:
            profile_path = (
                runtime_cfg.video_source.default_setup_profile
                or setup_io.default_setup_profile_path(video_path)
            )

        saved_path = setup_io.save_setup_profile(
            profile_path,
            str(video_path),
            first_frame.shape[:2],
            setup_bundle["roi_polygon"],
            setup_bundle["assigned_students"],
            setup_bundle["baseline_yaw_map"],
            setup_bundle["student_lines"],
        )
        _write_runtime_values(
            runtime_cfg.config_path,
            {
                "video_source": {
                    "default_video": _repo_relative(video_path),
                    "default_setup_profile": _repo_relative(saved_path),
                }
            },
        )
    finally:
        if cap is not None:
            cap.release()
        if pose_estimator is not None and hasattr(pose_estimator, "close"):
            pose_estimator.close()

    configured_lines = sum(
        1 for line in setup_bundle["student_lines"] if line is not None
    )
    print()
    print("=" * 78)
    print("  Central Dashboard Node Video Calibration Saved")
    print("-" * 78)
    print(f"  Node            : {node_config.display_name}")
    print(f"  Video           : {video_path.name}")
    print(f"  Students        : {len(setup_bundle['student_map'])}")
    print(f"  Desk lines      : {configured_lines}/{len(setup_bundle['student_lines'])}")
    print(f"  ROI             : {'yes' if setup_bundle['roi_polygon'] is not None else 'no'}")
    print(f"  Saved profile   : {saved_path}")
    print(f"  Runtime config  : {runtime_cfg.config_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
