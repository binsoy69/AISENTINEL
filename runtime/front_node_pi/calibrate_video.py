#!/usr/bin/env python3
"""Calibration entrypoint for the front-node video runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from runtime_config import (
    DEFAULT_VIDEO_CONFIG_PATH,
    load_runtime_config,
    resolve_cli_path,
)
from runtime_support import (
    apply_behavior_config,
    configure_runtime_paths,
    load_runtime_modules,
    require_setup_environment,
    resolve_video_path,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AISENTINEL front-node video calibration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runtime/front_node_pi/calibrate_video.py
  python runtime/front_node_pi/calibrate_video.py --video test-videos/front.mp4
  python runtime/front_node_pi/calibrate_video.py --profile runtime/front_node_pi/data/setup_profiles/front.json
        """,
    )
    parser.add_argument("--config", default=None, help="Optional path to a video runtime INI config file.")
    parser.add_argument("--video", default=None, help="Optional video path. If omitted, the file dialog flow is used.")
    parser.add_argument("--pose-model", default=None, help="Override the pose HEF model path from config.")
    parser.add_argument("--pose-confidence", type=float, default=None, help="Override the pose/person confidence threshold from config.")
    parser.add_argument("--profile", default=None, help="Optional output JSON path for the saved setup profile.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_runtime_config(args.config, DEFAULT_VIDEO_CONFIG_PATH)
    modules = load_runtime_modules()
    configure_runtime_paths(modules, config)
    apply_behavior_config(modules, config)

    combined_mod = modules.combined_mod
    setup_io = modules.setup_io
    head_mod = modules.head_mod
    pass_mod = modules.pass_mod

    pose_path = resolve_cli_path(args.pose_model) or config.pose_model
    pose_confidence = args.pose_confidence if args.pose_confidence is not None else config.pose_confidence
    requested_video = resolve_cli_path(args.video)
    profile_path = resolve_cli_path(args.profile) or config.video_source.default_setup_profile

    print()
    print("=" * 78)
    print("  AISENTINEL - Front-Node Video Calibration")
    print(f"  Config         : {config.config_path}")
    print("  Runtime logic  : runtime/front_node_pi/front_node_all_behavior_pi.py")
    print(f"  Pose model     : {pose_path}")
    print(f"  Setup profiles : {config.setup_profile_dir}")
    print("=" * 78)
    print()

    try:
        require_setup_environment(modules)
    except RuntimeError as exc:
        print(f"{head_mod.TC.RED}[ERROR] {exc}{head_mod.TC.RESET}")
        sys.exit(1)

    if not pose_path.exists():
        print(f"{head_mod.TC.RED}[ERROR] Pose HEF model not found: {pose_path}{head_mod.TC.RESET}")
        print("See POSE_MODEL_SETUP.md for download instructions.")
        sys.exit(1)

    video_path = resolve_video_path(requested_video, config, pass_mod, head_mod)
    if video_path is None:
        head_mod.log_info("No video selected. Exiting.")
        sys.exit(0)
    if not video_path.is_file():
        print(f"{head_mod.TC.RED}[ERROR] File not found: {video_path}{head_mod.TC.RESET}")
        sys.exit(1)
    head_mod.log_info(f"Selected: {video_path}")

    cap = None
    pose_estimator = None

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"{head_mod.TC.RED}[ERROR] Cannot open video: {video_path}{head_mod.TC.RESET}")
            sys.exit(1)

        ret, first_frame = cap.read()
        if not ret:
            print(f"{head_mod.TC.RED}[ERROR] Cannot read first frame.{head_mod.TC.RESET}")
            sys.exit(1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
        head_mod.log_info(f"Video resolution: {width}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        pose_estimator = combined_mod.SharedHailoPoseEstimator(
            str(pose_path),
            conf_threshold=pose_confidence,
        )
        tracker = combined_mod.ReacquiringLockedIoUTracker(
            iou_threshold=config.tracking.iou_threshold,
            max_lost=config.tracking.max_lost,
        )

        setup_bundle = combined_mod.run_manual_setup(
            first_frame,
            pose_estimator,
            tracker,
            disp_scale,
        )
        if setup_bundle is None:
            sys.exit(0)

        if profile_path is None:
            profile_path = setup_io.default_setup_profile_path(video_path)

        saved_path = setup_io.save_setup_profile(
            profile_path,
            str(video_path),
            first_frame.shape[:2],
            setup_bundle["roi_polygon"],
            setup_bundle["assigned_students"],
            setup_bundle["baseline_yaw_map"],
            setup_bundle["student_lines"],
        )
    finally:
        if cap is not None:
            cap.release()
        if pose_estimator is not None and hasattr(pose_estimator, "close"):
            pose_estimator.close()

    configured_lines = sum(1 for line in setup_bundle["student_lines"] if line is not None)
    print()
    print("=" * 78)
    print("  Video Calibration Saved")
    print("-" * 78)
    print(f"  Video           : {Path(video_path).name}")
    print(f"  Students        : {len(setup_bundle['student_map'])}")
    print(f"  Desk lines      : {configured_lines}/{len(setup_bundle['student_lines'])}")
    print(f"  ROI             : {'yes' if setup_bundle['roi_polygon'] is not None else 'no'}")
    print(f"  Saved profile   : {saved_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
