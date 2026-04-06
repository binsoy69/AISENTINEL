#!/usr/bin/env python3
"""Calibration entrypoint for the front-node webcam runtime."""

from __future__ import annotations

import argparse
import sys

from runtime_config import (
    DEFAULT_WEBCAM_CONFIG_PATH,
    load_runtime_config,
    resolve_cli_path,
)
from runtime_support import (
    apply_behavior_config,
    configure_runtime_paths,
    get_webcam_source_label,
    load_runtime_modules,
    open_webcam_capture,
    read_webcam_frame,
    require_setup_environment,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AISENTINEL front-node webcam calibration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runtime/front_node_pi/calibrate_webcam.py
  python runtime/front_node_pi/calibrate_webcam.py --config runtime/front_node_pi/config_webcam.ini
  python runtime/front_node_pi/calibrate_webcam.py --profile runtime/front_node_pi/data/setup_profiles/front_webcam_all_behavior_setup.json
        """,
    )
    parser.add_argument("--config", default=None, help="Optional path to a webcam runtime INI config file.")
    parser.add_argument("--pose-model", default=None, help="Override the pose HEF model path from config.")
    parser.add_argument("--pose-confidence", type=float, default=None, help="Override the pose/person confidence threshold from config.")
    parser.add_argument("--profile", default=None, help="Optional output JSON path for the saved setup profile.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_runtime_config(args.config, DEFAULT_WEBCAM_CONFIG_PATH)
    modules = load_runtime_modules()
    configure_runtime_paths(modules, config)
    apply_behavior_config(modules, config)

    combined_mod = modules.combined_mod
    setup_io = modules.setup_io
    head_mod = modules.head_mod

    pose_path = resolve_cli_path(args.pose_model) or config.pose_model
    pose_confidence = args.pose_confidence if args.pose_confidence is not None else config.pose_confidence
    source_label = get_webcam_source_label(config)
    profile_path = resolve_cli_path(args.profile) or config.webcam_source.default_setup_profile

    print()
    print("=" * 78)
    print("  AISENTINEL - Front-Node Webcam Calibration")
    print(f"  Config         : {config.config_path}")
    print("  Runtime logic  : runtime/front_node_pi/front_node_all_behavior_pi.py")
    print(f"  Webcam source  : {source_label} (index {config.webcam_source.camera_index})")
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

    cap = None
    pose_estimator = None

    try:
        cap = open_webcam_capture(config, head_mod)
        first_frame = read_webcam_frame(cap)
        if first_frame is None:
            print(f"{head_mod.TC.RED}[ERROR] Cannot read webcam frame.{head_mod.TC.RESET}")
            sys.exit(1)

        width = int(cap.get(3))
        disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
        head_mod.log_info(f"Webcam resolution: {width}x{int(cap.get(4))}")

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
            profile_path = setup_io.default_setup_profile_path(source_label)

        saved_path = setup_io.save_setup_profile(
            profile_path,
            source_label,
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
    print("  Webcam Calibration Saved")
    print("-" * 78)
    print(f"  Source          : {source_label}")
    print(f"  Students        : {len(setup_bundle['student_map'])}")
    print(f"  Desk lines      : {configured_lines}/{len(setup_bundle['student_lines'])}")
    print(f"  ROI             : {'yes' if setup_bundle['roi_polygon'] is not None else 'no'}")
    print(f"  Saved profile   : {saved_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
