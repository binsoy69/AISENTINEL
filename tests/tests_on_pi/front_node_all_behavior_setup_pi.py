#!/usr/bin/env python3
"""
One-time setup saver for front_node_all_behavior_pi.py.

This script lets you calibrate:
  - ROI polygon
  - Student assignments
  - Baseline yaw offsets
  - Per-student desk lines

It then saves the result to a JSON file that front_node_all_behavior_pi.py
can load on later runs to skip repeated setup.
"""

import os
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import front_node_all_behavior_pi as combined_mod
import front_node_all_behavior_setup_io as setup_io
import front_node_head_behavior_pi as head_mod
import front_node_passing_papers_pi as pass_mod


POSE_MODEL_PATH = combined_mod.POSE_MODEL_PATH


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AISENTINEL - Save All-Behavior Setup Profile (Pi + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 front_node_all_behavior_setup_pi.py
  python3 front_node_all_behavior_setup_pi.py --video /path/to/video.mp4
  python3 front_node_all_behavior_setup_pi.py --profile /path/to/setup.json
        """,
    )
    parser.add_argument(
        "--pose-model",
        default=str(POSE_MODEL_PATH),
        help=f"Path to pose HEF model (default: {POSE_MODEL_PATH})",
    )
    parser.add_argument(
        "--pose-confidence",
        type=float,
        default=0.5,
        help="Pose/person confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Optional path to the calibration video",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional output JSON path for the saved setup profile",
    )
    args = parser.parse_args()

    print()
    print("=" * 78)
    print("  AISENTINEL - Save All-Behavior Setup Profile")
    print("  Saves         : ROI | assignments | baseline yaw | desk lines")
    print("  Output        : JSON profile for front_node_all_behavior_pi.py")
    print("=" * 78)
    print()

    if not head_mod.HAILO_AVAILABLE:
        print(f"{head_mod.TC.RED}[ERROR] hailo_platform is required.{head_mod.TC.RESET}")
        print("Install: sudo apt install hailo-all")
        sys.exit(1)

    pose_path = Path(args.pose_model)
    if not pose_path.exists():
        print(
            f"{head_mod.TC.RED}[ERROR] Pose HEF model not found: "
            f"{pose_path}{head_mod.TC.RESET}"
        )
        print("See POSE_MODEL_SETUP.md for download instructions.")
        sys.exit(1)

    video_path = args.video
    if not video_path:
        head_mod.log_info("Opening file dialog...")
        video_path = pass_mod.select_video_dialog()
    if not video_path:
        head_mod.log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{head_mod.TC.RED}[ERROR] File not found: {video_path}{head_mod.TC.RESET}")
        sys.exit(1)
    head_mod.log_info(f"Selected: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{head_mod.TC.RED}[ERROR] Cannot open video: {video_path}{head_mod.TC.RESET}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{head_mod.TC.RED}[ERROR] Cannot read first frame.{head_mod.TC.RESET}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    head_mod.log_info(
        f"Video resolution: {width}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )

    pose_estimator = None
    try:
        pose_estimator = combined_mod.SharedHailoPoseEstimator(
            str(pose_path),
            conf_threshold=args.pose_confidence,
        )
        tracker = combined_mod.ReacquiringLockedIoUTracker(
            iou_threshold=0.3, max_lost=60
        )
        setup_bundle = combined_mod.run_manual_setup(
            first_frame,
            pose_estimator,
            tracker,
            disp_scale,
        )
        if setup_bundle is None:
            sys.exit(0)

        profile_path = Path(args.profile) if args.profile else (
            setup_io.default_setup_profile_path(video_path)
        )
        saved_path = setup_io.save_setup_profile(
            profile_path,
            video_path,
            first_frame.shape[:2],
            setup_bundle["roi_polygon"],
            setup_bundle["assigned_students"],
            setup_bundle["baseline_yaw_map"],
            setup_bundle["student_lines"],
        )
    finally:
        cap.release()
        if pose_estimator is not None and hasattr(pose_estimator, "close"):
            pose_estimator.close()

    configured_lines = sum(
        1 for line in setup_bundle["student_lines"] if line is not None
    )
    print()
    print("=" * 78)
    print("  Setup Profile Saved")
    print("-" * 78)
    print(f"  Video           : {Path(video_path).name}")
    print(f"  Students        : {len(setup_bundle['student_map'])}")
    print(f"  Desk lines      : {configured_lines}/{len(setup_bundle['student_lines'])}")
    print(
        f"  ROI             : "
        f"{'yes' if setup_bundle['roi_polygon'] is not None else 'no'}"
    )
    print(f"  Saved profile   : {saved_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
