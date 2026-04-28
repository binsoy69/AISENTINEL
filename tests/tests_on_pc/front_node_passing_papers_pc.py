#!/usr/bin/env python3
"""PC runner for the updated Pi passing-papers logic using Ultralytics .pt models."""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from front_node_pc_common import (
    CV_WINDOW_PORT_HINT,
    POSE_MODEL_CANDIDATES,
    SCRIPT_DIR,
    UltralyticsPoseEstimator,
    close_cv_window,
    enable_cv_window_stream,
    load_pi_module,
    resolve_model_path,
)


pass_mod = load_pi_module("front_node_passing_papers_pi")
pass_mod.EVIDENCE_DIR = SCRIPT_DIR / "evidence_passing"


def _select_video(video_arg):
    if video_arg:
        return video_arg
    pass_mod.log_info("Opening file dialog...")
    return pass_mod.select_video_dialog()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AISENTINEL - Passing Papers Detection (PC + Ultralytics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python front_node_passing_papers_pc.py
  python front_node_passing_papers_pc.py --video path/to/exam.mp4
  python front_node_passing_papers_pc.py --pose-model models/archive/yolo26s-pose.pt
  python front_node_passing_papers_pc.py --no-roi
        """,
    )
    parser.add_argument("--video", default=None, help="Optional path to a video file")
    parser.add_argument(
        "--pose-model",
        default=None,
        help="Path/name for an Ultralytics pose .pt model",
    )
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--confidence", type=float, default=0.5, help="Person confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. cpu, 0")
    parser.add_argument("--no-roi", action="store_true", help="Skip ROI calibration")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  AISENTINEL - Passing Papers Detection (PC + Ultralytics)")
    print("  Logic      : tests_on_pi front_node_passing_papers_pi")
    print("  Hardware   : PC only, no Hailo")
    print("=" * 70)
    print()

    video_path = _select_video(args.video)
    if not video_path:
        pass_mod.log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{pass_mod.TC.RED}[ERROR] File not found: {video_path}{pass_mod.TC.RESET}")
        sys.exit(1)

    pose_model = resolve_model_path(
        args.pose_model,
        POSE_MODEL_CANDIDATES,
        fallback_name="yolo11n-pose.pt",
    )
    pass_mod.log_info(f"Loading pose model: {pose_model}")
    estimator = UltralyticsPoseEstimator(
        pose_model,
        conf_threshold=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{pass_mod.TC.RED}[ERROR] Cannot open video: {video_path}{pass_mod.TC.RESET}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{pass_mod.TC.RED}[ERROR] Cannot read first frame.{pass_mod.TC.RESET}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    pass_mod.log_info(f"Video resolution: {width}x{height}")

    roi_polygon = None
    if not args.no_roi:
        pass_mod.log_info("Draw ROI boundary to limit tracking area, or press S to skip.")
        roi_result = pass_mod.calibrate_roi(first_frame, disp_scale)
        if isinstance(roi_result, str) and roi_result == "CANCEL":
            cap.release()
            pass_mod.log_info("ROI calibration cancelled. Exiting.")
            sys.exit(0)
        roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    pass_mod.log_info("Running pose detection on first frame for student assignment...")
    first_detections = estimator.detect_pose(first_frame)
    first_detections = pass_mod.filter_detections_by_roi(first_detections, roi_polygon)
    tracker = pass_mod.IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = tracker.update(first_detections)

    roi_label = " within ROI" if roi_polygon is not None else ""
    pass_mod.log_info(f"Detected {len(first_detections)} persons{roi_label}.")

    print()
    print(f"  {pass_mod.TC.BOLD}Instructions:{pass_mod.TC.RESET}")
    print("    1. Click on a person to select them")
    print("    2. Type the student number")
    print("    3. Press ENTER to assign")
    print("    4. Assign at least 2 students, then press S to start")
    print()

    student_map = pass_mod.run_assignment_phase(
        first_frame, first_detections, first_track_ids, disp_scale
    )
    if student_map is None:
        cap.release()
        pass_mod.log_info("Assignment cancelled. Exiting.")
        sys.exit(0)
    if len(student_map) < 2:
        cap.release()
        pass_mod.log_info("Need at least 2 students for passing-papers detection. Exiting.")
        sys.exit(0)

    tracker.keep_only(set(student_map.keys()))
    pass_mod.log_info(f"Tracker locked to {len(student_map)} assigned student(s).")

    window_name = enable_cv_window_stream(pass_mod, "AISENTINEL - Passing Papers")
    pass_mod.log_info(f"OpenCV window: {window_name} (press Q or Esc to stop)")
    pass_mod.log_info("Starting detection...")
    try:
        pass_mod.run_detection(
            cap,
            estimator,
            tracker,
            student_map,
            video_path,
            CV_WINDOW_PORT_HINT,
            roi_polygon=roi_polygon,
        )
    finally:
        close_cv_window(window_name)
        cap.release()
        estimator.close()
    pass_mod.log_info("Done!")


if __name__ == "__main__":
    main()
