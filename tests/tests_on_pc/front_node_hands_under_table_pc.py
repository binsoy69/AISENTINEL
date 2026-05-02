#!/usr/bin/env python3
"""PC runner for the updated Pi hands-under-the-table logic using Ultralytics .pt models."""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from front_node_pc_common import (
    CV_WINDOW_PORT_HINT,
    HAND_MODEL_CANDIDATES,
    POSE_MODEL_CANDIDATES,
    SCRIPT_DIR,
    UltralyticsObjectDetector,
    UltralyticsPoseEstimator,
    close_cv_window,
    enable_cv_window_stream,
    load_pi_module,
    resolve_model_path,
)


hands_mod = load_pi_module("front_node_hands_under_table_pi")
hands_mod.EVIDENCE_DIR = SCRIPT_DIR / "evidence_hands"


def _select_video(video_arg):
    if video_arg:
        return video_arg
    hands_mod.log_info("Opening file dialog...")
    return hands_mod.select_video_dialog()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AISENTINEL - Hands Under the Table Detection (PC + Ultralytics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python front_node_hands_under_table_pc.py
  python front_node_hands_under_table_pc.py --video path/to/exam.mp4
  python front_node_hands_under_table_pc.py --pose-model models/archive/yolo26s-pose.pt
  python front_node_hands_under_table_pc.py --hand-model models/archive/yolov11n-sentinel-new/sentinel_new.pt
        """,
    )
    parser.add_argument("--video", default=None, help="Optional path to a video file")
    parser.add_argument("--pose-model", default=None, help="Path/name for an Ultralytics pose .pt model")
    parser.add_argument("--hand-model", default=None, help="Path to a .pt detector with a hand class")
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--person-confidence", type=float, default=hands_mod.PERSON_CONFIDENCE)
    parser.add_argument("--hand-confidence", type=float, default=hands_mod.HAND_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. cpu, 0")
    parser.add_argument("--no-roi", action="store_true", help="Skip ROI calibration")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  AISENTINEL - Hands Under the Table Detection (PC + Ultralytics)")
    print("  Logic      : tests_on_pi front_node_hands_under_table_pi")
    print("  Hardware   : PC only, no Hailo")
    print("=" * 70)
    print()

    video_path = _select_video(args.video)
    if not video_path:
        hands_mod.log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{hands_mod.TC.RED}[ERROR] File not found: {video_path}{hands_mod.TC.RESET}")
        sys.exit(1)

    pose_model = resolve_model_path(
        args.pose_model,
        POSE_MODEL_CANDIDATES,
        fallback_name="yolo11n-pose.pt",
    )
    try:
        hand_model = resolve_model_path(
            args.hand_model,
            HAND_MODEL_CANDIDATES,
        )
    except FileNotFoundError as exc:
        print(f"{hands_mod.TC.RED}[ERROR] {exc}{hands_mod.TC.RESET}")
        sys.exit(1)

    hands_mod.log_info(f"Loading pose model: {pose_model}")
    person_detector = UltralyticsPoseEstimator(
        pose_model,
        conf_threshold=args.person_confidence,
        imgsz=args.imgsz,
        device=args.device,
    )

    hands_mod.log_info(f"Loading hand detector: {hand_model}")
    hand_detector = UltralyticsObjectDetector(
        hand_model,
        conf_threshold=args.hand_confidence,
        imgsz=args.imgsz,
        device=args.device,
    )

    hand_classes = ", ".join(f"{idx}:{name}" for idx, name in sorted(hand_detector.names.items()))
    hands_mod.log_info(f"Detector classes: {hand_classes}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{hands_mod.TC.RED}[ERROR] Cannot open video: {video_path}{hands_mod.TC.RESET}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{hands_mod.TC.RED}[ERROR] Cannot read first frame.{hands_mod.TC.RESET}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    hands_mod.log_info(f"Video resolution: {width}x{height}")

    roi_polygon = None
    if not args.no_roi:
        hands_mod.log_info("Draw ROI boundary to limit tracking area, or press S to skip.")
        roi_result = hands_mod.calibrate_roi(first_frame, disp_scale)
        if isinstance(roi_result, str) and roi_result == "CANCEL":
            cap.release()
            hands_mod.log_info("ROI calibration cancelled. Exiting.")
            sys.exit(0)
        roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    hands_mod.log_info("Running pose detection on first frame for student assignment...")
    first_detections = person_detector.detect_persons(first_frame)
    first_detections = hands_mod.filter_detections_by_roi(first_detections, roi_polygon)
    tracker = hands_mod.IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = tracker.update(first_detections)
    hands_mod.log_info(f"Detected {len(first_detections)} persons on first frame.")

    print()
    print(f"  {hands_mod.TC.BOLD}Instructions:{hands_mod.TC.RESET}")
    print("    1. Click on a person to select them")
    print("    2. Type the student number")
    print("    3. Press ENTER to assign")
    print("    4. Repeat, then press S to start")
    print("    5. Draw one table-edge line per assigned student")
    print()

    student_map = hands_mod.run_assignment_phase(
        first_frame, first_detections, first_track_ids, disp_scale
    )
    if student_map is None:
        cap.release()
        hands_mod.log_info("Assignment cancelled. Exiting.")
        sys.exit(0)
    if len(student_map) == 0:
        cap.release()
        hands_mod.log_info("No students assigned. Exiting.")
        sys.exit(0)

    tracker.keep_only(set(student_map.keys()))
    hands_mod.log_info(f"Tracker locked to {len(student_map)} assigned student(s).")

    assigned_students = hands_mod.build_assigned_student_list(
        first_detections, first_track_ids, student_map
    )
    if len(assigned_students) == 0:
        cap.release()
        hands_mod.log_info("No assigned students available for line calibration. Exiting.")
        sys.exit(0)

    hands_mod.log_info(
        "Draw one student-side table-edge line for each assigned student, or press S to skip."
    )
    student_lines = hands_mod.calibrate_table_edge_lines(first_frame, assigned_students)
    if student_lines is None:
        cap.release()
        hands_mod.log_info("Table-edge calibration cancelled. Exiting.")
        sys.exit(0)

    window_name = enable_cv_window_stream(hands_mod, "AISENTINEL - Hands Under the Table")
    hands_mod.log_info(f"OpenCV window: {window_name} (press Q or Esc to stop)")
    hands_mod.log_info("Starting detection...")
    try:
        hands_mod.run_detection(
            cap,
            person_detector,
            hand_detector,
            tracker,
            student_map,
            assigned_students,
            student_lines,
            video_path,
            CV_WINDOW_PORT_HINT,
            roi_polygon=roi_polygon,
        )
    finally:
        close_cv_window(window_name)
        cap.release()
        person_detector.close()
        hand_detector.close()
    hands_mod.log_info("Done!")


if __name__ == "__main__":
    main()
